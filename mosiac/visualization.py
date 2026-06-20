"""
Particle-flow visualization, rendered locally on the server (GPU-accelerated)
before the frames are streamed to the screen slaves.

The heavy rasterization (splatting + glow blur) runs on the GPU via PyTorch
(CUDA, or Apple-Silicon Metal/MPS), so it scales to high resolution cheaply.
If PyTorch is unavailable it falls back to a slower pure-CPU/OpenCV renderer.
"""

import numpy as np
import cv2

# --- adjust this ---------------------------------------------------------
# Render-resolution multiplier. The server picks a base size (long side ~960)
# from the screen layout; the particle field is rendered at (base*RESOLUTION_SCALE)
# and the slaves each warp their slice of it. With each physical screen ~2K and
# several screens in one shot, the shared field needs to be multiple-K wide:
#   4.0 -> 3840x2160 (4K): full 2K per screen for ~2 screens, ~17 fps on MPS
#   6.0 -> 5760x3240      : ~2K per screen for ~3 screens, but ~6-8 fps (heavier)
# Raise it for more/denser screens at the cost of stream frame-rate.
RESOLUTION_SCALE = 4.0
# -------------------------------------------------------------------------

# Brightness gains for the three composited layers (tuned for the look below).
_GAIN_GLOW, _GAIN_BODY, _GAIN_CORE = 26.0, 30.0, 42.0

try:
    import torch
    _DEVICE = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available()
               else "cpu")
except Exception:  # torch not installed -> CPU fallback renderer
    torch = None
    _DEVICE = "cpu"


def gpu_device() -> str:
    """Name of the device the visualization renders on ('mps' | 'cuda' | 'cpu')."""
    return _DEVICE if torch is not None else "cpu (numpy)"


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _gaussian_kernel(sigma, device):
    radius = max(1, int(round(sigma * 3)))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
    return k / k.sum(), radius


def _gauss(img, sigma):
    """Separable Gaussian blur of a (H, W) tensor (direct convolution)."""
    k, r = _gaussian_kernel(sigma, img.device)
    x = img.view(1, 1, *img.shape)
    x = torch.nn.functional.conv2d(x, k.view(1, 1, 1, -1), padding=(0, r))
    x = torch.nn.functional.conv2d(x, k.view(1, 1, -1, 1), padding=(r, 0))
    return x.view(*img.shape)


def _blur(img, sigma):
    """Gaussian blur; large (low-frequency) blurs run on a downsampled grid so
    the cost stays roughly constant as RESOLUTION_SCALE grows."""
    if sigma <= 6:
        return _gauss(img, sigma)
    F = torch.nn.functional
    down = min(8, max(2, int(round(sigma / 3))))
    small = F.avg_pool2d(img.view(1, 1, *img.shape), down)
    small = _gauss(small.view(*small.shape[2:]), sigma / down)
    up = F.interpolate(small.view(1, 1, *small.shape), size=img.shape,
                       mode="bilinear", align_corners=False)
    return up.view(*img.shape)


class ParticleFlow:
    def __init__(self, width, height, num_particles=800):
        sc = RESOLUTION_SCALE
        self.w = max(1, int(round(width * sc)))
        self.h = max(1, int(round(height * sc)))
        self.t = 0.0
        self.n = num_particles

        self.x = np.random.uniform(0, self.w, num_particles).astype(np.float32)
        self.y = np.random.uniform(0, self.h, num_particles).astype(np.float32)

        # sizes/speeds scale with resolution so the look is resolution-independent
        self.sizes = (np.random.uniform(2, 8, num_particles) * sc).astype(np.float32)
        self.brightness = np.random.uniform(0.4, 1.0, num_particles).astype(np.float32)
        self.speed = (np.random.uniform(0.3, 1.2, num_particles) * sc).astype(np.float32)
        self.drift = (np.random.uniform(-0.3, 0.3, num_particles) * sc).astype(np.float32)
        self.phase = np.random.uniform(0, 2 * np.pi, num_particles).astype(np.float32)

    def step(self):
        t = self.t
        self.y -= self.speed
        self.x += np.sin(t * 0.8 + self.phase) * self.drift + self.drift * 0.3
        self.y[self.y < 0] = self.h
        self.x[self.x < 0] += self.w
        self.x[self.x >= self.w] -= self.w
        self.t += 0.05

    # -- rendering ----------------------------------------------------------

    def render(self):
        if torch is not None:
            return self._render_gpu()
        return self._render_cpu()

    def _render_gpu(self):
        dev = _DEVICE
        H, W = self.h, self.w
        sc = RESOLUTION_SCALE

        x = torch.as_tensor(self.x, device=dev)
        y = torch.as_tensor(self.y, device=dev)
        b = torch.as_tensor(self.brightness, device=dev)
        s = torch.as_tensor(self.sizes, device=dev)

        xi = x.round().long().clamp_(0, W - 1)
        yi = y.round().long().clamp_(0, H - 1)
        idx = yi * W + xi

        if getattr(self, "_flat", None) is None:
            self._flat = torch.zeros(H * W, device=dev)

        def splat(weight):
            flat = self._flat.zero_()          # reuse buffer instead of re-allocating
            flat.scatter_add_(0, idx, weight)
            return flat.view(H, W)

        # three layers: wide soft glow, mid body, sharp bright core
        glow = _blur(splat(b * s * s), sigma=4.0 * sc)
        body = _blur(splat(b * s),     sigma=1.5 * sc)
        core = _blur(splat(b),         sigma=0.6 * sc)

        # colors in BGR (OpenCV order) — greenish-yellow particles
        c_glow = torch.tensor([20.0, 80.0, 60.0], device=dev).view(3, 1, 1)
        c_body = torch.tensor([80.0, 220.0, 200.0], device=dev).view(3, 1, 1)
        c_core = torch.tensor([180.0, 255.0, 240.0], device=dev).view(3, 1, 1)

        canvas = (glow * _GAIN_GLOW) * c_glow \
            + (body * _GAIN_BODY) * c_body \
            + (core * _GAIN_CORE) * c_core
        canvas = canvas.clamp(0, 255).to(torch.uint8)
        return canvas.permute(1, 2, 0).contiguous().cpu().numpy()

    def _render_cpu(self):
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for i in range(self.n):
            cx, cy = int(self.x[i]), int(self.y[i])
            r = int(self.sizes[i])
            bb = self.brightness[i]
            cv2.circle(canvas, (cx, cy), r,
                       (int(200 * bb), int(220 * bb), int(80 * bb)), -1, cv2.LINE_AA)
            if r > 3:
                cv2.circle(canvas, (cx, cy), r * 3,
                           (int(60 * bb), int(80 * bb), int(20 * bb)), -1, cv2.LINE_AA)
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=4, sigmaY=4)
        for i in range(self.n):
            cx, cy = int(self.x[i]), int(self.y[i])
            r = max(1, int(self.sizes[i] * 0.5))
            bb = self.brightness[i]
            cv2.circle(canvas, (cx, cy), r,
                       (int(240 * bb), int(255 * bb), int(180 * bb)), -1, cv2.LINE_AA)
        return canvas


if __name__ == "__main__":
    print("Rendering on:", gpu_device())
    sim = ParticleFlow(640, 360, num_particles=600)
    print("Particle flow running. Press Q to quit.")
    while True:
        sim.step()
        frame = sim.render()
        cv2.imshow("Particle Flow", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
