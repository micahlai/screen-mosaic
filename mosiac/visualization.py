import numpy as np
import cv2

class ParticleFlow:
    def __init__(self, width, height, num_particles=800):
        self.w = width
        self.h = height
        self.t = 0.0
        self.n = num_particles

        # Random starting positions
        self.x = np.random.uniform(0, width, num_particles).astype(np.float32)
        self.y = np.random.uniform(0, height, num_particles).astype(np.float32)

        # Random sizes and brightness
        self.sizes = np.random.uniform(2, 8, num_particles).astype(np.float32)
        self.brightness = np.random.uniform(0.4, 1.0, num_particles).astype(np.float32)

        # Each particle gets a slightly different speed/drift
        self.speed = np.random.uniform(0.3, 1.2, num_particles).astype(np.float32)
        self.drift = np.random.uniform(-0.3, 0.3, num_particles).astype(np.float32)

        # Phase offset so particles don't all move in sync
        self.phase = np.random.uniform(0, 2 * np.pi, num_particles).astype(np.float32)

    def step(self):
        t = self.t

        # Drift upward with gentle horizontal sway
        self.y -= self.speed
        self.x += np.sin(t * 0.8 + self.phase) * self.drift + self.drift * 0.3

        # Wrap around edges
        self.y[self.y < 0] = self.h
        self.x[self.x < 0] += self.w
        self.x[self.x >= self.w] -= self.w

        self.t += 0.05

    def render(self):
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        for i in range(self.n):
            cx = int(self.x[i])
            cy = int(self.y[i])
            r = int(self.sizes[i])
            b = self.brightness[i]

            color = (int(200 * b), int(220 * b), int(80 * b))
            cv2.circle(canvas, (cx, cy), r, color, -1, lineType=cv2.LINE_AA)

            if r > 3:
                glow_color = (int(60 * b), int(80 * b), int(20 * b))
                cv2.circle(canvas, (cx, cy), r * 3, glow_color, -1, lineType=cv2.LINE_AA)

        canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=4, sigmaY=4)

        for i in range(self.n):
            cx = int(self.x[i])
            cy = int(self.y[i])
            r = max(1, int(self.sizes[i] * 0.5))
            b = self.brightness[i]
            color = (int(240 * b), int(255 * b), int(180 * b))
            cv2.circle(canvas, (cx, cy), r, color, -1, lineType=cv2.LINE_AA)

        return canvas


if __name__ == "__main__":
    W, H = 1280, 720
    sim = ParticleFlow(W, H, num_particles=600)

    print("Particle flow running. Press Q to quit.")
    while True:
        sim.step()
        frame = sim.render()
        cv2.imshow("Particle Flow", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
