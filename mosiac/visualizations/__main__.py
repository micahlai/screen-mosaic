"""Preview a visualization in a window:  python -m mosiac.visualizations [name]"""

import sys
import cv2

from . import available, create, gpu_device

name = sys.argv[1] if len(sys.argv) > 1 else "smoke"
print("Rendering", name, "on", gpu_device(),
      "| options:", [v["name"] for v in available()])
sim = create(name, 640, 360)
while True:
    sim.step()
    cv2.imshow(name, sim.render())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
