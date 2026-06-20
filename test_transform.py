import cv2
import numpy as np
import sys
sys.path.insert(0, '.')
from shared.transform import warp_master_to_slave

master = np.zeros((1080, 1920, 3), dtype=np.uint8)
master[:, :640] = [255, 0, 0]
master[:, 640:1280] = [0, 255, 0]
master[:, 1280:] = [0, 0, 255]

# Change these corners to test different slices:
# Left third:   [[0.0, 0.0], [0.333, 0.0], [0.333, 1.0], [0.0, 1.0]]
# Middle third: [[0.333, 0.0], [0.666, 0.0], [0.666, 1.0], [0.333, 1.0]]
# Right third:  [[0.666, 0.0], [1.0, 0.0], [1.0, 1.0], [0.666, 1.0]]
corners_uv = [[0.333, 0.0], [0.666, 0.0], [0.666, 1.0], [0.333, 1.0]]

result = warp_master_to_slave(master, corners_uv, 960, 540)
cv2.imshow('Transform Test', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
