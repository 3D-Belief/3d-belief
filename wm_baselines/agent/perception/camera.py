import numpy as np

class Camera(object):
    def __init__(self, fx, fy, cx, cy, near=0.1, far=100, w=64, h=64):
        self.fx, self.fy, self.cx, self.cy, self.near, self.far = fx, fy, cx, cy, near, far
        self.h = h
        self.w = w

        self.intrinsics = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32,
        )

        self.w2c_mat = np.eye(4)
        self.c2w_mat = np.eye(4)