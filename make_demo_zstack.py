# -*- coding: utf-8 -*-
"""生成用于测试的共聚焦风格 Z-stack demo。

模拟真实共聚焦场景：黑背景 + 黑色 mark（圆孔），孔的边缘是一圈白色亮环。
亮环亮度沿 Z 呈高斯峰（焦面最亮），离焦时变暗且模糊；
并加入很小的 X 向焦面倾斜，让 FOV 高度图能拟合出一个带 Rx/Ry 的平面。
"""
import os
import numpy as np
import cv2

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_zstack")
N = 10                 # 层数
H = W = 512            # 图像尺寸
CX, CY = 256, 256      # mark 中心
R = 150                # 圆环半径
RING = 4.0             # 环厚度（像素，sigma）
MU = 4.5               # 最佳焦面层（0-based，落在第 5/6 层之间）
SIGMA_Z = 1.8          # 亮度沿 Z 的高斯宽度
TILT_LAYERS = 1.2      # 跨整幅 X 方向焦面层漂移量（制造倾斜）
SEED = 20240614


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dist = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
    # 锐利白环（高斯环），范围 0..1
    ring = np.exp(-((dist - R) ** 2) / (2.0 * RING ** 2)).astype(np.float32)

    # 每个像素的最佳焦面层：随 X 线性变化 -> 焦面倾斜
    mu_field = MU + (xx - CX) / float(W) * TILT_LAYERS

    for i in range(N):
        amp = np.exp(-((i - mu_field) ** 2) / (2.0 * SIGMA_Z ** 2)).astype(np.float32)
        layer = 255.0 * ring * amp

        # 离焦模糊：离最佳焦面越远越糊
        defocus = abs(i - MU)
        blur_sigma = 0.6 + defocus * 1.1
        k = int(max(3, round(blur_sigma * 4) | 1))  # 奇数核
        layer = cv2.GaussianBlur(layer, (k, k), blur_sigma)

        # 轻微 PMT/散粒噪声 + 微弱背景
        noise = rng.normal(2.0, 2.0, size=layer.shape).astype(np.float32)
        layer = np.clip(layer + noise, 0, 255).astype(np.uint8)

        path = os.path.join(OUT_DIR, f"z{i+1:02d}.png")
        cv2.imwrite(path, layer)
        print("wrote", path, "max=", int(layer.max()))

    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
