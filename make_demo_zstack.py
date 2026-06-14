# -*- coding: utf-8 -*-
"""生成用于测试的共聚焦风格 Z-stack demo（更贴近真实数据）。

场景：一块带倾斜的基底(substrate) + 一个圆形 mark（孔的亮边缘）。
- 基底：整幅 FOV 都有较暗但真实的反射信号，焦面层随 X 线性变化（制造倾斜，可被 Rx/Ry 量出）。
- mark：白色亮环，焦面比基底高出 RING_DELTA 层（作为高于基准平面的特征）。
- 每个像素的亮度沿 Z 呈高斯峰，离焦变暗变糊，叠加轻微 PMT 噪声。

这样：背景(基底)逐网格高斯拟合得到基准平面 -> Rx/Ry；mark 作为特征显示在基准之上，不会被当噪声滤掉。
"""
import os
import numpy as np
import cv2

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_zstack")
N = 10                  # 层数
H = W = 512             # 图像尺寸
CX, CY = 256, 256       # mark 中心
R = 150                 # 圆环半径
RING = 3.0              # 环厚度（像素，sigma）
MU = 4.5               # 基底基准焦面层（0-based）
SIGMA_Z = 1.8          # 亮度沿 Z 的高斯宽度
TILT_LAYERS = 4.0      # 基底焦面层跨整幅 X 的漂移量（制造倾斜 -> Ry）
RING_DELTA = 1.5       # mark 焦面比基底高出的层数（高于基准平面的特征）
SUBSTRATE = 32.0       # 基底反射亮度（较暗但有信号）
RING_BRIGHT = 255.0    # mark 亮环亮度
SEED = 20240614


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dist = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
    ring = np.exp(-((dist - R) ** 2) / (2.0 * RING ** 2)).astype(np.float32)   # 锐利白环 0..1

    # 基底焦面层随 X 线性变化 -> 倾斜；mark 焦面 = 基底 + RING_DELTA
    mu_sub = MU + (xx - CX) / float(W) * TILT_LAYERS
    mu_ring = mu_sub + RING_DELTA

    for i in range(N):
        amp_sub = np.exp(-((i - mu_sub) ** 2) / (2.0 * SIGMA_Z ** 2)).astype(np.float32)
        amp_ring = np.exp(-((i - mu_ring) ** 2) / (2.0 * SIGMA_Z ** 2)).astype(np.float32)
        layer = SUBSTRATE * amp_sub + RING_BRIGHT * ring * amp_ring

        # 离焦模糊（按到基底焦面的距离）
        defocus = abs(i - MU)
        blur_sigma = 0.6 + defocus * 0.8
        k = int(max(3, round(blur_sigma * 4) | 1))
        layer = cv2.GaussianBlur(layer, (k, k), blur_sigma)

        # 轻微 PMT/散粒噪声
        noise = rng.normal(1.5, 1.5, size=layer.shape).astype(np.float32)
        layer = np.clip(layer + noise, 0, 255).astype(np.uint8)

        path = os.path.join(OUT_DIR, f"z{i+1:02d}.png")
        cv2.imwrite(path, layer)
        print("wrote", path, "max=", int(layer.max()), "substrate_focus_max=", int((SUBSTRATE * amp_sub).max()))

    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
