# -*- coding: utf-8 -*-
"""
Z-stack Focus Analyzer v2.6 Fast projection + precision Gaussian fusion
用于对 LSM / 显微镜 Z-stack 导出的图片序列进行焦面分析，并支持 ROI 内 mark 圆孔/方孔/矩形孔亚像素拟合，并显示识别边界。

功能：
1. 导入图片文件夹
2. 按文件名自然排序
3. 计算每层清晰度指标：Laplacian 方差、Tenengrad、Brenner、局部对比度、熵
4. 支持 ROI：全图 / 手动输入 ROI / 鼠标框选 ROI
5. 支持输入第一层 Z 值和 Z step，直接输出焦面绝对 Z 坐标
6. 支持清晰度评分法 / 清晰度高斯自动峰谷 / 共聚焦亮度高斯拟合法
7. 支持 ROI 内 mark 圆孔/方孔/矩形孔拟合，显示边界，输出 X、Y、R/D 或 W/H/角度的像素值和 μm 值
8. 新增 FOV 高度图：按网格从 Z-stack 反推各 XY 点焦面高度，生成 3D 地形图
9. 对高度图拟合平面，计算 Rx/Ry、去倾斜面型 PV/RMS 和原始 TTV
10. 支持高度图 ROI 内添加多个排除区域 Mask，排除点不参与平面拟合/PV/RMS/TTV
11. 高度图平面拟合支持 nσ 残差迭代滤波；Rx/Ry 按左手坐标系 X右/Y里/Z下修正符号
11. 输出最佳焦面层、曲线、相邻层对比图、CSV 结果

依赖：opencv-python, pillow, numpy, pandas, matplotlib
运行：python zstack_focus_analyzer.py
"""

import os
import re
import json
import math
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib import rcParams

# 解决 Windows 上 Matplotlib 中文标题/坐标轴乱码问题。
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def safe_read_gray(path: str, normalize_per_image: bool = True) -> np.ndarray:
    """支持中文路径读取，返回 float32 灰度图。

    normalize_per_image=True：每张图单独拉伸到 0-255，适合清晰度/纹理算法。
    normalize_per_image=False：保留原始灰度比例，适合共聚焦亮度-Z 曲线分析。
    """
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")

    if img.ndim == 3:
        # BGR/BGRA -> gray
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    if normalize_per_image:
        # 多数 LSM tiff 可能是 16bit；清晰度算法按每张图单独归一，避免动态范围差异影响梯度尺度。
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return img


def normalize_to_255(img: np.ndarray) -> np.ndarray:
    """把单张图归一化到 0-255，用于纹理/频域焦度和显示。"""
    img = img.astype(np.float32)
    lo, hi = float(np.min(img)), float(np.max(img))
    if hi > lo:
        return (img - lo) / (hi - lo) * 255.0
    return np.zeros_like(img, dtype=np.float32)


def resize_max_dim(img: np.ndarray, max_dim: int) -> np.ndarray:
    """按最大边长降采样。max_dim<=0 表示不降采样。"""
    if max_dim is None or int(max_dim) <= 0:
        return img
    max_dim = int(max_dim)
    h, w = img.shape[:2]
    cur = max(h, w)
    if cur <= max_dim:
        return img
    scale = max_dim / float(cur)
    new_w = max(8, int(round(w * scale)))
    new_h = max(8, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop_roi(img: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if roi is None:
        return img
    x, y, w, h = roi
    h_img, w_img = img.shape[:2]
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))
    return img[y:y + h, x:x + w]


def normalize_series(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def image_entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))



def calc_metrics(gray: np.ndarray) -> Dict[str, float]:
    """计算多种焦度指标。数值越大，通常越清晰。"""
    gray = gray.astype(np.float32)

    # 轻微高斯降噪，减少 PMT 噪声/散粒噪声对二阶算子的影响
    denoise = cv2.GaussianBlur(gray, (3, 3), 0)

    lap = cv2.Laplacian(denoise, cv2.CV_32F, ksize=3)
    lap_var = float(lap.var())

    gx = cv2.Sobel(denoise, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(denoise, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))

    # Brenner 梯度，适合单方向纹理明显时作为参考
    if gray.shape[1] > 2:
        brenner_x = np.mean((gray[:, 2:] - gray[:, :-2]) ** 2)
    else:
        brenner_x = 0.0
    if gray.shape[0] > 2:
        brenner_y = np.mean((gray[2:, :] - gray[:-2, :]) ** 2)
    else:
        brenner_y = 0.0
    brenner = float(brenner_x + brenner_y)

    # 局部对比度：排除整体亮度变化的一种参考
    mean = cv2.GaussianBlur(gray, (31, 31), 0)
    local_contrast = float(np.mean(np.abs(gray - mean)))

    ent = image_entropy(gray)
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))
    metrics = {
        "laplacian_var": lap_var,
        "tenengrad": tenengrad,
        "brenner": brenner,
        "local_contrast": local_contrast,
        "entropy": ent,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
    }
    return metrics



def gaussian_model(z: np.ndarray, baseline: float, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(abs(sigma)), 1e-12)
    return baseline + amplitude * np.exp(-((z - mu) ** 2) / (2.0 * sigma ** 2))


def estimate_gaussian_focus(z: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """用亮度-Z 曲线估计共聚焦焦面。

    优先使用 scipy.optimize.curve_fit；如果用户环境未安装 scipy，则退化为 log-Gaussian 二次拟合。
    返回 dict：mu, sigma, baseline, amplitude, fitted_y, method, r2, warning。
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(z)
    z, y = z[order], y[order]
    if len(z) < 3:
        raise ValueError("亮度高斯拟合至少需要 3 层 Z-stack 图片。")

    y_min, y_max = float(np.min(y)), float(np.max(y))
    if abs(y_max - y_min) < 1e-12:
        return {
            "mu": float(z[int(np.argmax(y))]), "sigma": np.nan, "baseline": y_min, "amplitude": 0.0,
            "fitted_y": np.full_like(y, y_min), "method": "无有效拟合", "r2": np.nan,
            "warning": "亮度曲线几乎没有变化，无法可靠用亮度拟合焦面。"
        }

    baseline0 = float(np.percentile(y, 10))
    amp0 = max(1e-9, y_max - baseline0)
    mu0 = float(z[int(np.argmax(y))])
    sigma0 = max(float((z.max() - z.min()) / 6.0), 1e-6)

    warning = ""
    try:
        from scipy.optimize import curve_fit  # type: ignore
        lower = [y_min - abs(y_max - y_min) * 2, 0.0, float(z.min()), 1e-9]
        upper = [y_max + abs(y_max - y_min) * 2, abs(y_max - y_min) * 10, float(z.max()), max(float(z.max() - z.min()) * 2, 1e-6)]
        popt, _ = curve_fit(
            gaussian_model, z, y,
            p0=[baseline0, amp0, mu0, sigma0],
            bounds=(lower, upper),
            maxfev=20000,
        )
        baseline, amplitude, mu, sigma = [float(v) for v in popt]
        fitted = gaussian_model(z, baseline, amplitude, mu, sigma)
        method = "scipy curve_fit"
    except Exception:
        # 退化算法：先扣背景，再对 ln(y-baseline) 做二次拟合。
        # ln(y-b)=C-(z-mu)^2/(2*sigma^2)=az^2+bz+c
        baseline = min(baseline0, y_min - 1e-6)
        yy = np.maximum(y - baseline, 1e-9)
        # 只取靠近峰值的点，减少远离焦面的噪声影响
        threshold = np.max(yy) * 0.2
        mask = yy >= threshold
        if np.sum(mask) < 3:
            mask = np.ones_like(yy, dtype=bool)
        coef = np.polyfit(z[mask], np.log(yy[mask]), 2)
        a, b, c = coef
        if a >= 0:
            mu = mu0
            sigma = sigma0
            amplitude = amp0
            fitted = gaussian_model(z, baseline0, amplitude, mu, sigma)
            method = "fallback peak"
            warning = "亮度曲线不符合标准高斯峰，已退化为最大亮度层附近估计。"
        else:
            mu = float(-b / (2 * a))
            mu = float(np.clip(mu, z.min(), z.max()))
            sigma = float(np.sqrt(-1.0 / (2.0 * a)))
            amplitude = float(np.exp(c - a * mu * mu))
            fitted = gaussian_model(z, baseline, amplitude, mu, sigma)
            method = "log-Gaussian fallback"
            warning = "未安装 scipy 或 curve_fit 失败，已使用内置 log-Gaussian 近似拟合。"

    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
    if not warning and np.isfinite(r2) and r2 < 0.75:
        warning = "高斯拟合 R² 偏低，可能存在饱和、漂白、多层结构或导出图片自动拉伸，建议结合清晰度法/ROI 复核。"

    return {
        "mu": float(mu), "sigma": float(abs(sigma)), "baseline": float(baseline), "amplitude": float(amplitude),
        "fitted_y": np.asarray(fitted, dtype=float), "method": method, "r2": r2, "warning": warning,
    }


def estimate_focus_metric_fit(z: np.ndarray, y: np.ndarray, allow_valley: bool = True) -> Dict[str, object]:
    """对任意焦度曲线做 Z 向高斯峰/谷拟合。

    普通清晰度指标理论上常见为“焦面处峰值”。但 LSM 数据中，若饱和、噪声、背景或算法方向导致
    焦面表现为谷值，则用 baseline + amplitude*Gaussian 且允许 amplitude 为负，可以自动拟合“谷”。
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(z)
    z, y = z[order], y[order]
    if len(z) < 4:
        raise ValueError("清晰度高斯拟合至少建议 4 层以上；层数太少时只能取最大/最小层。")

    y_min, y_max = float(np.min(y)), float(np.max(y))
    y_range = y_max - y_min
    if abs(y_range) < 1e-12:
        return {
            "mu": float(z[len(z)//2]), "sigma": np.nan, "baseline": y_min, "amplitude": 0.0,
            "fitted_y": np.full_like(y, y_min), "method": "无有效拟合", "r2": np.nan,
            "polarity": "flat", "warning": "焦度曲线几乎没有变化，无法可靠拟合焦面。"
        }

    candidates = []
    span = max(float(z.max() - z.min()), 1e-9)
    sigma0 = max(span / 6.0, 1e-6)

    def try_fit(polarity: str):
        # polarity=peak: amplitude>0；polarity=valley: amplitude<0
        if polarity == "peak":
            baseline0 = float(np.percentile(y, 10))
            amp0 = max(y_range, 1e-9)
            mu0 = float(z[int(np.argmax(y))])
            lower = [y_min - 2*y_range, 0.0, float(z.min()), 1e-9]
            upper = [y_max + 2*y_range, 10*y_range, float(z.max()), max(2*span, 1e-6)]
        else:
            baseline0 = float(np.percentile(y, 90))
            amp0 = -max(y_range, 1e-9)
            mu0 = float(z[int(np.argmin(y))])
            lower = [y_min - 2*y_range, -10*y_range, float(z.min()), 1e-9]
            upper = [y_max + 2*y_range, 0.0, float(z.max()), max(2*span, 1e-6)]
        try:
            from scipy.optimize import curve_fit  # type: ignore
            popt, _ = curve_fit(
                gaussian_model, z, y,
                p0=[baseline0, amp0, mu0, sigma0],
                bounds=(lower, upper),
                maxfev=20000,
            )
            baseline, amplitude, mu, sigma = [float(v) for v in popt]
            fitted = gaussian_model(z, baseline, amplitude, mu, sigma)
            ss_res = float(np.sum((y - fitted) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
            candidates.append({
                "mu": float(mu), "sigma": float(abs(sigma)), "baseline": float(baseline),
                "amplitude": float(amplitude), "fitted_y": np.asarray(fitted, dtype=float),
                "method": "scipy curve_fit", "r2": r2, "polarity": polarity, "warning": ""
            })
        except Exception:
            pass

    try_fit("peak")
    if allow_valley:
        try_fit("valley")

    if candidates:
        best = max(candidates, key=lambda d: (-1e9 if not np.isfinite(float(d.get("r2", np.nan))) else float(d["r2"])))
        if np.isfinite(float(best.get("r2", np.nan))) and float(best["r2"]) < 0.70:
            best["warning"] = "拟合 R² 偏低，曲线可能不是单峰/单谷；建议缩小 ROI 或结合亮度高斯法复核。"
        if best["polarity"] == "valley":
            note = "已识别为谷值型焦度曲线：焦面对应评分最低点附近。"
            best["warning"] = (note + (" " + best["warning"] if best.get("warning") else ""))
        return best

    # 无 scipy 或拟合失败：退化为最大/最小值层，并用移动平滑后判断更像峰还是谷。
    yy = y.copy()
    if len(yy) >= 5:
        kernel = np.ones(3) / 3.0
        yy = np.convolve(yy, kernel, mode="same")
    peak_idx = int(np.argmax(yy))
    valley_idx = int(np.argmin(yy))
    # 简单启发式：哪个极值离曲线中位数更“尖”，就选哪个；允许谷值时才选 valley
    med = float(np.median(yy))
    peak_prom = float(yy[peak_idx] - med)
    valley_prom = float(med - yy[valley_idx])
    if allow_valley and valley_prom > peak_prom * 1.05:
        idx, polarity = valley_idx, "valley"
        warning = "未安装 scipy 或拟合失败，已退化为平滑曲线最小值；该曲线表现为谷值型。"
    else:
        idx, polarity = peak_idx, "peak"
        warning = "未安装 scipy 或拟合失败，已退化为平滑曲线最大值。"
    return {
        "mu": float(z[idx]), "sigma": np.nan, "baseline": float(np.mean(y)), "amplitude": float(y[idx]-np.mean(y)),
        "fitted_y": yy, "method": "fallback extremum", "r2": np.nan, "polarity": polarity, "warning": warning
    }




def refine_edge_points_subpixel(gray_roi: np.ndarray, points: np.ndarray, max_points: int = 900) -> np.ndarray:
    """沿局部梯度法线对二值轮廓点做亚像素边缘细化。

    思路：轮廓点来自阈值/形态学边界，坐标是整数；在每个点处计算 Sobel 梯度方向，
    沿法线方向采样灰度剖面，用局部最大灰度变化位置作为亚像素边缘点。
    这不是标准靶标 MTF 边缘算法，但对圆孔/方孔 mark 的边界圆拟合，比直接用整数轮廓点更稳。
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 8:
        return pts.astype(np.float64)
    img = normalize_to_255(gray_roi).astype(np.float32)
    if img.size == 0:
        return pts.astype(np.float64)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    h, w = img.shape[:2]

    if pts.shape[0] > max_points:
        step = max(1, int(math.ceil(pts.shape[0] / float(max_points))))
        pts_use = pts[::step].copy()
    else:
        pts_use = pts.copy()

    refined = []
    offsets = np.asarray([-1.5, -0.75, 0.0, 0.75, 1.5], dtype=np.float32)
    for x, y in pts_use:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 2 or yi < 2 or xi >= w - 2 or yi >= h - 2:
            refined.append((float(x), float(y)))
            continue
        gxn = float(gx[yi, xi])
        gyn = float(gy[yi, xi])
        norm = math.hypot(gxn, gyn)
        if norm < 1e-6:
            refined.append((float(x), float(y)))
            continue
        nx = gxn / norm
        ny = gyn / norm
        xs = (x + offsets * nx).astype(np.float32)
        ys = (y + offsets * ny).astype(np.float32)
        if np.any(xs < 0) or np.any(ys < 0) or np.any(xs > w - 1) or np.any(ys > h - 1):
            refined.append((float(x), float(y)))
            continue
        vals = cv2.remap(img, xs.reshape(1, -1), ys.reshape(1, -1), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101).ravel()
        # 边缘位置取相邻采样点灰度变化最大的区间中心；再用邻域导数抛物线做轻量细化。
        dv = np.abs(np.diff(vals))
        k = int(np.argmax(dv))
        off = float((offsets[k] + offsets[k + 1]) / 2.0)
        if 0 < k < len(dv) - 1:
            y0, y1, y2 = float(dv[k - 1]), float(dv[k]), float(dv[k + 1])
            denom = (y0 - 2 * y1 + y2)
            if abs(denom) > 1e-9:
                delta = 0.5 * (y0 - y2) / denom
                delta = max(-0.5, min(0.5, delta))
                off += float(delta * 0.75)
        refined.append((float(x + off * nx), float(y + off * ny)))
    return np.asarray(refined, dtype=np.float64)

def fit_circle_least_squares(points: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """对边缘点做代数最小二乘圆拟合，返回 cx, cy, r, rms。"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 8:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        a, b_coef, c = sol
        cx = -a / 2.0
        cy = -b_coef / 2.0
        rr = cx * cx + cy * cy - c
        if rr <= 0 or not np.isfinite(rr):
            return None
        r = float(np.sqrt(rr))
        residual = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
        rms = float(np.sqrt(np.mean(residual ** 2)))
        return float(cx), float(cy), r, rms
    except Exception:
        return None



def _prepare_mark_masks(gray_full: np.ndarray, roi: Tuple[int, int, int, int]):
    """ROI 内生成暗孔/亮孔二值轮廓候选。"""
    x0, y0, w, h = roi
    crop = crop_roi(gray_full, roi)
    if crop.size == 0 or min(crop.shape[:2]) < 12:
        raise ValueError("ROI 太小，无法识别 mark。建议 ROI 至少覆盖完整边界。")

    img8 = normalize_to_255(crop).astype(np.uint8)
    blur = cv2.GaussianBlur(img8, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    out = []
    for polarity_name, thresh_type in [("dark_hole", cv2.THRESH_BINARY_INV), ("bright_hole", cv2.THRESH_BINARY)]:
        try:
            _, mask = cv2.threshold(blur, 0, 255, thresh_type + cv2.THRESH_OTSU)
        except Exception:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out.append((polarity_name, blur, mask, contours))
    return out


def _contour_basic_filter(cnt, roi_area: float) -> Optional[Tuple[float, float, float]]:
    area = float(cv2.contourArea(cnt))
    if area < max(20.0, roi_area * 0.002) or area > roi_area * 0.95:
        return None
    perim = float(cv2.arcLength(cnt, True))
    if perim <= 1e-6:
        return None
    circularity = 4.0 * math.pi * area / (perim * perim)
    return area, perim, circularity


def detect_circle_in_roi(gray_full: np.ndarray, roi: Tuple[int, int, int, int]) -> Dict[str, object]:
    """在 ROI 内自动识别圆孔，支持暗孔/亮孔自动判断，并做亚像素圆拟合。"""
    x0, y0, w, h = roi
    candidates = []
    roi_area = float(w * h)
    for polarity_name, _blur, _mask, contours in _prepare_mark_masks(gray_full, roi):
        for cnt in contours:
            basic = _contour_basic_filter(cnt, roi_area)
            if basic is None:
                continue
            area, perim, circularity = basic
            if circularity < 0.45:
                continue
            pts = cnt.reshape(-1, 2).astype(np.float64)
            refined_pts = refine_edge_points_subpixel(_blur, pts)
            fit = fit_circle_least_squares(refined_pts)
            if fit is None:
                fit = fit_circle_least_squares(pts)
                refined_pts = pts
            if fit is None:
                continue
            cx, cy, r, rms = fit
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            if r < 3 or r > max(w, h):
                continue
            rms_norm = rms / max(r, 1e-9)
            score = circularity * math.sqrt(area) / (1.0 + 8.0 * rms_norm)
            candidates.append({
                "shape": "circle", "polarity": polarity_name,
                "cx_roi": cx, "cy_roi": cy, "r_px": r, "rms_px": rms,
                "area_px2": area, "circularity": circularity, "score": score,
                "boundary_points_roi": refined_pts,
                "raw_boundary_points_roi": pts,
                "edge_fit": "subpixel_gradient_refined",
            })

    if not candidates:
        # 兜底：Canny 边缘点 + Hough 圆候选，再最小二乘 refine。
        crop = crop_roi(gray_full, roi)
        blur = cv2.GaussianBlur(normalize_to_255(crop).astype(np.uint8), (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(10, min(w, h) // 3),
            param1=100, param2=15, minRadius=3, maxRadius=max(4, min(w, h) // 2)
        )
        if circles is not None and len(circles) > 0:
            c = np.asarray(circles[0][0], dtype=float)
            cx, cy, r = float(c[0]), float(c[1]), float(c[2])
            yy, xx = np.where(edges > 0)
            dist = np.abs(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - r)
            keep = dist < max(2.5, r * 0.12)
            pts = np.column_stack([xx[keep], yy[keep]]).astype(np.float64)
            refined_pts = refine_edge_points_subpixel(blur, pts)
            fit = fit_circle_least_squares(refined_pts)
            if fit:
                cx, cy, r, rms = fit
                pts = refined_pts
            else:
                rms = float("nan")
            candidates.append({
                "shape": "circle", "polarity": "edge_hough", "cx_roi": cx, "cy_roi": cy, "r_px": r, "rms_px": rms,
                "area_px2": math.pi * r * r, "circularity": float("nan"), "score": 0.1,
                "boundary_points_roi": pts, "edge_fit": "subpixel_gradient_refined_hough",
            })

    if not candidates:
        raise ValueError("没有在 ROI 内找到可靠圆孔。请确认 ROI 包含完整圆边缘，或调整到对比度更明显的层。")

    best = max(candidates, key=lambda d: float(d.get("score", 0.0)))
    best["x_px"] = float(x0 + best["cx_roi"])
    best["y_px"] = float(y0 + best["cy_roi"])
    best["d_px"] = float(2.0 * best["r_px"])
    best["roi_x"] = float(x0)
    best["roi_y"] = float(y0)
    best["roi_w"] = float(w)
    best["roi_h"] = float(h)
    return best


def detect_rectangle_in_roi(gray_full: np.ndarray, roi: Tuple[int, int, int, int], require_square: bool = False) -> Dict[str, object]:
    """在 ROI 内识别方孔/矩形孔，返回中心、宽高、角度和边界点。"""
    x0, y0, w, h = roi
    roi_area = float(w * h)
    candidates = []
    for polarity_name, _blur, _mask, contours in _prepare_mark_masks(gray_full, roi):
        for cnt in contours:
            basic = _contour_basic_filter(cnt, roi_area)
            if basic is None:
                continue
            area, perim, circularity = basic
            if len(cnt) < 4:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            rw, rh = float(rw), float(rh)
            if rw < 3 or rh < 3:
                continue
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            short = max(1e-9, min(rw, rh))
            long = max(rw, rh)
            aspect = long / short
            if require_square and aspect > 1.30:
                continue
            rect_area = max(rw * rh, 1e-9)
            extent = max(0.0, min(1.0, area / rect_area))
            approx = cv2.approxPolyDP(cnt, 0.02 * perim, True)
            vertex_score = 1.0 / (1.0 + abs(len(approx) - 4))
            aspect_score = 1.0 / aspect if require_square else min(1.0, aspect / 1.2)
            score = math.sqrt(area) * extent * vertex_score * aspect_score
            box = cv2.boxPoints(rect).astype(np.float64)
            pts = cnt.reshape(-1, 2).astype(np.float64)
            candidates.append({
                "shape": "square" if require_square else "rectangle",
                "polarity": polarity_name,
                "cx_roi": float(cx), "cy_roi": float(cy),
                "width_px": rw, "height_px": rh, "angle_deg": float(angle),
                "area_px2": area, "rect_area_px2": rect_area, "extent": extent,
                "aspect_ratio": aspect, "score": score,
                "rect_corners_roi": box, "boundary_points_roi": pts,
            })
    if not candidates:
        name = "方孔" if require_square else "矩形/方孔"
        raise ValueError(f"没有在 ROI 内找到可靠{name}。请确认 ROI 包含完整边界，且孔与背景有明显灰度差。")
    best = max(candidates, key=lambda d: float(d.get("score", 0.0)))
    best["x_px"] = float(x0 + best["cx_roi"])
    best["y_px"] = float(y0 + best["cy_roi"])
    best["roi_x"] = float(x0)
    best["roi_y"] = float(y0)
    best["roi_w"] = float(w)
    best["roi_h"] = float(h)
    return best


def detect_mark_in_roi(gray_full: np.ndarray, roi: Tuple[int, int, int, int], mode: str = "自动识别") -> Dict[str, object]:
    """根据模式识别圆孔/方孔/矩形孔。自动模式会在圆和矩形候选间择优。"""
    if mode == "圆孔":
        return detect_circle_in_roi(gray_full, roi)
    if mode == "方孔":
        return detect_rectangle_in_roi(gray_full, roi, require_square=True)
    if mode == "矩形/方孔":
        return detect_rectangle_in_roi(gray_full, roi, require_square=False)

    errors = []
    candidates = []
    for fn in (detect_circle_in_roi, lambda img, r: detect_rectangle_in_roi(img, r, require_square=False)):
        try:
            candidates.append(fn(gray_full, roi))
        except Exception as e:
            errors.append(str(e))
    if not candidates:
        raise ValueError("自动识别失败：" + "；".join(errors))
    # 分数来自不同模型，先用稳健性规则：圆度高优先圆；矩形 extent/顶点吻合高优先矩形；最后比较归一化得分。
    def auto_score(d):
        base = float(d.get("score", 0.0))
        if d.get("shape") == "circle":
            circ = d.get("circularity", 0.0)
            try:
                circ = float(circ)
            except Exception:
                circ = 0.5
            return base * (1.2 if circ >= 0.72 else 0.8)
        extent = float(d.get("extent", 0.5))
        aspect = float(d.get("aspect_ratio", 1.0))
        return base * extent / max(1.0, aspect / 3.0)
    return max(candidates, key=auto_score)


# =========================
# FOV 高度图 / 面型分析工具函数
# =========================

def _parabolic_extremum_z(z: np.ndarray, y: np.ndarray, polarity: str = "peak") -> float:
    """三点二次曲线亚层估计。polarity=peak 取峰；valley 取谷。"""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(z)
    z, y = z[order], y[order]
    if len(z) == 0:
        return float("nan")
    yy = y if polarity == "peak" else -y
    idx = int(np.nanargmax(yy))
    if idx <= 0 or idx >= len(z) - 1:
        return float(z[idx])
    zz = z[idx-1:idx+2]
    vv = yy[idx-1:idx+2]
    try:
        a, b, _c = np.polyfit(zz, vv, 2)
        if a < 0:
            mu = -b / (2.0 * a)
            if float(zz.min()) <= mu <= float(zz.max()):
                return float(mu)
    except Exception:
        pass
    return float(z[idx])


def _fast_log_gaussian_focus_z(z: np.ndarray, y: np.ndarray, polarity: str = "peak") -> float:
    """快速 log-Gaussian 近似。用于高度图批量计算，失败时退化为三点抛物线。"""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(z)
    z, y = z[order], y[order]
    if len(z) < 4:
        return _parabolic_extremum_z(z, y, polarity)
    yy = y.copy()
    if polarity == "valley":
        yy = -yy
    finite = np.isfinite(yy)
    if np.count_nonzero(finite) < 4:
        return _parabolic_extremum_z(z, y, polarity)
    zz = z[finite]
    yy = yy[finite]
    rng = float(np.max(yy) - np.min(yy))
    if rng <= 1e-12:
        return float("nan")
    baseline = float(np.percentile(yy, 10)) - 1e-9 * max(1.0, abs(float(np.max(yy))))
    pos = yy - baseline
    pos = np.maximum(pos, 1e-12)
    thr = float(np.max(pos) * 0.25)
    mask = pos >= thr
    if np.count_nonzero(mask) < 4:
        # 三点拟合通常比强行全点 log 拟合更稳
        return _parabolic_extremum_z(z, y, polarity)
    try:
        a, b, _c = np.polyfit(zz[mask], np.log(pos[mask]), 2)
        if a < 0:
            mu = -b / (2.0 * a)
            if float(zz.min()) <= mu <= float(zz.max()):
                return float(mu)
    except Exception:
        pass
    return _parabolic_extremum_z(z, y, polarity)


def _true_gaussian_focus_z(z: np.ndarray, y: np.ndarray, polarity: str = "peak", source: str = "intensity") -> Tuple[float, str, float]:
    """对单个 XY 网格点的 Z 曲线做真实高斯拟合，返回 mu/method/r2。

    source=intensity 且 polarity=peak 时调用亮度高斯拟合；其余情况调用支持峰/谷的焦度高斯拟合。
    若 scipy 不可用或拟合失败，底层函数会退化，但 method/r2 会记录。
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(y)) or len(z) < 3:
        return float("nan"), "invalid", float("nan")
    if float(np.nanmax(y) - np.nanmin(y)) <= 1e-12:
        return float("nan"), "flat", float("nan")
    try:
        if source == "intensity" and polarity == "peak":
            fit = estimate_gaussian_focus(z, y)
        else:
            fit = estimate_focus_metric_fit(z, y, allow_valley=(polarity == "valley"))
        return float(fit.get("mu", float("nan"))), str(fit.get("method", "gaussian")), float(fit.get("r2", float("nan")))
    except Exception:
        return float("nan"), "fit_failed", float("nan")


def _score_image_for_topography(raw_crop: np.ndarray, mode: str) -> np.ndarray:
    """把单层图像转换为每个 XY 点/网格的 Z 向评分图。"""
    # 亮度投影 / 快速投影：直接用原始灰度做 Z 向亮度评分。
    # 共聚焦黑背景下，焦面对应亮度峰值，因此所有“快速投影/亮度”类算法都按亮度处理，
    # 而不是清晰度高频能量（修正历史上快速模式名义亮度、实际跑清晰度的问题）。
    if mode.startswith("共聚焦亮度") or mode.startswith("快速") or "亮度" in mode:
        return raw_crop.astype(np.float32)

    # 清晰度高度图：用局部高频能量图，而不是整块 variance。再对网格做面积平均。
    img = normalize_to_255(raw_crop).astype(np.float32)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    focus_energy = 0.45 * (lap * lap) + 0.55 * (gx * gx + gy * gy)
    return focus_energy.astype(np.float32)


def build_height_map_from_zstack(
    image_paths: List[str],
    z_values: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
    pixel_size_um: float,
    grid_px: int,
    mode: str,
    exclude_rois: Optional[List[Tuple[int, int, int, int]]] = None,
    sigma_filter: float = 3.0,
    progress_callback=None,
) -> Dict[str, object]:
    """从 Z-stack 反推网格高度图。

    输出：height_map[ny,nx]，x/y 中心坐标，平面拟合、残差和统计量。
    exclude_rois 使用原图像素坐标；被排除区域内的网格点会设置为 NaN，不参与拟合/统计。
    """
    if not image_paths:
        raise ValueError("未选择 Z-stack 图片。")
    if grid_px <= 1:
        raise ValueError("网格尺寸 grid_px 必须大于 1。")
    if pixel_size_um <= 0:
        raise ValueError("Pixel size 必须大于 0。")
    try:
        sigma_filter = float(sigma_filter)
    except Exception:
        sigma_filter = 0.0
    if not np.isfinite(sigma_filter) or sigma_filter < 0:
        sigma_filter = 0.0

    first = safe_read_gray(image_paths[0], normalize_per_image=False)
    h0, w0 = first.shape[:2]
    if roi is None:
        x0, y0, rw, rh = 0, 0, w0, h0
    else:
        x0, y0, rw, rh = roi
        x0 = max(0, min(int(x0), w0 - 1))
        y0 = max(0, min(int(y0), h0 - 1))
        rw = max(2, min(int(rw), w0 - x0))
        rh = max(2, min(int(rh), h0 - y0))

    nx = max(2, int(math.ceil(rw / float(grid_px))))
    ny = max(2, int(math.ceil(rh / float(grid_px))))
    z_values = np.asarray(z_values, dtype=np.float64)
    curves = np.full((len(image_paths), ny, nx), np.nan, dtype=np.float32)

    for zi, path in enumerate(image_paths):
        raw = safe_read_gray(path, normalize_per_image=False)
        crop = raw[y0:y0+rh, x0:x0+rw]
        score_img = _score_image_for_topography(crop, mode)
        # INTER_AREA 等价于按网格做面积平均，适合把 FOV 压成高度采样网格。
        small = cv2.resize(score_img, (nx, ny), interpolation=cv2.INTER_AREA)
        curves[zi, :, :] = small.astype(np.float32)
        if progress_callback and (zi % 2 == 0 or zi == len(image_paths) - 1):
            progress_callback(zi + 1, len(image_paths), 0, 0)

    height = np.full((ny, nx), np.nan, dtype=np.float64)
    polarity = "peak"
    use_gaussian = False
    if "谷值" in mode:
        polarity = "valley"
    if "高斯" in mode:
        use_gaussian = True

    # 曲线变化太小的点不参与拟合，防止背景区输出假高度。
    global_range = float(np.nanpercentile(curves, 99) - np.nanpercentile(curves, 1))
    min_range = max(global_range * 0.005, 1e-9)
    fit_method_map = np.full((ny, nx), "", dtype=object)
    fit_r2_map = np.full((ny, nx), np.nan, dtype=np.float64)
    fused_peak_map = np.full((ny, nx), np.nan, dtype=np.float64)
    fused_weighted_map = np.full((ny, nx), np.nan, dtype=np.float64)
    confidence_map = np.full((ny, nx), np.nan, dtype=np.float64)

    total_points = int(nx * ny)
    done_points = 0
    source = "intensity" if (mode.startswith("共聚焦亮度") or mode.startswith("快速")) else "clarity"

    # v2.6 快速路径：类似 Zeiss 最大亮度投影，核心是 numpy 向量化 max/argmax。
    # 这类算法主要用于快速观察/快速高度趋势，不做非线性高斯优化，因此会比逐点高斯快很多。
    fast_mode = mode.startswith("快速")
    if fast_mode:
        finite_curve = np.all(np.isfinite(curves), axis=0)
        curve_range = np.nanmax(curves, axis=0) - np.nanmin(curves, axis=0)
        valid_fast = finite_curve & (curve_range >= min_range)

        if "最大亮度投影" in mode or "峰值Z" in mode:
            idx_map = np.nanargmax(curves, axis=0)
            # np.take_along_axis 取每个网格点峰值灰度。
            peak_vals = np.take_along_axis(curves, idx_map[None, :, :], axis=0)[0].astype(np.float64)
            height[:, :] = z_values[idx_map]
            fused_peak_map[:, :] = peak_vals
            fused_weighted_map[:, :] = peak_vals
            confidence_map[:, :] = np.where(valid_fast, 0.65, np.nan)
            fit_method_map[:, :] = "fast_argmax"
            fit_r2_map[:, :] = np.nan

            if "抛物线" in mode:
                # 峰值附近 3 点二次插值，得到亚层 Z。仍比 scipy 高斯拟合快很多。
                for yy in range(ny):
                    for xx in range(nx):
                        done_points += 1
                        if valid_fast[yy, xx]:
                            c = curves[:, yy, xx].astype(np.float64)
                            height[yy, xx] = _parabolic_extremum_z(z_values, c, polarity="peak")
                            fit_method_map[yy, xx] = "fast_parabolic_peak"
                            confidence_map[yy, xx] = 0.75
                        if progress_callback and (done_points % max(1, total_points // 40) == 0 or done_points == total_points):
                            progress_callback(len(image_paths), len(image_paths), done_points, total_points)
            else:
                done_points = total_points
                if progress_callback:
                    progress_callback(len(image_paths), len(image_paths), done_points, total_points)

        height[~valid_fast] = np.nan
        fused_peak_map[~valid_fast] = np.nan
        fused_weighted_map[~valid_fast] = np.nan
        confidence_map[~valid_fast] = np.nan
    else:
        for yy in range(ny):
            for xx in range(nx):
                done_points += 1
                c = curves[:, yy, xx].astype(np.float64)
                try:
                    if np.all(np.isfinite(c)) and float(np.max(c) - np.min(c)) >= min_range:
                        if use_gaussian:
                            if source == "intensity" and polarity == "peak":
                                fit = estimate_gaussian_focus(z_values, c)
                                mu = float(fit.get("mu", float("nan")))
                                sigma = float(fit.get("sigma", float("nan")))
                                baseline = float(fit.get("baseline", float("nan")))
                                amp = float(fit.get("amplitude", float("nan")))
                                r2 = float(fit.get("r2", float("nan")))
                                method = str(fit.get("method", "gaussian"))
                                if np.isfinite(mu):
                                    height[yy, xx] = mu
                                    fit_method_map[yy, xx] = method
                                    fit_r2_map[yy, xx] = r2
                                    if np.isfinite(baseline) and np.isfinite(amp):
                                        fused_peak_map[yy, xx] = baseline + amp
                                    if np.isfinite(sigma) and sigma > 1e-9:
                                        sigma_w = max(sigma, float(np.median(np.diff(np.sort(z_values)))) if len(z_values) > 1 else sigma)
                                        weights = np.exp(-((z_values - mu) ** 2) / (2.0 * sigma_w ** 2))
                                        sw = float(np.sum(weights))
                                        if sw > 1e-12:
                                            fused_weighted_map[yy, xx] = float(np.sum(weights * c) / sw)
                                    else:
                                        fused_weighted_map[yy, xx] = c[int(np.nanargmax(c))]
                                    confidence_map[yy, xx] = max(0.0, min(1.0, r2 if np.isfinite(r2) else 0.0))
                            else:
                                mu, method, r2 = _true_gaussian_focus_z(z_values, c, polarity=polarity, source=source)
                                if np.isfinite(mu):
                                    height[yy, xx] = mu
                                    fit_method_map[yy, xx] = method
                                    fit_r2_map[yy, xx] = r2
                                    confidence_map[yy, xx] = max(0.0, min(1.0, r2 if np.isfinite(r2) else 0.0))
                        else:
                            height[yy, xx] = _parabolic_extremum_z(z_values, c, polarity=polarity)
                            fit_method_map[yy, xx] = "parabolic_extremum"
                            # 非高斯算法下，融合灰度退化为极值层网格灰度。
                            if source == "intensity":
                                idx_ext = int(np.nanargmax(c) if polarity == "peak" else np.nanargmin(c))
                                fused_peak_map[yy, xx] = float(c[idx_ext])
                                fused_weighted_map[yy, xx] = float(c[idx_ext])
                                confidence_map[yy, xx] = 0.5
                finally:
                    if progress_callback and (done_points % max(1, total_points // 50) == 0 or done_points == total_points):
                        progress_callback(len(image_paths), len(image_paths), done_points, total_points)

    # 网格中心坐标。坐标使用原图像素坐标乘 pixel size；平面斜率与 ROI 偏移无关。
    x_edges = np.linspace(x0, x0 + rw, nx + 1)
    y_edges = np.linspace(y0, y0 + rh, ny + 1)
    x_centers_px = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers_px = (y_edges[:-1] + y_edges[1:]) / 2.0
    x_um = x_centers_px * pixel_size_um
    y_um = y_centers_px * pixel_size_um
    xx_um, yy_um = np.meshgrid(x_um, y_um)
    x_px_grid, y_px_grid = np.meshgrid(x_centers_px, y_centers_px)

    # ROI 内排除区域：例如夹具、脏点、孔洞、异常反光区。
    # 这里按“网格中心点落入排除框”判定；grid 越小，排除边界越贴近实际。
    excluded_grid_mask = np.zeros_like(height, dtype=bool)
    normalized_exclude_rois = []
    if exclude_rois:
        for ex in exclude_rois:
            try:
                ex_x, ex_y, ex_w, ex_h = [int(round(v)) for v in ex]
            except Exception:
                continue
            if ex_w <= 0 or ex_h <= 0:
                continue
            ex_x0 = max(0, min(ex_x, w0 - 1))
            ex_y0 = max(0, min(ex_y, h0 - 1))
            ex_x1 = max(0, min(ex_x + ex_w, w0))
            ex_y1 = max(0, min(ex_y + ex_h, h0))
            if ex_x1 <= ex_x0 or ex_y1 <= ex_y0:
                continue
            normalized_exclude_rois.append((ex_x0, ex_y0, ex_x1 - ex_x0, ex_y1 - ex_y0))
            excluded_grid_mask |= (x_px_grid >= ex_x0) & (x_px_grid <= ex_x1) & (y_px_grid >= ex_y0) & (y_px_grid <= ex_y1)
        height[excluded_grid_mask] = np.nan
        fused_peak_map[excluded_grid_mask] = np.nan
        fused_weighted_map[excluded_grid_mask] = np.nan
        confidence_map[excluded_grid_mask] = np.nan

    base_mask = np.isfinite(height) & (~excluded_grid_mask)
    base_valid_count = int(np.count_nonzero(base_mask))
    if base_valid_count < 3:
        raise ValueError("有效高度点少于 3 个，无法拟合平面。请增大 ROI、增大网格尺寸，或换用亮度峰值算法。")

    def _fit_plane(current_mask: np.ndarray):
        n = int(np.count_nonzero(current_mask))
        A_local = np.column_stack([xx_um[current_mask], yy_um[current_mask], np.ones(n)])
        z_local = height[current_mask]
        coeff_local, *_ = np.linalg.lstsq(A_local, z_local, rcond=None)
        return [float(v) for v in coeff_local]

    # nσ 残差迭代滤波：先拟合平面，再按残差标准差剔除异常高度点，重新拟合。
    # sigma_filter <= 0 时关闭滤波。滤波后的 inlier_mask 同时用于平面、PV/RMS/TTV 统计。
    inlier_mask = base_mask.copy()
    sigma_iterations = 0
    residual_sigma_um = float("nan")
    residual_threshold_um = float("nan")
    if sigma_filter > 0:
        for _ in range(6):
            if int(np.count_nonzero(inlier_mask)) < 3:
                break
            a_tmp, b_tmp, c_tmp = _fit_plane(inlier_mask)
            plane_tmp = a_tmp * xx_um + b_tmp * yy_um + c_tmp
            resid_tmp = height - plane_tmp
            rv = resid_tmp[inlier_mask]
            sigma_um = float(np.nanstd(rv))
            residual_sigma_um = sigma_um
            if (not np.isfinite(sigma_um)) or sigma_um <= 1e-12:
                break
            threshold_um = float(sigma_filter * sigma_um)
            residual_threshold_um = threshold_um
            new_inlier_mask = base_mask & (np.abs(resid_tmp) <= threshold_um)
            if int(np.count_nonzero(new_inlier_mask)) < 3:
                break
            sigma_iterations += 1
            if np.array_equal(new_inlier_mask, inlier_mask):
                inlier_mask = new_inlier_mask
                break
            inlier_mask = new_inlier_mask

    valid_count = int(np.count_nonzero(inlier_mask))
    if valid_count < 3:
        raise ValueError("sigma 残差滤波后有效高度点少于 3 个，无法拟合平面。请增大 sigma 或关闭滤波。")

    a, b_slope, c0 = _fit_plane(inlier_mask)
    plane = a * xx_um + b_slope * yy_um + c0
    residual = height - plane
    residual_valid = residual[inlier_mask]
    height_valid = height[inlier_mask]
    sigma_outlier_mask = base_mask & (~inlier_mask)

    # 左手坐标系修正：X 向右、Y 向里、Z 向下。
    # 对平面 Z=aX+bY+c：Rx 对应绕 X 的倾角，Ry 对应绕 Y 的倾角。
    # 在该左手系下，Rx=atan(dZ/dY)，Ry=-atan(dZ/dX)。
    rx_rad = math.atan(b_slope)
    ry_rad = -math.atan(a)
    stats = {
        "valid_count": valid_count,
        "pre_filter_valid_count": base_valid_count,
        "total_count": int(nx * ny),
        "excluded_count": int(np.count_nonzero(excluded_grid_mask)),
        "sigma_outlier_count": int(np.count_nonzero(sigma_outlier_mask)),
        "valid_ratio": float(valid_count / float(nx * ny)),
        "z_min_um": float(np.nanmin(height_valid)),
        "z_max_um": float(np.nanmax(height_valid)),
        "z_mean_um": float(np.nanmean(height_valid)),
        "ttv_um": float(np.nanmax(height_valid) - np.nanmin(height_valid)),
        "pv_raw_um": float(np.nanmax(height_valid) - np.nanmin(height_valid)),
        "pv_residual_um": float(np.nanmax(residual_valid) - np.nanmin(residual_valid)),
        "rms_residual_um": float(np.sqrt(np.nanmean(residual_valid ** 2))),
        "rx_deg": float(math.degrees(rx_rad)),
        "ry_deg": float(math.degrees(ry_rad)),
        "rx_mrad": float(rx_rad * 1000.0),
        "ry_mrad": float(ry_rad * 1000.0),
        "slope_x_dzdx": a,
        "slope_y_dzdy": b_slope,
        "plane_c_um": c0,
        "sigma_filter": float(sigma_filter),
        "sigma_iterations": int(sigma_iterations),
        "residual_sigma_um": residual_sigma_um,
        "residual_threshold_um": residual_threshold_um,
        "coordinate_system": "左手系：X向右，Y向里，Z向下；Rx=atan(dZ/dY)，Ry=-atan(dZ/dX)",
        "grid_px": int(grid_px),
        "pixel_size_um": float(pixel_size_um),
        "mode": mode,
        "fit_r2_mean": float(np.nanmean(fit_r2_map[inlier_mask])) if np.any(np.isfinite(fit_r2_map[inlier_mask])) else float("nan"),
        "confidence_mean": float(np.nanmean(confidence_map[inlier_mask])) if np.any(np.isfinite(confidence_map[inlier_mask])) else float("nan"),
        "fused_gray_available": bool(np.any(np.isfinite(fused_weighted_map))),
        "roi": (int(x0), int(y0), int(rw), int(rh)),
        "exclude_rois": normalized_exclude_rois,
    }

    return {
        "height_map": height,
        "plane_map": plane,
        "residual_map": residual,
        "x_um": x_um,
        "y_um": y_um,
        "x_px": x_centers_px,
        "y_px": y_centers_px,
        "stats": stats,
        "curves": curves,
        "excluded_grid_mask": excluded_grid_mask,
        "base_valid_mask": base_mask,
        "inlier_mask": inlier_mask,
        "sigma_outlier_mask": sigma_outlier_mask,
        "fit_method_map": fit_method_map,
        "fit_r2_map": fit_r2_map,
        "fused_peak_map": fused_peak_map,
        "fused_weighted_map": fused_weighted_map,
        "confidence_map": confidence_map,
    }


@dataclass
class LayerResult:
    """单层 Z-stack 分析结果。"""
    index: int
    filename: str
    path: str
    metrics: Dict[str, float]
    score: float = 0.0
    z_um: float = 0.0


class ZStackFocusAnalyzer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Z-stack 焦面分析工具 v2.6 快速投影/精密高斯融合版")
        self.geometry("1500x920")
        self.minsize(1280, 820)

        self.folder: Optional[str] = None
        self.image_paths: List[str] = []
        self.results: List[LayerResult] = []
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self.exclude_rois: List[Tuple[int, int, int, int]] = []
        self.current_index: int = 0
        self.selector: Optional[RectangleSelector] = None
        self.selector_mode: str = "roi"

        self.metric_weights = {
            "laplacian_var": tk.DoubleVar(value=0.35),
            "tenengrad": tk.DoubleVar(value=0.35),
            "brenner": tk.DoubleVar(value=0.15),
            "local_contrast": tk.DoubleVar(value=0.10),
            "entropy": tk.DoubleVar(value=0.05),
        }
        self.first_z_um = tk.DoubleVar(value=0.0)
        self.z_step_um = tk.DoubleVar(value=1.0)
        self.algorithm_var = tk.StringVar(value="综合清晰度评分")
        self.use_roi_var = tk.BooleanVar(value=False)
        self.gaussian_fit: Optional[Dict[str, object]] = None
        self.focus_fit: Optional[Dict[str, object]] = None
        self.max_analysis_dim = tk.IntVar(value=1024)
        self.pixel_size_um = tk.DoubleVar(value=1.0)
        self.mark_result: Optional[Dict[str, object]] = None
        self.mark_drift_results: List[Dict[str, object]] = []
        self.mark_shape_var = tk.StringVar(value="自动识别")
        self.mark_progress_var = tk.StringVar(value="未进行逐层中心识别")
        self.mark_drift_ref_mode = tk.StringVar(value="最佳焦面")
        self.mark_drift_custom_layer = tk.IntVar(value=0)
        self.mark_drift_scope_mode = tk.StringVar(value="全部层")
        self.mark_drift_window_layers = tk.IntVar(value=3)
        self.mark_drift_axis_mode = tk.StringVar(value="局部自动缩放")
        self.is_mark_analyzing = False

        # 页面 3：FOV 高度图 / 面型分析
        self.topo_algorithm_var = tk.StringVar(value="快速最大亮度投影")
        self.topo_grid_px = tk.IntVar(value=32)
        self.topo_sigma_filter_var = tk.DoubleVar(value=3.0)
        self.topo_use_roi_var = tk.BooleanVar(value=False)
        self.topo_result: Optional[Dict[str, object]] = None
        self.topo_progress_var = tk.StringVar(value="未生成高度图")
        self.is_topo_analyzing = False

        self.progress_var = tk.StringVar(value="就绪")
        self.is_analyzing = False
        self.plot_mode = "standard"
        self._display_cache: Dict[int, np.ndarray] = {}

        self._setup_style()
        self._build_ui()


    def _setup_style(self):
        """轻量 UI 主题：只使用 tkinter/ttk 自带能力，不增加额外依赖。"""
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.colors = {
            "bg": "#F4F7FB",
            "panel": "#FFFFFF",
            "primary": "#2563EB",
            "primary_dark": "#1D4ED8",
            "success": "#16A34A",
            "success_dark": "#15803D",
            "warning": "#F59E0B",
            "warning_dark": "#D97706",
            "muted": "#64748B",
            "border": "#CBD5E1",
            "text": "#0F172A",
        }
        self.configure(bg=self.colors["bg"])

        self.style.configure("TFrame", background=self.colors["bg"])
        self.style.configure("Panel.TFrame", background=self.colors["panel"], relief="flat")
        self.style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"], font=("Microsoft YaHei", 9))
        self.style.configure("Title.TLabel", background=self.colors["bg"], foreground=self.colors["text"], font=("Microsoft YaHei", 16, "bold"))
        self.style.configure("Hint.TLabel", background=self.colors["bg"], foreground=self.colors["muted"], font=("Microsoft YaHei", 8))
        self.style.configure("TLabelframe", background=self.colors["bg"], bordercolor=self.colors["border"], relief="solid")
        self.style.configure("TLabelframe.Label", background=self.colors["bg"], foreground=self.colors["text"], font=("Microsoft YaHei", 10, "bold"))
        self.style.configure("TNotebook", background=self.colors["bg"], borderwidth=0)
        self.style.configure("TNotebook.Tab", padding=(12, 7), font=("Microsoft YaHei", 9, "bold"))
        self.style.map("TNotebook.Tab", background=[("selected", self.colors["panel"])], foreground=[("selected", self.colors["primary"])])

        self.style.configure("TButton", font=("Microsoft YaHei", 9), padding=(8, 5))
        self.style.configure("Primary.TButton", background=self.colors["primary"], foreground="white", font=("Microsoft YaHei", 10, "bold"), padding=(10, 7), borderwidth=0)
        self.style.map("Primary.TButton", background=[("active", self.colors["primary_dark"]), ("disabled", "#94A3B8")], foreground=[("disabled", "#E2E8F0")])
        self.style.configure("Success.TButton", background=self.colors["success"], foreground="white", font=("Microsoft YaHei", 10, "bold"), padding=(10, 7), borderwidth=0)
        self.style.map("Success.TButton", background=[("active", self.colors["success_dark"]), ("disabled", "#94A3B8")])
        self.style.configure("Warn.TButton", background=self.colors["warning"], foreground="white", font=("Microsoft YaHei", 9, "bold"), padding=(8, 5), borderwidth=0)
        self.style.map("Warn.TButton", background=[("active", self.colors["warning_dark"]), ("disabled", "#94A3B8")])
        self.style.configure("Ghost.TButton", background="#E2E8F0", foreground=self.colors["text"], padding=(8, 5), borderwidth=0)
        self.style.map("Ghost.TButton", background=[("active", "#CBD5E1")])

        self.style.configure("Treeview", font=("Microsoft YaHei", 8), rowheight=24, background="white", fieldbackground="white")
        self.style.configure("Treeview.Heading", font=("Microsoft YaHei", 8, "bold"), background="#E2E8F0", foreground=self.colors["text"])
        self.style.configure("Horizontal.TScale", background=self.colors["bg"])

    def _build_ui(self):
        main = ttk.Frame(self, style="TFrame")
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(main, width=400, style="TFrame")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)

        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 左侧控制区：拆成两个页面，避免把逐层分析按钮挤没
        ttk.Label(left, text="🔬 Z-stack 分析工具", style="Title.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Button(left, text="📁 选择图片文件夹", style="Primary.TButton", command=self.select_folder).pack(fill=tk.X, pady=3)
        self.folder_label = ttk.Label(left, text="未选择", wraplength=370)
        self.folder_label.pack(anchor="w", pady=(0, 8))

        self.left_tabs = ttk.Notebook(left)
        self.left_tabs.pack(fill=tk.BOTH, expand=True)
        focus_tab = ttk.Frame(self.left_tabs)
        mark_tab = ttk.Frame(self.left_tabs)
        topo_tab = ttk.Frame(self.left_tabs)
        self.left_tabs.add(focus_tab, text="📈 逐层焦面分析")
        self.left_tabs.add(mark_tab, text="🎯 ROI中心识别")
        self.left_tabs.add(topo_tab, text="🗺 FOV高度图")

        # ========== 页面 1：逐层焦面分析 ==========
        step_frame = ttk.LabelFrame(focus_tab, text="Z 方向参数")
        step_frame.pack(fill=tk.X, pady=6, padx=2)
        ttk.Label(step_frame, text="第一层 Z / μm：").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(step_frame, textvariable=self.first_z_um, width=10).grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(step_frame, text="Z step / μm：").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(step_frame, textvariable=self.z_step_um, width=10).grid(row=1, column=1, sticky="w", padx=6, pady=4)

        alg_frame = ttk.LabelFrame(focus_tab, text="焦面算法")
        alg_frame.pack(fill=tk.X, pady=6, padx=2)
        self.algorithm_combo = ttk.Combobox(
            alg_frame,
            textvariable=self.algorithm_var,
            values=("综合清晰度评分", "综合清晰度高斯拟合-自动峰谷", "综合清晰度反向谷值", "共聚焦亮度高斯拟合", "平均亮度最大值"),
            state="readonly",
            width=24,
        )
        self.algorithm_combo.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(
            alg_frame,
            text="清晰度若出现‘焦面最低’，建议选：综合清晰度高斯拟合-自动峰谷；共聚焦亮度法要求灰度不要被逐张自动增强。",
            wraplength=360,
        ).pack(anchor="w", padx=6, pady=(0, 6))
        speed_frame = ttk.Frame(alg_frame)
        speed_frame.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Label(speed_frame, text="分析最大边长 px：").pack(side=tk.LEFT)
        ttk.Entry(speed_frame, textvariable=self.max_analysis_dim, width=8).pack(side=tk.LEFT)
        ttk.Label(alg_frame, text="建议 1024；填 0 表示全分辨率但可能很慢。", wraplength=360).pack(anchor="w", padx=6, pady=(0, 6))

        roi_frame = ttk.LabelFrame(focus_tab, text="ROI 设置")
        roi_frame.pack(fill=tk.X, pady=6, padx=2)
        ttk.Checkbutton(roi_frame, text="使用 ROI 分析", variable=self.use_roi_var).pack(anchor="w", padx=6, pady=4)
        ttk.Button(roi_frame, text="▣ 在图像上框选 ROI", style="Warn.TButton", command=self.enable_roi_selector).pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(roi_frame, text="↺ 清除 ROI，使用全图", style="Ghost.TButton", command=self.clear_roi).pack(fill=tk.X, padx=6, pady=3)
        self.roi_label = ttk.Label(roi_frame, text="当前 ROI：全图", wraplength=360)
        self.roi_label.pack(anchor="w", padx=6, pady=4)

        weight_frame = ttk.LabelFrame(focus_tab, text="综合评分权重")
        weight_frame.pack(fill=tk.X, pady=6, padx=2)
        labels = {
            "laplacian_var": "Laplacian 方差",
            "tenengrad": "Tenengrad 梯度",
            "brenner": "Brenner 梯度",
            "local_contrast": "局部对比度",
            "entropy": "熵",
        }
        for i, key in enumerate(labels):
            ttk.Label(weight_frame, text=labels[key]).grid(row=i, column=0, sticky="w", padx=6, pady=3)
            ttk.Entry(weight_frame, textvariable=self.metric_weights[key], width=8).grid(row=i, column=1, sticky="e", padx=6, pady=3)

        self.analyze_button = ttk.Button(focus_tab, text="▶ 开始逐层分析 / 更新曲线", style="Success.TButton", command=self.analyze)
        self.analyze_button.pack(fill=tk.X, pady=(10, 4), padx=2)
        self.progress_label = ttk.Label(focus_tab, textvariable=self.progress_var, wraplength=360, style="Hint.TLabel")
        self.progress_label.pack(anchor="w", pady=(0, 4), padx=2)
        export_line = ttk.Frame(focus_tab)
        export_line.pack(fill=tk.X, pady=3, padx=2)
        ttk.Button(export_line, text="⬇ 导出 CSV", style="Ghost.TButton", command=self.export_csv).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        ttk.Button(export_line, text="🖼 导出报告图", style="Ghost.TButton", command=self.export_report_png).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        ttk.Label(
            focus_tab,
            text="分析结论已移到右下方「逐层分析结论」页，避免被左侧参数区挤压。",
            wraplength=360,
            style="Hint.TLabel",
        ).pack(anchor="w", pady=6, padx=4)

        # ========== 页面 2：ROI Mark 中心识别 ==========
        mark_intro = ttk.Label(
            mark_tab,
            text="流程：选择图片文件夹 → 滑块切到目标层 → 框选只包含一个完整 Mark 的 ROI → 选择形状 → 识别中心。",
            wraplength=360,
        )
        mark_intro.pack(anchor="w", padx=6, pady=(8, 6))

        mark_roi_frame = ttk.LabelFrame(mark_tab, text="ROI 设置")
        mark_roi_frame.pack(fill=tk.X, pady=6, padx=2)
        ttk.Button(mark_roi_frame, text="▣ 在图像上框选 ROI", style="Warn.TButton", command=self.enable_roi_selector).pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(mark_roi_frame, text="↺ 清除 ROI", style="Ghost.TButton", command=self.clear_roi).pack(fill=tk.X, padx=6, pady=3)
        self.roi_label_mark = ttk.Label(mark_roi_frame, text="当前 ROI：全图", wraplength=360)
        self.roi_label_mark.pack(anchor="w", padx=6, pady=4)

        mark_frame = ttk.LabelFrame(mark_tab, text="Mark孔 / 方孔 / 矩形孔拟合")
        mark_frame.pack(fill=tk.X, pady=6, padx=2)
        ps_line = ttk.Frame(mark_frame)
        ps_line.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(ps_line, text="Pixel size / μm：").pack(side=tk.LEFT)
        ttk.Entry(ps_line, textvariable=self.pixel_size_um, width=10).pack(side=tk.LEFT)
        shape_line = ttk.Frame(mark_frame)
        shape_line.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(shape_line, text="目标形状：").pack(side=tk.LEFT)
        ttk.Combobox(
            shape_line, textvariable=self.mark_shape_var,
            values=("自动识别", "圆孔", "方孔", "矩形/方孔"),
            state="readonly", width=12
        ).pack(side=tk.LEFT)
        ttk.Button(mark_frame, text="◎ 识别当前层 ROI Mark", style="Success.TButton", command=self.detect_mark_hole_current).pack(fill=tk.X, padx=6, pady=(6, 3))
        ttk.Button(mark_frame, text="📈 逐层识别 ROI 中心 / 漂移三图", style="Primary.TButton", command=self.analyze_mark_drift).pack(fill=tk.X, padx=6, pady=(3, 6))
        self.mark_progress_label = ttk.Label(mark_frame, textvariable=self.mark_progress_var, wraplength=360, style="Hint.TLabel")
        self.mark_progress_label.pack(anchor="w", padx=6, pady=(0, 5))

        drift_view_frame = ttk.LabelFrame(mark_tab, text="逐层漂移图显示范围")
        drift_view_frame.pack(fill=tk.X, pady=6, padx=2)
        row1 = ttk.Frame(drift_view_frame)
        row1.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(row1, text="参考层：").pack(side=tk.LEFT)
        ttk.Combobox(row1, textvariable=self.mark_drift_ref_mode, values=("最佳焦面", "当前层", "自定义层"), state="readonly", width=10).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(row1, text="自定义 layer：").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.mark_drift_custom_layer, width=7).pack(side=tk.LEFT)
        row2 = ttk.Frame(drift_view_frame)
        row2.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(row2, text="显示：").pack(side=tk.LEFT)
        ttk.Combobox(row2, textvariable=self.mark_drift_scope_mode, values=("全部层", "参考层±N层"), state="readonly", width=12).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(row2, text="N：").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.mark_drift_window_layers, width=6).pack(side=tk.LEFT, padx=(0, 8))
        row3 = ttk.Frame(drift_view_frame)
        row3.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(row3, text="坐标轴：").pack(side=tk.LEFT)
        ttk.Combobox(row3, textvariable=self.mark_drift_axis_mode, values=("局部自动缩放", "全局固定"), state="readonly", width=12).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Button(row3, text="🔄 更新漂移图", style="Ghost.TButton", command=self.update_mark_drift_plot).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Label(
            mark_frame,
            text="圆孔输出 X/Y/R/D；方孔或矩形孔输出 X/Y/W/H/角度。逐层漂移图按 X 向右为正、Y 向上为正显示。",
            wraplength=360,
        ).pack(anchor="w", padx=6, pady=(0, 6))

        mark_result_frame = ttk.LabelFrame(mark_tab, text="识别结果")
        mark_result_frame.pack(fill=tk.BOTH, expand=True, pady=8, padx=2)
        self.mark_label = ttk.Label(mark_result_frame, text="未识别。请先框选包含完整 mark 边界的 ROI。", wraplength=360, justify="left")
        self.mark_label.pack(anchor="nw", fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ========== 页面 3：FOV 高度图 / 面型分析 ==========
        topo_intro = ttk.Label(
            topo_tab,
            text="从 Z-stack 反推 FOV 内各 XY 网格点的焦面 Z，高度图再拟合平面，输出 Rx/Ry、面型 PV/RMS 和 TTV。",
            wraplength=360,
        )
        topo_intro.pack(anchor="w", padx=6, pady=(8, 6))

        topo_param_frame = ttk.LabelFrame(topo_tab, text="高度图参数")
        topo_param_frame.pack(fill=tk.X, pady=6, padx=2)
        line = ttk.Frame(topo_param_frame)
        line.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(line, text="Pixel size / μm：").pack(side=tk.LEFT)
        ttk.Entry(line, textvariable=self.pixel_size_um, width=10).pack(side=tk.LEFT)
        line2 = ttk.Frame(topo_param_frame)
        line2.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(line2, text="网格尺寸 / px：").pack(side=tk.LEFT)
        ttk.Entry(line2, textvariable=self.topo_grid_px, width=10).pack(side=tk.LEFT)
        line3 = ttk.Frame(topo_param_frame)
        line3.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(line3, text="残差滤波 nσ：").pack(side=tk.LEFT)
        ttk.Entry(line3, textvariable=self.topo_sigma_filter_var, width=10).pack(side=tk.LEFT)
        ttk.Label(line3, text="  0=关闭").pack(side=tk.LEFT)
        ttk.Label(
            topo_param_frame,
            text="建议先用 32 或 64 px；残差滤波建议 3σ，异常点多可用 2.5σ，想看原始结果填 0 关闭。",
            wraplength=360,
        ).pack(anchor="w", padx=6, pady=(0, 6))

        topo_alg_frame = ttk.LabelFrame(topo_tab, text="每个 XY 点的焦面判定")
        topo_alg_frame.pack(fill=tk.X, pady=6, padx=2)
        ttk.Combobox(
            topo_alg_frame,
            textvariable=self.topo_algorithm_var,
            values=("快速最大亮度投影", "快速峰值Z图", "快速峰值Z+抛物线插值", "共聚焦亮度高斯拟合（逐点）", "清晰度高斯拟合-峰值（逐点）", "清晰度高斯拟合-谷值（逐点）", "共聚焦亮度峰值", "清晰度峰值", "清晰度谷值"),
            state="readonly",
            width=24,
        ).pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(
            topo_alg_frame,
            text="快速最大亮度投影适合快速观察；快速峰值Z/抛物线适合快速高度趋势；逐点高斯更准但更慢。",
            wraplength=360,
        ).pack(anchor="w", padx=6, pady=(0, 6))

        topo_roi_frame = ttk.LabelFrame(topo_tab, text="FOV / ROI 区域")
        topo_roi_frame.pack(fill=tk.X, pady=6, padx=2)
        ttk.Checkbutton(topo_roi_frame, text="只对当前 ROI 生成高度图", variable=self.topo_use_roi_var).pack(anchor="w", padx=6, pady=4)
        ttk.Button(topo_roi_frame, text="▣ 在图像上框选高度图 ROI", style="Warn.TButton", command=self.enable_roi_selector).pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(topo_roi_frame, text="↺ 清除 ROI，使用全 FOV", style="Ghost.TButton", command=self.clear_roi).pack(fill=tk.X, padx=6, pady=3)
        self.roi_label_topo = ttk.Label(topo_roi_frame, text="当前 ROI：全图", wraplength=360)
        self.roi_label_topo.pack(anchor="w", padx=6, pady=4)
        ttk.Separator(topo_roi_frame).pack(fill=tk.X, padx=6, pady=5)
        ttk.Button(topo_roi_frame, text="⊖ 添加排除区域 Mask", style="Warn.TButton", command=self.enable_exclude_selector).pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(topo_roi_frame, text="🧹 清除排除区域", style="Ghost.TButton", command=self.clear_exclude_rois).pack(fill=tk.X, padx=6, pady=3)
        self.exclude_label_topo = ttk.Label(topo_roi_frame, text="排除区域：0 个", wraplength=360)
        self.exclude_label_topo.pack(anchor="w", padx=6, pady=4)

        self.topo_button = ttk.Button(topo_tab, text="🗺 生成高度图 / 计算面型", style="Success.TButton", command=self.build_topography)
        self.topo_button.pack(fill=tk.X, pady=(10, 4), padx=2)
        ttk.Button(topo_tab, text="⬇ 导出高度点 CSV", style="Ghost.TButton", command=self.export_topography_csv).pack(fill=tk.X, pady=3, padx=2)
        ttk.Button(topo_tab, text="🖼 导出高斯融合图 PNG", style="Ghost.TButton", command=self.export_fused_gray_png).pack(fill=tk.X, pady=3, padx=2)
        self.topo_progress_label = ttk.Label(topo_tab, textvariable=self.topo_progress_var, wraplength=360, style="Hint.TLabel")
        self.topo_progress_label.pack(anchor="w", pady=(0, 4), padx=2)

        ttk.Label(
            topo_tab,
            text="高度图 / 面型结果已移到右下方结果页，生成后可直接查看，不再挤在左侧。",
            wraplength=360,
            style="Hint.TLabel",
        ).pack(anchor="w", pady=6, padx=4)

        # 右侧图表区
        top_bar = ttk.Frame(right)
        top_bar.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(top_bar, text="🧭 层选择：").pack(side=tk.LEFT)
        self.layer_scale = ttk.Scale(top_bar, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_scale_change)
        self.layer_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.layer_label = ttk.Label(top_bar, text="0 / 0")
        self.layer_label.pack(side=tk.LEFT, padx=6)

        self.fig = Figure(figsize=(10, 7), dpi=100, constrained_layout=True)
        self.ax_img = self.fig.add_subplot(2, 2, 1)
        self.ax_curve = self.fig.add_subplot(2, 2, 2)
        self.ax_prev = self.fig.add_subplot(2, 2, 3)
        self.ax_next = self.fig.add_subplot(2, 2, 4)
        # 使用 constrained_layout，避免标题、坐标轴和图例重叠。

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, right)
        toolbar.update()

        # 右下方固定结果区：把重要结论从左侧控制栏移出，避免被参数控件挤压。
        bottom_panel = ttk.Frame(right)
        bottom_panel.pack(fill=tk.BOTH, expand=False, pady=(6, 0))
        bottom_panel.configure(height=230)
        bottom_panel.pack_propagate(False)

        self.output_tabs = ttk.Notebook(bottom_panel)
        self.output_tabs.pack(fill=tk.BOTH, expand=True)

        table_frame = ttk.Frame(self.output_tabs)
        focus_result_frame = ttk.Frame(self.output_tabs)
        topo_result_frame = ttk.Frame(self.output_tabs)
        self.output_tabs.add(table_frame, text="📋 逐层结果表")
        self.output_tabs.add(focus_result_frame, text="📈 逐层分析结论")
        self.output_tabs.add(topo_result_frame, text="🗺 高度图/面型结果")

        columns = ("index", "z_um", "filename", "score", "mean", "lap", "ten", "brenner", "contrast", "entropy")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=7)
        headers = {
            "index": "层号", "z_um": "Z/μm", "filename": "文件名", "score": "算法评分", "mean": "平均灰度",
            "lap": "Laplacian", "ten": "Tenengrad", "brenner": "Brenner", "contrast": "局部对比", "entropy": "熵"
        }
        widths = {"index": 55, "z_um": 85, "filename": 260, "score": 95, "mean": 90, "lap": 90, "ten": 90, "brenner": 90, "contrast": 90, "entropy": 75}
        for c in columns:
            self.tree.heading(c, text=headers[c])
            self.tree.column(c, width=widths[c], anchor="center")
        tree_y = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        tree_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=tree_y.set, xscrollcommand=tree_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_y.grid(row=0, column=1, sticky="ns")
        tree_x.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        self.result_text = tk.Text(focus_result_frame, height=8, wrap="word", font=("Consolas", 10))
        focus_y = ttk.Scrollbar(focus_result_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=focus_y.set)
        self.result_text.grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=4)
        focus_y.grid(row=0, column=1, sticky="ns", pady=4)
        focus_result_frame.rowconfigure(0, weight=1)
        focus_result_frame.columnconfigure(0, weight=1)
        self.result_text.insert(tk.END, "逐层分析结果会显示在这里。\n")

        self.topo_result_text = tk.Text(topo_result_frame, height=8, wrap="word", font=("Consolas", 10))
        topo_y = ttk.Scrollbar(topo_result_frame, orient="vertical", command=self.topo_result_text.yview)
        self.topo_result_text.configure(yscrollcommand=topo_y.set)
        self.topo_result_text.grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=4)
        topo_y.grid(row=0, column=1, sticky="ns", pady=4)
        topo_result_frame.rowconfigure(0, weight=1)
        topo_result_frame.columnconfigure(0, weight=1)
        self.topo_result_text.insert(tk.END, "高度图 / 面型结果会显示在这里。\n")


    def _make_standard_axes(self):
        """恢复焦面分析/Mark 识别使用的标准 2x2 图表布局。"""
        self.fig.clear()
        self.ax_img = self.fig.add_subplot(2, 2, 1)
        self.ax_curve = self.fig.add_subplot(2, 2, 2)
        self.ax_prev = self.fig.add_subplot(2, 2, 3)
        self.ax_next = self.fig.add_subplot(2, 2, 4)
        self.plot_mode = "standard"

    def _make_topography_axes(self):
        """高度图页面使用：高斯融合灰度图、高度图、残差图、拟合质量图。"""
        self.fig.clear()
        self.ax_img = self.fig.add_subplot(2, 2, 1)
        self.ax_curve = self.fig.add_subplot(2, 2, 2)
        self.ax_prev = self.fig.add_subplot(2, 2, 3)
        self.ax_next = self.fig.add_subplot(2, 2, 4)
        self.plot_mode = "topography"

    def _make_mark_drift_axes(self):
        """ROI 中心逐层漂移页面：当前图、XY轨迹、ΔX/ΔY-Z、dr-Z。"""
        self.fig.clear()
        self.ax_img = self.fig.add_subplot(2, 2, 1)
        self.ax_curve = self.fig.add_subplot(2, 2, 2)
        self.ax_prev = self.fig.add_subplot(2, 2, 3)
        self.ax_next = self.fig.add_subplot(2, 2, 4)
        self.plot_mode = "mark_drift"


    def set_roi_label(self, text: str):
        if hasattr(self, "roi_label"):
            self.roi_label.config(text=text)
        if hasattr(self, "roi_label_mark"):
            self.roi_label_mark.config(text=text)
        if hasattr(self, "roi_label_topo"):
            self.roi_label_topo.config(text=text)

    def update_exclude_label(self):
        if not hasattr(self, "exclude_label_topo"):
            return
        if not self.exclude_rois:
            self.exclude_label_topo.config(text="排除区域：0 个")
            return
        parts = []
        for i, (x, y, w, h) in enumerate(self.exclude_rois[:3], start=1):
            parts.append(f"#{i}: x={x}, y={y}, w={w}, h={h}")
        more = "" if len(self.exclude_rois) <= 3 else f"；另有 {len(self.exclude_rois)-3} 个"
        self.exclude_label_topo.config(text=f"排除区域：{len(self.exclude_rois)} 个；" + "；".join(parts) + more)

    def log_result(self, text: str):
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        if hasattr(self, "output_tabs"):
            try:
                self.output_tabs.select(1)
            except tk.TclError:
                pass

    def select_folder(self):
        folder = filedialog.askdirectory(title="选择 Z-stack 图片文件夹")
        if not folder:
            return
        self.folder = folder
        paths = []
        for p in Path(folder).iterdir():
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
                paths.append(str(p))
        paths.sort(key=lambda x: natural_key(os.path.basename(x)))
        self.image_paths = paths
        self.results = []
        self.mark_result = None
        self.mark_drift_results = []
        self.exclude_rois = []
        if hasattr(self, "mark_progress_var"):
            self.mark_progress_var.set("未进行逐层中心识别")
        if hasattr(self, "mark_label"):
            self.mark_label.config(text="未识别。请先框选包含完整 mark 边界的 ROI。")
        self._display_cache.clear()
        self.folder_label.config(text=f"{folder}\n共发现 {len(paths)} 张图片")
        if not paths:
            messagebox.showwarning("未找到图片", "文件夹中没有找到 png/jpg/tif/bmp 图片。")
            return
        self.current_index = 0
        self.layer_scale.configure(from_=0, to=max(0, len(paths) - 1))
        self.update_image_panel()

    def get_active_roi(self):
        return self.roi if self.use_roi_var.get() else None

    def z_value(self, index: int) -> float:
        return float(self.first_z_um.get()) + index * float(self.z_step_um.get())

    def z_values(self) -> np.ndarray:
        return np.array([self.z_value(i) for i in range(len(self.image_paths))], dtype=float)

    def analyze(self):
        if not self.image_paths:
            messagebox.showwarning("未选择图片", "请先选择 Z-stack 图片文件夹。")
            return
        if self.is_analyzing:
            messagebox.showinfo("正在分析", "当前 Z-stack 还在分析中，请等本轮完成。")
            return

        self.is_analyzing = True
        self.config(cursor="watch")
        self.analyze_button.config(state="disabled")
        self.progress_var.set("分析中：0 / {}".format(len(self.image_paths)))
        self.log_result("正在后台分析，不需要关闭窗口。大图或很多层时会稍慢。")

        roi = self.get_active_roi()
        try:
            max_dim = int(self.max_analysis_dim.get())
        except Exception:
            max_dim = 1024
            self.max_analysis_dim.set(max_dim)

        worker = threading.Thread(target=self._analyze_worker, args=(roi, max_dim), daemon=True)
        worker.start()

    def _analyze_worker(self, roi, max_dim: int):
        try:
            tmp_results = []
            total = len(self.image_paths)
            for idx, path in enumerate(self.image_paths):
                # 只读取一次原始灰度，避免同一张大图反复 imdecode 导致 UI 假死。
                img_raw = safe_read_gray(path, normalize_per_image=False)
                img_raw_roi = crop_roi(img_raw, roi)

                # 对焦度计算默认降采样；焦面判断一般看趋势，1024px 足够，速度稳定很多。
                img_for_metric = resize_max_dim(img_raw_roi, max_dim)
                img_focus_roi = normalize_to_255(img_for_metric)
                metrics = calc_metrics(img_focus_roi)

                # 亮度法保留原始灰度关系，不做逐张归一化。
                metrics["mean_intensity_raw"] = float(np.mean(img_raw_roi))
                metrics["max_intensity_raw"] = float(np.max(img_raw_roi))
                tmp_results.append(LayerResult(index=idx, filename=os.path.basename(path), path=path, metrics=metrics))

                if idx % 2 == 0 or idx == total - 1:
                    self.after(0, lambda i=idx + 1, t=total: self.progress_var.set(f"分析中：{i} / {t}"))

            self.after(0, lambda: self._finish_analysis(tmp_results, None))
        except Exception as e:
            err = f"{e}\n\n{traceback.format_exc()}"
            self.after(0, lambda: self._finish_analysis(None, err))

    def _finish_analysis(self, tmp_results, error: Optional[str]):
        try:
            if error:
                messagebox.showerror("分析失败", error)
                self.progress_var.set("分析失败")
                return
            self.results = tmp_results or []
            self.compute_algorithm_score()
            self.update_table()
            self.update_plot()
            self.update_conclusion()
            self.progress_var.set(f"完成：共分析 {len(self.results)} 层")
        finally:
            self.is_analyzing = False
            self.config(cursor="")
            self.analyze_button.config(state="normal")

    def compute_algorithm_score(self):
        if not self.results:
            return
        self.gaussian_fit = None
        self.focus_fit = None
        algorithm = self.algorithm_var.get()

        def set_combined_score_from_weighted_clarity():
            metric_keys = ["laplacian_var", "tenengrad", "brenner", "local_contrast", "entropy"]
            norm_map = {}
            for key in metric_keys:
                norm_map[key] = normalize_series([r.metrics[key] for r in self.results])

            weights = {k: float(v.get()) for k, v in self.metric_weights.items()}
            total_w = sum(max(0.0, w) for w in weights.values())
            if total_w <= 0:
                total_w = 1.0
            scores = []
            for i, r in enumerate(self.results):
                score = 0.0
                for key in metric_keys:
                    score += max(0.0, weights[key]) / total_w * norm_map[key][i]
                scores.append(float(score))
                r.combined_score = float(score)
            return np.asarray(scores, dtype=float)

        if algorithm in ("综合清晰度评分", "综合清晰度高斯拟合-自动峰谷", "综合清晰度反向谷值"):
            scores = set_combined_score_from_weighted_clarity()
            if algorithm == "综合清晰度反向谷值":
                inv = 1.0 - normalize_series(scores)
                for i, r in enumerate(self.results):
                    r.combined_score = float(inv[i])
                return
            if algorithm == "综合清晰度高斯拟合-自动峰谷":
                self.focus_fit = estimate_focus_metric_fit(self.z_values(), scores, allow_valley=True)
            return

        intensities = np.array([r.metrics["mean_intensity_raw"] for r in self.results], dtype=float)
        norm_i = normalize_series(intensities).astype(float)
        for i, r in enumerate(self.results):
            r.combined_score = float(norm_i[i])

        if algorithm == "共聚焦亮度高斯拟合":
            self.gaussian_fit = estimate_gaussian_focus(self.z_values(), intensities)

    def best_index(self) -> Optional[int]:
        if not self.results:
            return None
        if self.algorithm_var.get() == "共聚焦亮度高斯拟合" and self.gaussian_fit:
            mu = float(self.gaussian_fit["mu"])
            z = self.z_values()
            return int(np.argmin(np.abs(z - mu)))
        if self.focus_fit:
            mu = float(self.focus_fit["mu"])
            z = self.z_values()
            return int(np.argmin(np.abs(z - mu)))
        scores = [r.combined_score for r in self.results]
        return int(np.argmax(scores))

    def best_z_um(self) -> Optional[float]:
        if not self.results:
            return None
        if self.algorithm_var.get() == "共聚焦亮度高斯拟合" and self.gaussian_fit:
            return float(self.gaussian_fit["mu"])
        if self.focus_fit:
            return float(self.focus_fit["mu"])
        best = self.best_index()
        return self.z_value(best) if best is not None else None

    def update_conclusion(self):
        best = self.best_index()
        best_z = self.best_z_um()
        if best is None or best_z is None:
            return
        r = self.results[best]
        scores = np.asarray([x.combined_score for x in self.results], dtype=float)
        algorithm = self.algorithm_var.get()

        sorted_scores = np.sort(scores)[::-1]
        gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else np.nan
        half_max_count = int(np.sum(scores > (scores.min() + 0.5 * (scores.max() - scores.min())))) if scores.max() > scores.min() else len(scores)

        confidence = "中"
        if algorithm == "共聚焦亮度高斯拟合" and self.gaussian_fit:
            r2 = float(self.gaussian_fit.get("r2", np.nan))
            if np.isfinite(r2) and r2 >= 0.9:
                confidence = "高"
            elif np.isfinite(r2) and r2 < 0.75:
                confidence = "低：亮度-Z 曲线不太符合高斯单峰"
        else:
            if len(scores) >= 5 and gap > 0.12 and half_max_count <= max(3, len(scores) // 5):
                confidence = "高"
            elif len(scores) >= 3 and gap < 0.04:
                confidence = "低：多个 Z 层评分接近，建议缩小 Z step 或用 ROI 复核"

        text = []
        text.append(f"当前算法：{algorithm}")
        text.append(f"推荐焦面层：第 {best} 层（表格从 0 开始计数）")
        text.append(f"推荐 Z 位置：{best_z:.4f} μm")
        text.append(f"邻近文件名：{r.filename}")
        text.append(f"算法评分：{r.combined_score:.4f}")
        text.append(f"置信度判断：{confidence}")

        if algorithm == "共聚焦亮度高斯拟合" and self.gaussian_fit:
            fit = self.gaussian_fit
            text.append("")
            text.append("亮度高斯拟合结果：")
            text.append(f"- 拟合焦面 μ：{float(fit['mu']):.4f} μm")
            text.append(f"- σ：{float(fit['sigma']):.4f} μm")
            text.append(f"- R²：{float(fit['r2']):.4f}" if np.isfinite(float(fit['r2'])) else "- R²：N/A")
            text.append(f"- 拟合方法：{fit['method']}")
            if fit.get("warning"):
                text.append(f"- 注意：{fit['warning']}")

        if self.focus_fit:
            fit = self.focus_fit
            polarity_cn = "峰值型：焦面取评分最高处" if fit.get("polarity") == "peak" else "谷值型：焦面取评分最低处"
            text.append("")
            text.append("清晰度曲线拟合结果：")
            text.append(f"- 曲线方向：{polarity_cn}")
            text.append(f"- 拟合焦面 μ：{float(fit['mu']):.4f} μm")
            text.append(f"- σ：{float(fit['sigma']):.4f} μm" if np.isfinite(float(fit['sigma'])) else "- σ：N/A")
            text.append(f"- R²：{float(fit['r2']):.4f}" if np.isfinite(float(fit['r2'])) else "- R²：N/A")
            text.append(f"- 拟合方法：{fit['method']}")
            if fit.get("warning"):
                text.append(f"- 注意：{fit['warning']}")

        text.append("")
        text.append("核心指标：")
        text.append(f"- 原始平均灰度：{r.metrics['mean_intensity_raw']:.3f}")
        text.append(f"- 原始最大灰度：{r.metrics['max_intensity_raw']:.3f}")
        text.append(f"- Laplacian 方差：{r.metrics['laplacian_var']:.3f}")
        text.append(f"- Tenengrad：{r.metrics['tenengrad']:.3f}")
        text.append(f"- Brenner：{r.metrics['brenner']:.3f}")
        text.append(f"- 局部对比度：{r.metrics['local_contrast']:.3f}")
        text.append(f"- 熵：{r.metrics['entropy']:.3f}")
        text.append("")
        text.append("使用建议：")
        text.append("1. 如果清晰度曲线在肉眼焦面处是最低点，优先试‘综合清晰度高斯拟合-自动峰谷’或‘综合清晰度反向谷值’。")
        text.append("2. 亮度高斯法适合共聚焦单层荧光/反射峰明显的情况；如果发生饱和或漂白，结果会偏。")
        text.append("3. 如果导出图片被软件逐张自动增强，亮度法会失效，应使用原始灰度 TIFF 或改用清晰度评分法。")
        text.append("4. 样品有倾斜/高度差时，建议分 ROI 分析，不要只看全图平均。")
        text.append("5. Mark 识别建议使用只包含一个完整圆孔/方孔边界的 ROI；ROI 过大或包含多个孔会影响识别。")
        self.log_result("\n".join(text))

    def update_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for r in self.results:
            m = r.metrics
            self.tree.insert("", tk.END, values=(
                r.index, f"{self.z_value(r.index):.3f}", r.filename, f"{r.combined_score:.4f}",
                f"{m.get('mean_intensity_raw', m.get('mean_intensity', 0.0)):.2f}",
                f"{m['laplacian_var']:.2f}", f"{m['tenengrad']:.2f}", f"{m['brenner']:.2f}",
                f"{m['local_contrast']:.2f}", f"{m['entropy']:.2f}"
            ))

    def update_plot(self):
        if getattr(self, "plot_mode", "standard") != "standard":
            self._make_standard_axes()
        self.ax_curve.clear()
        if not self.results:
            self.update_image_panel()
            return
        xs = self.z_values()
        scores = np.array([r.combined_score for r in self.results])
        algorithm = self.algorithm_var.get()

        if algorithm == "共聚焦亮度高斯拟合":
            intensities = np.array([r.metrics["mean_intensity_raw"] for r in self.results], dtype=float)
            self.ax_curve.plot(xs, intensities, marker="o", label="平均灰度")
            if self.gaussian_fit:
                dense_x = np.linspace(float(xs.min()), float(xs.max()), 300)
                fit = self.gaussian_fit
                dense_y = gaussian_model(dense_x, float(fit["baseline"]), float(fit["amplitude"]), float(fit["mu"]), float(fit["sigma"]))
                self.ax_curve.plot(dense_x, dense_y, linestyle="--", label="Gaussian fit")
                self.ax_curve.axvline(float(fit["mu"]), linestyle=":", label=f"焦面 Z={float(fit['mu']):.3f} μm")
            self.ax_curve.set_title("共聚焦亮度-Z 高斯拟合")
            self.ax_curve.set_ylabel("Mean intensity / raw gray")
        elif algorithm == "平均亮度最大值":
            intensities = np.array([r.metrics["mean_intensity_raw"] for r in self.results], dtype=float)
            self.ax_curve.plot(xs, intensities, marker="o", label="平均灰度")
            best = self.best_index()
            if best is not None:
                self.ax_curve.axvline(self.z_value(best), linestyle="--", label=f"最大亮度层 {best}")
            self.ax_curve.set_title("平均亮度-Z 曲线")
            self.ax_curve.set_ylabel("Mean intensity / raw gray")
        else:
            plot_scores = scores
            raw_scores = np.array([r.combined_score for r in self.results], dtype=float)
            if algorithm == "综合清晰度高斯拟合-自动峰谷" and self.focus_fit:
                # 拟合用的是未反向的综合清晰度原始评分；这里也按原始评分画，避免视觉误导。
                metric_keys = ["laplacian_var", "tenengrad", "brenner", "local_contrast", "entropy"]
                norm_map = {key: normalize_series([r.metrics[key] for r in self.results]) for key in metric_keys}
                weights = {k: float(v.get()) for k, v in self.metric_weights.items()}
                total_w = sum(max(0.0, w) for w in weights.values()) or 1.0
                raw_scores = np.zeros(len(self.results), dtype=float)
                for key in metric_keys:
                    raw_scores += max(0.0, weights[key]) / total_w * norm_map[key]
                plot_scores = raw_scores
            self.ax_curve.plot(xs, plot_scores, marker="o", label="综合清晰度评分")
            if self.focus_fit:
                dense_x = np.linspace(float(xs.min()), float(xs.max()), 300)
                fit = self.focus_fit
                if np.isfinite(float(fit["sigma"])):
                    dense_y = gaussian_model(dense_x, float(fit["baseline"]), float(fit["amplitude"]), float(fit["mu"]), float(fit["sigma"]))
                    self.ax_curve.plot(dense_x, dense_y, linestyle="--", label="Gaussian fit")
                self.ax_curve.axvline(float(fit["mu"]), linestyle=":", label=f"拟合焦面 Z={float(fit['mu']):.3f} μm")
            else:
                best = self.best_index()
                if best is not None:
                    self.ax_curve.axvline(self.z_value(best), linestyle="--", label=f"最佳层 {best}")
            self.ax_curve.set_title("清晰度-Z 曲线")
            self.ax_curve.set_ylabel("Normalized focus score")

        self.ax_curve.set_xlabel("Z position / μm")
        self.ax_curve.grid(True, alpha=0.3)
        self.ax_curve.legend(loc="best")
        best = self.best_index()
        self.current_index = best if best is not None else self.current_index
        self.layer_scale.set(self.current_index)
        self.update_image_panel(redraw=False)
        self.canvas.draw_idle()

    def update_image_panel(self, redraw=True):
        if getattr(self, "plot_mode", "standard") != "standard":
            self._make_standard_axes()
        self.ax_img.clear()
        self.ax_prev.clear()
        self.ax_next.clear()
        if not self.image_paths:
            self.ax_img.set_title("未导入图片")
            self.canvas.draw_idle()
            return

        idx = max(0, min(self.current_index, len(self.image_paths) - 1))
        self.current_index = idx
        self.layer_label.config(text=f"{idx + 1} / {len(self.image_paths)}")

        def show(ax, i, title):
            if i < 0 or i >= len(self.image_paths):
                ax.axis("off")
                return
            if i not in self._display_cache:
                raw = safe_read_gray(self.image_paths[i], normalize_per_image=False)
                disp = normalize_to_255(raw)
                disp = resize_max_dim(disp, 1200)
                self._display_cache[i] = disp
                # 缓存限制，避免大量大图占内存。
                if len(self._display_cache) > 8:
                    for k in list(self._display_cache.keys())[:-8]:
                        self._display_cache.pop(k, None)
            img = self._display_cache[i]
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            # 显示图可能被缩放，ROI 框和圆拟合结果也需要按比例缩放。
            raw0 = safe_read_gray(self.image_paths[i], normalize_per_image=False)
            scale_x = img.shape[1] / raw0.shape[1]
            scale_y = img.shape[0] / raw0.shape[0]
            if self.roi is not None:
                x, y, w, h = self.roi
                rect = plt_rectangle(x * scale_x, y * scale_y, w * scale_x, h * scale_y, edgecolor="#2563EB")
                ax.add_patch(rect)
            for ex_x, ex_y, ex_w, ex_h in self.exclude_rois:
                ex_rect = plt_rectangle(ex_x * scale_x, ex_y * scale_y, ex_w * scale_x, ex_h * scale_y, edgecolor="#EF4444", linestyle="--")
                ax.add_patch(ex_rect)
            if self.mark_result is not None and int(self.mark_result.get("layer_index", -1)) == i:
                # 1) 叠加实际识别到的边界点/轮廓。
                bx = float(self.mark_result.get("roi_x", 0.0))
                by = float(self.mark_result.get("roi_y", 0.0))
                pts = self.mark_result.get("boundary_points_roi")
                if pts is not None:
                    pts_arr = np.asarray(pts, dtype=float)
                    if pts_arr.ndim == 2 and pts_arr.shape[0] > 1:
                        ax.plot((pts_arr[:, 0] + bx) * scale_x, (pts_arr[:, 1] + by) * scale_y, linewidth=1.0, label="detected boundary")

                # 2) 叠加几何拟合结果：圆 or 最小外接矩形。
                mx = float(self.mark_result["x_px"]) * scale_x
                my = float(self.mark_result["y_px"]) * scale_y
                if self.mark_result.get("shape") == "circle" and "r_px" in self.mark_result:
                    rr = float(self.mark_result["r_px"]) * (scale_x + scale_y) / 2.0
                    circ = plt_circle(mx, my, rr)
                    ax.add_patch(circ)
                elif self.mark_result.get("rect_corners_roi") is not None:
                    corners = np.asarray(self.mark_result.get("rect_corners_roi"), dtype=float)
                    if corners.ndim == 2 and corners.shape[0] == 4:
                        corners[:, 0] = (corners[:, 0] + bx) * scale_x
                        corners[:, 1] = (corners[:, 1] + by) * scale_y
                        poly = plt_polygon(corners)
                        ax.add_patch(poly)
                ax.plot([mx - 8, mx + 8], [my, my], linewidth=1.2)
                ax.plot([mx, mx], [my - 8, my + 8], linewidth=1.2)

        score_txt = ""
        if self.results:
            score_txt = f" | Z={self.z_value(idx):.3f} μm | score={self.results[idx].combined_score:.4f}"
        show(self.ax_img, idx, f"当前层 {idx}: {os.path.basename(self.image_paths[idx])}{score_txt}")
        show(self.ax_prev, idx - 1, f"上一层 {idx - 1}" if idx > 0 else "上一层：无")
        show(self.ax_next, idx + 1, f"下一层 {idx + 1}" if idx + 1 < len(self.image_paths) else "下一层：无")
        if redraw:
            self.canvas.draw_idle()

    def on_scale_change(self, value):
        if not self.image_paths:
            return
        self.current_index = int(round(float(value)))
        self.update_image_panel()

    def on_tree_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        values = self.tree.item(sel[0], "values")
        if values:
            self.current_index = int(values[0])
            self.layer_scale.set(self.current_index)
            self.update_image_panel()

    def enable_roi_selector(self):
        if not self.image_paths:
            messagebox.showwarning("未选择图片", "请先导入图片文件夹。")
            return
        self.update_image_panel()
        if self.selector:
            self.selector.set_active(False)
            self.selector = None
        self.selector_mode = "roi"
        self.selector = RectangleSelector(
            self.ax_img,
            self.on_roi_selected,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        self.canvas.draw_idle()
        messagebox.showinfo("ROI 框选", "请在左上方当前层图像上拖拽鼠标框选 ROI。框选完成后会自动启用 ROI。")

    def enable_exclude_selector(self):
        if not self.image_paths:
            messagebox.showwarning("未选择图片", "请先导入图片文件夹。")
            return
        self.update_image_panel()
        if self.selector:
            self.selector.set_active(False)
            self.selector = None
        self.selector_mode = "exclude"
        self.selector = RectangleSelector(
            self.ax_img,
            self.on_roi_selected,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        self.canvas.draw_idle()
        messagebox.showinfo("排除区域", "请在左上方当前层图像上拖拽要排除的区域。可重复添加多个 Mask。")

    def on_roi_selected(self, eclick, erelease):
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return
        x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        if w <= 5 or h <= 5:
            return
        # 如果当前显示图做过降采样，把框选坐标换算回原始图坐标。
        if self.current_index in self._display_cache:
            disp = self._display_cache[self.current_index]
            raw = safe_read_gray(self.image_paths[self.current_index], normalize_per_image=False)
            sx = raw.shape[1] / max(1, disp.shape[1])
            sy = raw.shape[0] / max(1, disp.shape[0])
            x, y, w, h = int(round(x * sx)), int(round(y * sy)), int(round(w * sx)), int(round(h * sy))
        if self.selector_mode == "exclude":
            self.exclude_rois.append((x, y, w, h))
            self.update_exclude_label()
            if self.selector:
                self.selector.set_active(False)
            self.update_image_panel()
            messagebox.showinfo("排除区域已添加", f"已添加排除 Mask：x={x}, y={y}, w={w}, h={h}。")
            return

        self.roi = (x, y, w, h)
        self.use_roi_var.set(True)
        self.set_roi_label(f"当前 ROI：x={x}, y={y}, w={w}, h={h}")
        if self.selector:
            self.selector.set_active(False)
        self.update_image_panel()

    def clear_exclude_rois(self):
        self.exclude_rois = []
        self.update_exclude_label()
        self.update_image_panel()

    def clear_roi(self):
        self.roi = None
        self.use_roi_var.set(False)
        self.set_roi_label("当前 ROI：全图")
        self.update_image_panel()

    def detect_mark_hole_current(self):
        if not self.image_paths:
            messagebox.showwarning("未选择图片", "请先选择 Z-stack 图片文件夹。")
            return
        if self.roi is None:
            messagebox.showwarning("未设置 ROI", "请先框选一个包含完整 mark 边界的 ROI。")
            return
        try:
            pix = float(self.pixel_size_um.get())
            if pix <= 0:
                raise ValueError
        except Exception:
            messagebox.showwarning("Pixel size 无效", "请填写大于 0 的 pixel size，单位 μm/pixel。")
            return
        try:
            raw = safe_read_gray(self.image_paths[self.current_index], normalize_per_image=False)
            mode = self.mark_shape_var.get()
            res = detect_mark_in_roi(raw, self.roi, mode=mode)
            res["pixel_size_um"] = pix
            res["x_um"] = float(res["x_px"]) * pix
            res["y_um"] = float(res["y_px"]) * pix
            if res.get("shape") == "circle":
                res["r_um"] = float(res["r_px"]) * pix
                res["d_um"] = float(res["d_px"]) * pix
            else:
                res["width_um"] = float(res["width_px"]) * pix
                res["height_um"] = float(res["height_px"]) * pix
            res["layer_index"] = float(self.current_index)
            res["z_um"] = self.z_value(self.current_index)
            self.mark_result = res

            common = (
                f"层 {self.current_index}, Z={res['z_um']:.4f} μm\n"
                f"形状={res.get('shape')}，极性={res.get('polarity')}\n"
                f"X={res['x_px']:.3f} px / {res['x_um']:.4f} μm\n"
                f"Y={res['y_px']:.3f} px / {res['y_um']:.4f} μm\n"
            )
            if res.get("shape") == "circle":
                detail = (
                    f"R={res['r_px']:.3f} px / {res['r_um']:.4f} μm\n"
                    f"D={res['d_px']:.3f} px / {res['d_um']:.4f} μm\n"
                    f"圆拟合 RMS={float(res.get('rms_px', float('nan'))):.3f} px，圆度={float(res.get('circularity', float('nan'))):.3f}\n"
                    f"边缘算法={res.get('edge_fit', 'contour_lsq')}"
                )
            else:
                detail = (
                    f"W={res['width_px']:.3f} px / {res['width_um']:.4f} μm\n"
                    f"H={res['height_px']:.3f} px / {res['height_um']:.4f} μm\n"
                    f"角度={res['angle_deg']:.3f}°，长宽比={res['aspect_ratio']:.3f}"
                )
            self.mark_label.config(text=common + detail)
            self.update_image_panel()
        except Exception as e:
            messagebox.showerror("Mark识别失败", str(e))


    def analyze_mark_drift(self):
        """对每一层 Z-stack 在同一 ROI 内识别 mark 中心，并绘制 XY 漂移三子图。"""
        if not self.image_paths:
            messagebox.showwarning("未选择图片", "请先选择 Z-stack 图片文件夹。")
            return
        if self.roi is None:
            messagebox.showwarning("未设置 ROI", "请先框选一个包含完整 mark 边界的 ROI。")
            return
        if self.is_mark_analyzing:
            messagebox.showinfo("正在识别", "当前逐层中心识别还在进行中，请等本轮完成。")
            return
        try:
            pix = float(self.pixel_size_um.get())
            if pix <= 0:
                raise ValueError
        except Exception:
            messagebox.showwarning("Pixel size 无效", "请填写大于 0 的 pixel size，单位 μm/pixel。")
            return

        self.is_mark_analyzing = True
        self.config(cursor="watch")
        self.mark_progress_var.set(f"逐层中心识别中：0 / {len(self.image_paths)}")
        if hasattr(self, "mark_label"):
            self.mark_label.config(text="正在逐层识别 ROI Mark 中心，完成后会生成三张漂移图。")
        worker = threading.Thread(target=self._mark_drift_worker, args=(self.roi, self.mark_shape_var.get(), pix), daemon=True)
        worker.start()

    def _mark_drift_worker(self, roi, mode: str, pix: float):
        try:
            rows = []
            errors = []
            total = len(self.image_paths)
            for idx, path in enumerate(self.image_paths):
                try:
                    raw = safe_read_gray(path, normalize_per_image=False)
                    res = detect_mark_in_roi(raw, roi, mode=mode)
                    res = dict(res)
                    res["layer_index"] = int(idx)
                    res["z_um"] = float(self.z_value(idx))
                    res["filename"] = os.path.basename(path)
                    res["pixel_size_um"] = float(pix)
                    res["x_um_abs"] = float(res["x_px"]) * pix
                    # 绝对 Y 仍记录原图像素坐标对应的物理量；漂移图会改用 Y 向上为正。
                    res["y_um_abs_image_down"] = float(res["y_px"]) * pix
                    if res.get("shape") == "circle":
                        res["r_um"] = float(res.get("r_px", np.nan)) * pix
                        res["d_um"] = float(res.get("d_px", np.nan)) * pix
                    else:
                        res["width_um"] = float(res.get("width_px", np.nan)) * pix
                        res["height_um"] = float(res.get("height_px", np.nan)) * pix
                    rows.append(res)
                except Exception as e:
                    errors.append((idx, os.path.basename(path), str(e)))
                if idx % 1 == 0 or idx == total - 1:
                    self.after(0, lambda i=idx + 1, t=total: self.mark_progress_var.set(f"逐层中心识别中：{i} / {t}"))
            self.after(0, lambda: self._finish_mark_drift(rows, errors, pix))
        except Exception as e:
            err = f"{e}\n\n{traceback.format_exc()}"
            self.after(0, lambda: self._finish_mark_drift(None, err, pix))

    def _finish_mark_drift(self, rows, errors, pix: float):
        try:
            if isinstance(errors, str):
                messagebox.showerror("逐层中心识别失败", errors)
                self.mark_progress_var.set("逐层中心识别失败")
                return
            rows = rows or []
            self.mark_drift_results = rows
            if len(rows) < 2:
                msg = "有效识别层数少于 2，无法绘制漂移曲线。"
                if errors:
                    msg += "\n\n失败层：\n" + "\n".join([f"{i}: {name} - {err}" for i, name, err in errors[:10]])
                messagebox.showwarning("有效层不足", msg)
                self.mark_progress_var.set(f"完成但有效层不足：{len(rows)} / {len(self.image_paths)}")
                return
            self.update_mark_drift_plot()
            ok = len(rows)
            fail = len(errors or [])
            self.mark_progress_var.set(f"完成：有效识别 {ok} / {len(self.image_paths)} 层，失败 {fail} 层")
            self._update_mark_drift_text(errors or [], pix)
        finally:
            self.is_mark_analyzing = False
            self.config(cursor="")

    def _mark_reference_index(self, valid_indices: np.ndarray, z_arr: np.ndarray) -> int:
        """逐层漂移参考层。支持最佳焦面、当前层、自定义层；若参考层无有效识别，则回退到最接近层。"""
        valid_set = set(int(v) for v in valid_indices)
        mode = getattr(self, "mark_drift_ref_mode", tk.StringVar(value="最佳焦面")).get()
        ref = None
        if mode == "最佳焦面":
            try:
                if self.results:
                    ref = int(self.best_index())
            except Exception:
                ref = None
            if ref is None:
                ref = int(self.current_index)
        elif mode == "当前层":
            ref = int(self.current_index)
        else:
            try:
                ref = int(self.mark_drift_custom_layer.get())
            except Exception:
                ref = int(self.current_index)

        if ref in valid_set:
            return int(ref)
        if ref is not None and len(valid_indices) > 0:
            pos = int(np.argmin(np.abs(valid_indices.astype(int) - int(ref))))
            return int(valid_indices[pos])
        # fallback：选最接近有效 Z 中位数的层
        med_z = float(np.nanmedian(z_arr))
        pos = int(np.argmin(np.abs(z_arr - med_z)))
        return int(valid_indices[pos])

    def _select_mark_drift_display_mask(self, idx_arr: np.ndarray, ref_layer: int) -> np.ndarray:
        """根据 UI 选择返回漂移图显示层 mask；计算结果仍保留全层。"""
        mask = np.ones(len(idx_arr), dtype=bool)
        if getattr(self, "mark_drift_scope_mode", tk.StringVar(value="全部层")).get() == "参考层±N层":
            try:
                n = max(0, int(self.mark_drift_window_layers.get()))
            except Exception:
                n = 3
            mask = np.abs(idx_arr.astype(int) - int(ref_layer)) <= n
            if not np.any(mask):
                mask = np.ones(len(idx_arr), dtype=bool)
        return mask

    @staticmethod
    def _set_range_with_margin(ax, xdata=None, ydata=None, xlim=None, ylim=None):
        """给坐标轴设置带边距的范围，避免单点/小范围数据显示贴边。"""
        def span(v):
            arr = np.asarray(v, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return None
            lo = float(np.nanmin(arr)); hi = float(np.nanmax(arr))
            if abs(hi - lo) < 1e-12:
                pad = max(abs(lo) * 0.2, 0.05)
            else:
                pad = (hi - lo) * 0.12
            return lo - pad, hi + pad
        if xlim is None and xdata is not None:
            xlim = span(xdata)
        if ylim is None and ydata is not None:
            ylim = span(ydata)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

    def update_mark_drift_plot(self):
        if not self.mark_drift_results:
            return
        rows_all = sorted(self.mark_drift_results, key=lambda d: int(d.get("layer_index", 0)))
        idx_arr = np.array([int(r["layer_index"]) for r in rows_all], dtype=int)
        z_all = np.array([float(r["z_um"]) for r in rows_all], dtype=float)
        x_px_all = np.array([float(r["x_px"]) for r in rows_all], dtype=float)
        y_px_all = np.array([float(r["y_px"]) for r in rows_all], dtype=float)
        pix = float(rows_all[0].get("pixel_size_um", self.pixel_size_um.get())) if rows_all else float(self.pixel_size_um.get())

        # 上位机图像坐标原点在左上，像素 y 向下；漂移图要求 Y 向上为正，因此 ΔY = -(y-y_ref)*pix。
        ref_layer = self._mark_reference_index(idx_arr, z_all)
        ref_pos = int(np.where(idx_arr == ref_layer)[0][0])
        ref_x = float(x_px_all[ref_pos])
        ref_y = float(y_px_all[ref_pos])
        dx_all = (x_px_all - ref_x) * pix
        dy_up_all = -(y_px_all - ref_y) * pix
        dr_all = np.sqrt(dx_all * dx_all + dy_up_all * dy_up_all)

        display_mask = self._select_mark_drift_display_mask(idx_arr, ref_layer)
        rows = [r for r, m in zip(rows_all, display_mask) if bool(m)]
        z = z_all[display_mask]
        dx = dx_all[display_mask]
        dy_up = dy_up_all[display_mask]
        dr = dr_all[display_mask]
        idx_show_arr = idx_arr[display_mask]
        local_mode = getattr(self, "mark_drift_scope_mode", tk.StringVar(value="全部层")).get()
        axis_mode = getattr(self, "mark_drift_axis_mode", tk.StringVar(value="局部自动缩放")).get()

        self._make_mark_drift_axes()

        # 左上：当前/参考层图像，叠加 ROI 和参考识别边界。
        idx_show = max(0, min(ref_layer, len(self.image_paths) - 1))
        raw = safe_read_gray(self.image_paths[idx_show], normalize_per_image=False)
        disp = resize_max_dim(normalize_to_255(raw), 1200)
        self.ax_img.imshow(disp, cmap="gray")
        self.ax_img.set_title(f"参考层图像 / layer={ref_layer}, Z={self.z_value(ref_layer):.3f} μm", fontsize=10)
        self.ax_img.axis("off")
        sx = disp.shape[1] / raw.shape[1]
        sy = disp.shape[0] / raw.shape[0]
        if self.roi is not None:
            x, y, w, h = self.roi
            self.ax_img.add_patch(plt_rectangle(x * sx, y * sy, w * sx, h * sy, edgecolor="#2563EB"))
        ref_row = rows_all[ref_pos]
        bx = float(ref_row.get("roi_x", 0.0)); by = float(ref_row.get("roi_y", 0.0))
        pts = ref_row.get("boundary_points_roi")
        if pts is not None:
            pts_arr = np.asarray(pts, dtype=float)
            if pts_arr.ndim == 2 and pts_arr.shape[0] > 1:
                self.ax_img.plot((pts_arr[:, 0] + bx) * sx, (pts_arr[:, 1] + by) * sy, linewidth=1.0)
        self.ax_img.plot([ref_x*sx-8, ref_x*sx+8], [ref_y*sy, ref_y*sy], linewidth=1.2)
        self.ax_img.plot([ref_x*sx, ref_x*sx], [ref_y*sy-8, ref_y*sy+8], linewidth=1.2)

        # 右上：XY 漂移轨迹。X 右正，Y 上正。
        sc = self.ax_curve.scatter(dx, dy_up, c=z, s=38)
        self.ax_curve.plot(dx, dy_up, linewidth=1.0, alpha=0.75)
        for i in range(len(dx) - 1):
            self.ax_curve.annotate("", xy=(dx[i+1], dy_up[i+1]), xytext=(dx[i], dy_up[i]), arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.7))
        self.ax_curve.scatter([0], [0], marker="+", s=90, label=f"ref layer {ref_layer}")
        self.ax_curve.axhline(0, linewidth=0.8, alpha=0.5)
        self.ax_curve.axvline(0, linewidth=0.8, alpha=0.5)
        self.ax_curve.set_aspect("equal", adjustable="box")
        self.ax_curve.set_title(f"XY 中心漂移轨迹（{local_mode}）", fontsize=10)
        self.ax_curve.set_xlabel("ΔX / μm")
        self.ax_curve.set_ylabel("ΔY / μm")
        self.ax_curve.grid(True, alpha=0.3)
        self.ax_curve.legend(loc="best", fontsize=8)
        if axis_mode == "全局固定":
            lim = max(float(np.nanmax(np.abs(dx_all))) if dx_all.size else 0.0, float(np.nanmax(np.abs(dy_up_all))) if dy_up_all.size else 0.0, 0.05)
            lim *= 1.15
            self.ax_curve.set_xlim(-lim, lim)
            self.ax_curve.set_ylim(-lim, lim)
        else:
            lim = max(float(np.nanmax(np.abs(dx))) if dx.size else 0.0, float(np.nanmax(np.abs(dy_up))) if dy_up.size else 0.0, 0.02)
            lim *= 1.25
            self.ax_curve.set_xlim(-lim, lim)
            self.ax_curve.set_ylim(-lim, lim)
        self.fig.colorbar(sc, ax=self.ax_curve, fraction=0.046, pad=0.04, label="Z / μm")

        # 左下：ΔX/ΔY vs Z。
        self.ax_prev.plot(z, dx, marker="o", label="ΔX")
        self.ax_prev.plot(z, dy_up, marker="s", label="ΔY 上正")
        self.ax_prev.axhline(0, linewidth=0.8, alpha=0.5)
        self.ax_prev.axvline(float(z_all[ref_pos]), linestyle=":", linewidth=1.0, label="reference Z")
        self.ax_prev.set_title("ΔX / ΔY - Z 曲线", fontsize=10)
        self.ax_prev.set_xlabel("Z / μm")
        self.ax_prev.set_ylabel("中心偏移 / μm")
        self.ax_prev.grid(True, alpha=0.3)
        self.ax_prev.legend(loc="best", fontsize=8)
        if axis_mode == "全局固定":
            self._set_range_with_margin(self.ax_prev, z_all, np.r_[dx_all, dy_up_all])
        else:
            self._set_range_with_margin(self.ax_prev, z, np.r_[dx, dy_up])

        # 右下：总漂移 dr vs Z。
        self.ax_next.plot(z, dr, marker="o")
        self.ax_next.axvline(float(z_all[ref_pos]), linestyle=":", linewidth=1.0)
        self.ax_next.set_title("总漂移量 dr - Z 曲线", fontsize=10)
        self.ax_next.set_xlabel("Z / μm")
        self.ax_next.set_ylabel("dr = sqrt(ΔX²+ΔY²) / μm")
        self.ax_next.grid(True, alpha=0.3)
        if axis_mode == "全局固定":
            self._set_range_with_margin(self.ax_next, z_all, dr_all)
        else:
            self._set_range_with_margin(self.ax_next, z, dr)

        # 图内提示当前显示层范围，便于避免误判。
        if len(idx_show_arr) > 0:
            note = f"显示层数：{len(idx_show_arr)} / {len(idx_arr)}；layer {int(np.min(idx_show_arr))}~{int(np.max(idx_show_arr))}；坐标轴：{axis_mode}"
            self.ax_img.text(0.01, 0.02, note, transform=self.ax_img.transAxes, fontsize=8, color="yellow", bbox=dict(facecolor="black", alpha=0.45, edgecolor="none"))
        self.canvas.draw_idle()
    def _update_mark_drift_text(self, errors, pix: float):
        rows = sorted(self.mark_drift_results, key=lambda d: int(d.get("layer_index", 0)))
        idx_arr = np.array([int(r["layer_index"]) for r in rows], dtype=int)
        z = np.array([float(r["z_um"]) for r in rows], dtype=float)
        x_px = np.array([float(r["x_px"]) for r in rows], dtype=float)
        y_px = np.array([float(r["y_px"]) for r in rows], dtype=float)
        ref_layer = self._mark_reference_index(idx_arr, z)
        ref_pos = int(np.where(idx_arr == ref_layer)[0][0])
        dx = (x_px - x_px[ref_pos]) * pix
        dy_up = -(y_px - y_px[ref_pos]) * pix
        dr = np.sqrt(dx * dx + dy_up * dy_up)
        text = []
        text.append(f"逐层 ROI 中心识别完成：有效 {len(rows)} / {len(self.image_paths)} 层")
        text.append(f"坐标方向：图像原点在左上；漂移图中 X 向右为正，Y 向上为正")
        text.append(f"参考层：layer {ref_layer}，Z={z[ref_pos]:.4f} μm，参考点 X={x_px[ref_pos]:.3f}px，Y={y_px[ref_pos]:.3f}px")
        try:
            display_mask = self._select_mark_drift_display_mask(idx_arr, ref_layer)
            show_idx = idx_arr[display_mask]
            text.append(f"当前漂移图显示：{self.mark_drift_scope_mode.get()}，显示 {len(show_idx)} / {len(idx_arr)} 层；坐标轴：{self.mark_drift_axis_mode.get()}")
        except Exception:
            pass
        text.append(f"ΔX范围：{np.nanmin(dx):.6f} ~ {np.nanmax(dx):.6f} μm")
        text.append(f"ΔY范围：{np.nanmin(dy_up):.6f} ~ {np.nanmax(dy_up):.6f} μm")
        text.append(f"dr最大值：{np.nanmax(dr):.6f} μm；dr RMS：{np.sqrt(np.nanmean(dr**2)):.6f} μm")
        if errors:
            text.append("")
            text.append(f"失败层数：{len(errors)}，前 8 个如下：")
            for i, name, err in errors[:8]:
                text.append(f"- layer {i}: {name} | {err}")
        if hasattr(self, "mark_label"):
            self.mark_label.config(text="\n".join(text))



    def build_topography(self):
        if not self.image_paths:
            messagebox.showwarning("未选择图片", "请先选择 Z-stack 图片文件夹。")
            return
        if self.is_topo_analyzing:
            messagebox.showinfo("正在计算", "当前高度图还在计算中，请等本轮完成。")
            return
        try:
            pix = float(self.pixel_size_um.get())
            if pix <= 0:
                raise ValueError
        except Exception:
            messagebox.showwarning("Pixel size 无效", "请填写大于 0 的 pixel size，单位 μm/pixel。")
            return
        try:
            grid_px = int(self.topo_grid_px.get())
            if grid_px <= 1:
                raise ValueError
        except Exception:
            messagebox.showwarning("网格尺寸无效", "请填写大于 1 的网格尺寸，单位 pixel。建议先用 32 或 64。")
            return

        try:
            sigma_filter = float(self.topo_sigma_filter_var.get())
            if sigma_filter < 0:
                raise ValueError
        except Exception:
            messagebox.showwarning("残差滤波参数无效", "请填写大于等于 0 的 nσ 数值；0 表示关闭滤波。")
            return

        roi = self.roi if (self.topo_use_roi_var.get() and self.roi is not None) else None
        exclude_rois = list(self.exclude_rois)
        mode = self.topo_algorithm_var.get()
        self.is_topo_analyzing = True
        self.config(cursor="watch")
        self.topo_button.config(state="disabled")
        self.topo_progress_var.set(f"高度图计算中：0 / {len(self.image_paths)}")
        self.topo_result_text.delete("1.0", tk.END)
        self.topo_result_text.insert(tk.END, "正在后台生成高度图。第一次建议用 grid=32 或 64 先验证趋势。")
        if hasattr(self, "output_tabs"):
            try:
                self.output_tabs.select(2)
            except tk.TclError:
                pass

        worker = threading.Thread(target=self._topography_worker, args=(roi, pix, grid_px, mode, exclude_rois, sigma_filter), daemon=True)
        worker.start()

    def _topography_worker(self, roi, pix: float, grid_px: int, mode: str, exclude_rois: List[Tuple[int, int, int, int]], sigma_filter: float):
        try:
            def cb(i, total, done_points=0, total_points=0):
                if total_points and done_points:
                    self.after(0, lambda dp=done_points, tp=total_points: self.topo_progress_var.set(f"高度图网格计算中：{dp} / {tp} 个网格点"))
                else:
                    self.after(0, lambda i=i, total=total: self.topo_progress_var.set(f"高度图读取中：{i} / {total} 层"))
            result = build_height_map_from_zstack(
                self.image_paths,
                self.z_values(),
                roi=roi,
                pixel_size_um=pix,
                grid_px=grid_px,
                mode=mode,
                exclude_rois=exclude_rois,
                sigma_filter=sigma_filter,
                progress_callback=cb,
            )
            self.after(0, lambda: self._finish_topography(result, None))
        except Exception as e:
            err = f"{e}\n\n{traceback.format_exc()}"
            self.after(0, lambda: self._finish_topography(None, err))

    def _finish_topography(self, result, error: Optional[str]):
        try:
            if error:
                messagebox.showerror("高度图计算失败", error)
                self.topo_progress_var.set("高度图计算失败")
                return
            self.topo_result = result
            self.update_topography_plot()
            self.update_topography_text()
            stats = result["stats"]
            self.topo_progress_var.set(
                f"完成：有效点 {stats['valid_count']} / {stats['total_count']}，滤波剔除 {stats.get('sigma_outlier_count', 0)}，有效率 {stats['valid_ratio']*100:.1f}%"
            )
        finally:
            self.is_topo_analyzing = False
            self.config(cursor="")
            self.topo_button.config(state="normal")

    def update_topography_text(self):
        if not self.topo_result:
            return
        stats = self.topo_result["stats"]
        roi = stats.get("roi", None)
        text = []
        text.append(f"高度图算法：{stats['mode']}")
        if np.isfinite(float(stats.get('fit_r2_mean', np.nan))):
            text.append(f"逐点高斯拟合平均 R²：{stats.get('fit_r2_mean'):.4f}")
        if np.isfinite(float(stats.get('confidence_mean', np.nan))):
            text.append(f"融合/高度图平均置信度：{stats.get('confidence_mean'):.4f}")
        if stats.get('fused_gray_available'):
            text.append("已生成：融合灰度图 / 高度图 / 残差图 / 质量图")
        text.append(f"Pixel size：{stats['pixel_size_um']:.6f} μm/pixel")
        text.append(f"网格尺寸：{stats['grid_px']} px")
        if roi:
            text.append(f"分析区域 ROI/FOV：x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]} px")
        text.append(f"有效高度点：{stats['valid_count']} / {stats['total_count']} ({stats['valid_ratio']*100:.1f}%)")
        text.append(f"滤波前有效点：{stats.get('pre_filter_valid_count', stats['valid_count'])}")
        text.append(f"排除网格点：{stats.get('excluded_count', 0)}")
        if stats.get("exclude_rois"):
            text.append(f"排除区域 Mask：{len(stats.get('exclude_rois', []))} 个")
        text.append(f"sigma残差滤波：{stats.get('sigma_filter', 0):.3g}σ；剔除异常点 {stats.get('sigma_outlier_count', 0)}；迭代 {stats.get('sigma_iterations', 0)} 次")
        if stats.get('sigma_filter', 0) > 0 and np.isfinite(float(stats.get('residual_sigma_um', np.nan))):
            text.append(f"- 最终残差 σ：{stats['residual_sigma_um']:.6f} μm；阈值：±{stats['residual_threshold_um']:.6f} μm")
        text.append("")
        text.append("平面拟合：Z = a·X + b·Y + c，X/Y/Z 单位均为 μm")
        text.append("坐标系：左手系，X向右，Y向里，Z向下")
        text.append(f"- a=dZ/dX：{stats['slope_x_dzdx']:.8e}")
        text.append(f"- b=dZ/dY：{stats['slope_y_dzdy']:.8e}")
        text.append(f"- c：{stats['plane_c_um']:.6f} μm")
        text.append(f"- Ry = -atan(dZ/dX)：{stats['ry_deg']:.6f}° / {stats['ry_mrad']:.4f} mrad")
        text.append(f"- Rx =  atan(dZ/dY)：{stats['rx_deg']:.6f}° / {stats['rx_mrad']:.4f} mrad")
        text.append("")
        text.append("高度 / 面型指标：")
        text.append(f"- 原始高度 Z min：{stats['z_min_um']:.6f} μm")
        text.append(f"- 原始高度 Z max：{stats['z_max_um']:.6f} μm")
        text.append(f"- 原始高度 Z mean：{stats['z_mean_um']:.6f} μm")
        text.append(f"- TTV / 原始全局高度差：{stats['ttv_um']:.6f} μm")
        text.append(f"- 去平面后面型 PV：{stats['pv_residual_um']:.6f} μm")
        text.append(f"- 去平面后面型 RMS：{stats['rms_residual_um']:.6f} μm")
        text.append("")
        text.append("说明：")
        text.append("1. 当前版本为网格高度图；快速投影/峰值Z速度快，逐点高斯更准但会明显更慢。")
        text.append("2. TTV 这里按原始高度 max-min 计算；面型 PV/RMS 是扣除最佳拟合平面后的残差。")
        text.append("3. Rx/Ry 符号按左手坐标系修正：X向右、Y向里、Z向下；因此 Ry=-atan(dZ/dX)，Rx=atan(dZ/dY)。")
        text.append("4. sigma残差滤波会剔除局部异常高度点；0 表示关闭滤波。滤波后点参与平面拟合、TTV、PV、RMS。")
        text.append("5. 若有效率低，通常是背景区域没有明显 Z 向峰值，建议框选样品 ROI 或增大 grid。")
        text.append("6. 若 ROI 内有夹具、坏点、孔洞或异常反光区，可在第三页添加排除 Mask；排除点不参与平面拟合和面型统计。")
        self.topo_result_text.delete("1.0", tk.END)
        self.topo_result_text.insert(tk.END, "\n".join(text))
        if hasattr(self, "output_tabs"):
            try:
                self.output_tabs.select(2)
            except tk.TclError:
                pass

    def update_topography_plot(self):
        if not self.topo_result:
            return
        self._make_topography_axes()
        height = np.asarray(self.topo_result["height_map"], dtype=float)
        residual = np.asarray(self.topo_result["residual_map"], dtype=float)
        fused_weighted = np.asarray(self.topo_result.get("fused_weighted_map", np.full_like(height, np.nan)), dtype=float)
        confidence = np.asarray(self.topo_result.get("confidence_map", np.full_like(height, np.nan)), dtype=float)
        x_um = np.asarray(self.topo_result["x_um"], dtype=float)
        y_um = np.asarray(self.topo_result["y_um"], dtype=float)
        stats = self.topo_result["stats"]
        roi = stats.get("roi", None)

        # 左上：高斯融合灰度图。若不是亮度高斯算法导致融合图为空，则回退显示当前层原图。
        if np.any(np.isfinite(fused_weighted)):
            im0 = self.ax_img.imshow(fused_weighted, cmap="gray", origin="upper", aspect="auto")
            self.ax_img.set_title("高斯加权融合灰度图 / Gaussian EDF", fontsize=10)
            self.ax_img.set_xlabel("X grid")
            self.ax_img.set_ylabel("Y grid")
            self.fig.colorbar(im0, ax=self.ax_img, fraction=0.046, pad=0.04)
        else:
            idx = max(0, min(self.current_index, len(self.image_paths) - 1))
            raw = safe_read_gray(self.image_paths[idx], normalize_per_image=False)
            disp = resize_max_dim(normalize_to_255(raw), 1200)
            self.ax_img.imshow(disp, cmap="gray")
            self.ax_img.set_title(f"当前层图像 / Z={self.z_value(idx):.3f} μm", fontsize=10)
            self.ax_img.axis("off")
            sx = disp.shape[1] / raw.shape[1]
            sy = disp.shape[0] / raw.shape[0]
            if roi:
                rect = plt_rectangle(roi[0] * sx, roi[1] * sy, roi[2] * sx, roi[3] * sy, edgecolor="#2563EB")
                self.ax_img.add_patch(rect)
            for ex_x, ex_y, ex_w, ex_h in stats.get("exclude_rois", []):
                ex_rect = plt_rectangle(ex_x * sx, ex_y * sy, ex_w * sx, ex_h * sy, edgecolor="#EF4444", linestyle="--")
                self.ax_img.add_patch(ex_rect)

        im1 = self.ax_curve.imshow(height, cmap="viridis", origin="upper", aspect="auto")
        sigma_out = np.asarray(self.topo_result.get("sigma_outlier_mask", np.zeros_like(height, dtype=bool)))
        if sigma_out.shape == height.shape and np.any(sigma_out):
            oy, ox = np.where(sigma_out)
            self.ax_curve.scatter(ox, oy, marker="x", s=18, label="sigma outlier")
            self.ax_curve.legend(loc="upper right", fontsize=7)
        self.ax_curve.set_title("FOV 高度图 Z / μm", fontsize=10)
        self.ax_curve.set_xlabel("X grid")
        self.ax_curve.set_ylabel("Y grid")
        self.fig.colorbar(im1, ax=self.ax_curve, fraction=0.046, pad=0.04)

        im2 = self.ax_prev.imshow(residual, cmap="coolwarm", origin="upper", aspect="auto")
        self.ax_prev.set_title(f"去平面残差 / PV={stats['pv_residual_um']:.4f} μm", fontsize=10)
        self.ax_prev.set_xlabel("X grid")
        self.ax_prev.set_ylabel("Y grid")
        self.fig.colorbar(im2, ax=self.ax_prev, fraction=0.046, pad=0.04)

        # 右下：拟合质量/置信度图；比 3D surface 更便于判断哪些区域融合图和高度图可信。
        if np.any(np.isfinite(confidence)):
            im3 = self.ax_next.imshow(confidence, cmap="magma", origin="upper", aspect="auto", vmin=0, vmax=1)
            self.ax_next.set_title("高斯拟合质量 / 置信度", fontsize=10)
            self.ax_next.set_xlabel("X grid")
            self.ax_next.set_ylabel("Y grid")
            self.fig.colorbar(im3, ax=self.ax_next, fraction=0.046, pad=0.04)
        else:
            self.ax_next.text(0.5, 0.5, "当前算法未生成拟合质量图", ha="center", va="center", transform=self.ax_next.transAxes)
            self.ax_next.set_title("拟合质量 / 置信度", fontsize=10)
            self.ax_next.axis("off")
        self.canvas.draw_idle()

    def export_fused_gray_png(self):
        if not self.topo_result:
            messagebox.showwarning("无高度图", "请先生成 FOV 高度图。")
            return
        fused = np.asarray(self.topo_result.get("fused_weighted_map", []), dtype=float)
        if fused.size == 0 or not np.any(np.isfinite(fused)):
            messagebox.showwarning("无融合图", "当前结果没有可导出的高斯融合灰度图。请使用共聚焦亮度高斯拟合算法生成高度图。")
            return
        out = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")], title="保存高斯融合灰度图 PNG")
        if not out:
            return
        img = fused.copy()
        finite = np.isfinite(img)
        if not np.any(finite):
            messagebox.showwarning("无有效像素", "高斯融合图没有有效像素。")
            return
        lo = float(np.nanpercentile(img[finite], 1))
        hi = float(np.nanpercentile(img[finite], 99))
        if hi <= lo:
            lo = float(np.nanmin(img[finite])); hi = float(np.nanmax(img[finite]))
        img8 = np.zeros_like(img, dtype=np.uint8)
        if hi > lo:
            img8[finite] = np.clip((img[finite] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", img8)
        if not ok:
            messagebox.showerror("导出失败", "PNG 编码失败。")
            return
        buf.tofile(out)
        messagebox.showinfo("导出完成", f"已保存：{out}")

    def export_topography_csv(self):
        if not self.topo_result:
            messagebox.showwarning("无高度图", "请先生成 FOV 高度图。")
            return
        out = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], title="保存高度点 CSV")
        if not out:
            return
        height = np.asarray(self.topo_result["height_map"], dtype=float)
        residual = np.asarray(self.topo_result["residual_map"], dtype=float)
        plane = np.asarray(self.topo_result["plane_map"], dtype=float)
        x_um = np.asarray(self.topo_result["x_um"], dtype=float)
        y_um = np.asarray(self.topo_result["y_um"], dtype=float)
        x_px = np.asarray(self.topo_result["x_px"], dtype=float)
        y_px = np.asarray(self.topo_result["y_px"], dtype=float)
        rows = []
        for iy in range(height.shape[0]):
            for ix in range(height.shape[1]):
                rows.append({
                    "ix": ix,
                    "iy": iy,
                    "x_px": x_px[ix],
                    "y_px": y_px[iy],
                    "x_um": x_um[ix],
                    "y_um": y_um[iy],
                    "z_um": height[iy, ix],
                    "plane_z_um": plane[iy, ix],
                    "residual_um": residual[iy, ix],
                    "excluded": bool(self.topo_result.get("excluded_grid_mask", np.zeros_like(height, dtype=bool))[iy, ix]),
                    "inlier": bool(self.topo_result.get("inlier_mask", np.zeros_like(height, dtype=bool))[iy, ix]),
                    "sigma_outlier": bool(self.topo_result.get("sigma_outlier_mask", np.zeros_like(height, dtype=bool))[iy, ix]),
                    "fit_r2": float(np.asarray(self.topo_result.get("fit_r2_map", np.full_like(height, np.nan)), dtype=float)[iy, ix]),
                    "confidence": float(np.asarray(self.topo_result.get("confidence_map", np.full_like(height, np.nan)), dtype=float)[iy, ix]),
                    "fused_peak_gray": float(np.asarray(self.topo_result.get("fused_peak_map", np.full_like(height, np.nan)), dtype=float)[iy, ix]),
                    "fused_weighted_gray": float(np.asarray(self.topo_result.get("fused_weighted_map", np.full_like(height, np.nan)), dtype=float)[iy, ix]),
                })
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        messagebox.showinfo("导出完成", f"已保存：{out}")

    def export_csv(self):
        if not self.results:
            messagebox.showwarning("无结果", "请先完成分析。")
            return
        out = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], title="保存 CSV")
        if not out:
            return
        rows = []
        for r in self.results:
            row = {
                "index": r.index,
                "z_um": self.z_value(r.index),
                "filename": r.filename,
                "path": r.path,
                "algorithm": self.algorithm_var.get(),
                "algorithm_score": r.combined_score,
            }
            row.update(r.metrics)
            rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        messagebox.showinfo("导出完成", f"已保存：{out}")

    def export_report_png(self):
        if not self.results:
            messagebox.showwarning("无结果", "请先完成分析。")
            return
        out = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")], title="保存报告图片")
        if not out:
            return
        self.fig.savefig(out, dpi=180, bbox_inches="tight")
        messagebox.showinfo("导出完成", f"已保存：{out}")


def plt_rectangle(x, y, w, h, edgecolor=None, linestyle="-"):
    # 放在函数里，避免全局导入 patch 时污染命名空间
    import matplotlib.patches as patches
    kwargs = {"fill": False, "linewidth": 1.8, "linestyle": linestyle}
    if edgecolor is not None:
        kwargs["edgecolor"] = edgecolor
    return patches.Rectangle((x, y), w, h, **kwargs)


def plt_circle(x, y, r):
    import matplotlib.patches as patches
    return patches.Circle((x, y), r, fill=False, linewidth=1.8)


def plt_polygon(points):
    import matplotlib.patches as patches
    return patches.Polygon(points, closed=True, fill=False, linewidth=1.8)


if __name__ == "__main__":
    app = ZStackFocusAnalyzer()
    app.mainloop()
