# Z-stack Focus Analyzer

Z-stack 多焦面分析工具。当前仓库版本来自 `zstack_focus_analyzer_v2_6_fast_projection.zip`，主版本为 v2.6 fast projection。

## 当前版本

- `zstack_focus_analyzer.py`: Tkinter GUI 主程序。
- `requirements.txt`: Python 依赖。

v2.6 重点包括：

- 快速最大亮度投影。
- 快速峰值 Z 高度图。
- 快速峰值 Z + 抛物线插值。
- 保留逐点高斯拟合，用于更精密的正式分析。
- 支持 ROI、mask 排除区、平面拟合、Rx/Ry、PV、RMS、TTV 等分析输出。

## 运行

```powershell
python -m pip install -r requirements.txt
python .\zstack_focus_analyzer.py
```

## 说明

这是网页版 ChatGPT 生成并打包的多焦面分析工具版本。本次上传将压缩包内容展开到仓库根目录，方便后续直接 clone 和继续开发。
