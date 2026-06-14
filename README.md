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

## Demo 测试数据

`demo_zstack/` 内含 10 张共聚焦风格的 Z-stack 演示图（黑背景、黑色圆 mark、白色亮边缘），
亮度沿 Z 呈高斯峰（焦面在第 5/6 层），并带有轻微 X 向焦面倾斜，方便快速体验三个页面的功能。
可用 `make_demo_zstack.py` 重新生成：

```powershell
python .\make_demo_zstack.py
```

## 更新记录

- **修复**：快速最大亮度投影 / 快速峰值Z 等“快速投影/亮度”类高度图算法此前误用清晰度高频能量作为
  Z 向评分，导致名义“亮度投影”实际跑的是清晰度。现已统一改为使用原始灰度亮度，与“共聚焦亮度峰值”
  结果一致（清晰度类算法不受影响）。
- **修复**：高度图、逐层漂移的后台线程此前在子线程里读取 Tkinter 变量（Z 起点/步距等），存在崩溃风险。
  现改为在主线程先取好 Z 值和图片列表再传入 worker，子线程不再触碰任何 tk 变量。
- **优化**：切层显示不再为获取原图尺寸而重复解码整张大图（改用尺寸缓存），并给层选择滑块加了防抖，
  拖动大图时明显更顺。

## 说明

这是网页版 ChatGPT 生成并打包的多焦面分析工具版本。本次上传将压缩包内容展开到仓库根目录，方便后续直接 clone 和继续开发。
