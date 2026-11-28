# ISSUE 7 参数绑定复核

这是纯主机端的 GTSim 配置诊断，与实卡 benchmark 分开运行。不需要 PPU、SDK、TileLang 安装或内网实测数据，但需要实现 ISSUE 7 工作描述的 GTSim 源码及其构建产物。

```bash
python3 gtsim_issue7_probe.py \
  --gtsim ../GTSim-ppu \
  --sim ../GTSim-ppu/build/gtsim_ir \
  --out results-issue7-binding
```

输出目录必须是新目录。脚本仅修改输出目录里的 profile 副本，保存各场景的配置、DAG、日志、时间线和 `summary.json`。脚本是观察性诊断，不会因为复现了已知问题就返回“修复验收通过”；检查每个场景的 `returncode` 和下述指标。

## 1. 同一矩阵分型的时序来源

三个输入都描述 BF16 的 `(m,n,k)=(16,16,16)`、FP32 累加：TileLang 当前采用的旧 `mma` 入口、新 `mma.bfloat16.float32` 入口、逻辑 `matrix_tile` 入口。

将 `cube.bfloat16.float32` 的 II/L 改为 2/32 后，对比 `end - issue` 与 `fu_service_cycles`。2026-11-28 复核的工作区中，新入口得到 32/2，旧入口仍为 16/1。旧入口继续读取 `tc02_*`，导致同一分型的校准参数不统一。修复应统一有效时序来源；如果保留两套键，应明确兼容优先级或拒绝冲突，不能静默按入口选择不同数值。

## 2. 新 cost 与旧 conversion 的优先级

对 64 个 FP32→FP16 转换元素，新表指定 64 元素/周期、tail=2；旧键指定 8 元素/周期、tail=10。分别运行仅新键、两种键同时存在的配置。

同次复核中，仅新键的服务时间为 1 周期，同时存在时变为 8 周期，`cost_source` 变为 `legacy_conversion_assumption`。兼容翻译发生在新表读取之后并覆盖新值。建议新配置优先，或者明确报告冲突。若选择拒绝冲突，第二个场景非零退出且日志说明冲突，也属于正确结果。

这两项是参数绑定问题，不是硬件测量结论。复核时既有 GTSim CTest 36/36 通过；默认配置相同及缺少冲突组合，使这些问题未被原有回归覆盖。未上传设备数据或本地完整运行日志。
