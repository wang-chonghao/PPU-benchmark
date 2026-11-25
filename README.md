# PPU 1.5 单卡校准 benchmark

在真武 M890P 上测 GTSim 所需的设备资源、持续带宽、矩阵和向量吞吐，并用四个 TileLang kernel 对照实机与仿真。八卡服务器上每次只占用一张指定卡。设备端无需 GTSim、外网或管理员权限。脚本沿用机器当前的频率策略。

## 在内网运行

进入带 PPU SDK 的 Linux 环境。原生测试使用 Python 3.8+ 标准库和 SDK 的 CUDA 兼容 `nvcc`；同源算子还需要能在该卡执行的 PyTorch 和 `t-head/tilelang-for-sail` PPU 版本。JIT 接口按 `cd65f13d` 的 `tvm_ffi` 路径设计。ACU 采集需要 SDK 的 `acu`。如果只有 SDK，先运行 `--suite native` 即可。[PPU 工具链对应关系](https://help.aliyun.com/en/document_detail/2864731.html) · [TileLang PPU fork](https://github.com/t-head/tilelang-for-sail)

第一轮建议用一张空闲卡，收集普通计时和单独的 ACU 报告。下面以物理卡 0 为例。如果作业系统已设置可见设备，保留它的映射；程序使用进程内的逻辑设备 0。`--smi-id` 填对应的物理编号，用 `device.csv` 的 PCI 地址与 `inventory.log` 交叉确认。

```bash
CUDA_VISIBLE_DEVICES=0 python3 run.py board results-card0 --smi-id 0 --acu
python3 run.py pack results-card0 results-card0.zip
```

带回 **`results-card0.zip`** 即可。即使部分测试失败也可以打包，`manifest.json` 列出各阶段状态和时间，对应的 `.stderr` 和 `.command.json` 保存原因和原始命令。每次使用新目录，保留之前的数据。只想快速检查环境或分开运行时可用：

```bash
CUDA_VISIBLE_DEVICES=0 python3 run.py board results-native --suite native --max-mib 32 --smi-id 0
CUDA_VISIBLE_DEVICES=0 python3 run.py board results-pair --suite paired --cases gemm_aiu_on gemm_aiu_off --smi-id 0 --acu
```

32 MiB 适合快速检查；正式带宽扫描默认到 512 MiB/数组，两数组共约 1 GiB，可用 `--max-mib 1024` 或 `2048` 扩展。原生测试默认编译目标为兼容 ISA `sm_89`，TileLang 目标为 `ppu_15`。可用 `--nvcc /path/to/nvcc` 指定 PPU SDK 编译器。为了核对板间差异，可在另一张空闲卡用相同参数复测，输出到新目录。

**频率对照。** `clocks.log` 保存指定卡的 `ppu-smi dmon` 采样，`--clock-note` 只记录实验标签。若机房已给同一张卡设置 1300 MHz，可以同样命令另跑一轮，加上 `--clock-note site-locked-1300`。记录机房的实际设置，并用监控核对运行工况。启动时的标称时钟和每秒一次的板卡采样均无法证明每个短 kernel 的实际频率。ACU 统一使用 `--clock-control none`。普通计时在无 profiler 的进程中完成。

## 各测试能校准什么

| 产物 | 工作负载与口径 | 对应 GTSim 参数 |
| --- | --- | --- |
| `device.csv`、`occupancy.csv` | 记录 HBM 字节、CU、warp、TSM、运行时寄存器计数单位。以 1/8/32/64 条独立链构造 kernel，读回编译器实际分配的每线程寄存器、local 字节；扫描线程数和动态 TSM，查询 occupancy API 的理论 block/CU。 | CU 资源和驻留上限。结合 ACU 的 Block Limit Registers/Shared Mem/Warps 确认约束来源，再换算寄存器字节。API 上限与一次执行中实际并驻的 block 数分别记录。 |
| `stream.csv` | 1–512 MiB 默认工作集，分别测读、写、copy。以设备 event 计时，保存五组原始样本。读 kernel 每线程写一个归约结果，写入字节另列；copy 合计读写两侧。 | HBM/LLC 有效带宽、工作集拐点。表内是程序逻辑字节。用指定工作集单 launch 的 ACU 下游字节、命中率确认流量实际经过哪一级。分不出 CE 私有容量时保留观测拐点。 |
| `shared.csv` | 对 32 位共享内存读扫描 lane 间距 0/1/2/4/8/16/32 个 word；0 为同地址广播。用两种循环长度扣除固定初始化和栅栏成本。 | TSM bank 冲突相对代价和读服务率。跨 warp 的并发与循环里的累加也影响绝对值。 |
| `math.csv`、`mma.csv` | FP32 FMA、`exp2(-x)`、`s/(1+x)`、BF16/FP8 warp MMA。1 warp/1 条依赖链测时间斜率，多 CU/4 条独立链测吞吐。32/128/512 步各五组。MMA 校验全 1 矩阵的累加结果。 | SIMD/SFU、矩阵依赖延迟和饱和率。数学循环记录每元素递推次数；除法还包含一次加法。兼容 MMA 为 m16n8k16 或 m16n8k32，模型 TC02 为 m16n16。先比 FLOP/s，再用 PASM 确认展开后推断单条 II。 |
| `gemm_aiu_on/`、`gemm_aiu_off/` | BF16 GEMM，M/N/K=128/64/96，两个 block，128 线程/block，tile=64×64×32，两级流水，只切换 AIU lowering。 | AIU 与普通搬运的相对表现、计算和数据到达的依赖。可核对源码、动态指令数、缓存字节及耗时差。 |
| `gemm_multiwave/`、`vector/` | 前者用相同 tile 计算 640×512×96，grid 共 80 个 block，可考察 39 CU 上的分配与并驻；后者为 10240 个 FP32 元素计算 `(A+B)*(A-B)`，80 个 block。 | 把小 GEMM 得到的参数用于更多 block 和不同工作负载。80 个 block 的实际波数取决于资源与调度，需查驻留数据。 |

原生 CSV 的时间单位为 µs/launch。`*-summary.json` 保留各组中位数、最小/最大值和逻辑服务率。计算依赖斜率可用 `(t512 − t128) × 实测核心MHz / 384` 周期/递推步，前提是同卡、同操作、同并行度、工况稳定。每步可能包含多条机器指令。内存绝对延迟、独立 AIU 命令启动延迟、各缓存写回策略仍需专门实验；本套件不会自动覆盖设备配置。

## 同源算子与流水图核对

指定 `--acu` 后，原生测试还会对 8/64 条 FMA 链的寄存器压力变体和 BF16/FP8 的四累加链 MMA 各采一个正式 launch。寄存器实验与 `occupancy.csv` 的 256 线程、0 B 动态共享内存行对应；以编译器实际分配值和报告中的限制资源解释驻留数。

每个 TileLang 用例只编译一次，普通计时直接运行该 JIT 对象，同时从其 `artifact.device_mod` 保存 `device.json`，从该对象保存 `kernel.ppu.cc`。运行前逐元素对照独立 CPU 参考答案，然后在已分配的输入输出上测量。每个用例有七组热态时间，以及七组间隔 256 MiB 写扫后的单次时间；写扫在计时区间之外。这是驱逐尝试，是否已冷却 LLC 由 ACU 字节数判断。Python event 计时可能包含设备等待 host 提交的空隙，尤其是很短的 kernel。

`--acu` 为 TileLang 各用例另起进程，在 profiler start/stop 内只发起一次目标调用，并核对 profile 与普通计时的 TIR 和生成源码哈希。同时对 stream 的 32 MiB（较小扫描上限时取其上限）及最大工作集采集读、copy 报告。`.acurep` 可用官方 GUI 查看，`acu-*.csv` 可直接分析。默认 `--acu-cache none` 仍可能刷新 L1/L2；若需跟冷启动模拟对照，可在另一轮使用 `--acu-cache all`。官方工具报告 warp stall、PASM 计数、寄存器及流量；这些统计约束流水图，无法恢复每一条指令的逐周期时间戳。[ACU 时钟与缓存控制](https://help.aliyun.com/zh/document_detail/3012044.html) · [ACU 指标和 GUI](https://help.aliyun.com/zh/document_detail/3030030.html)

把 zip 解压到仿真机器。以下示例假设 `PPU-benchmark` 与工作版 `GTSim-ppu` 是相邻目录；其他布局可传 `--adaptor`，smoke 还可传 `--fixtures`。

```bash
python3 run.py smoke smoke-out --sim ../GTSim-ppu/build/gtsim_ir
python3 run.py replay results-card0 sim-out --sim ../GTSim-ppu/build/gtsim_ir
```

`smoke` 用 GTSim 仓内历史 AIU 开/关样例检查工具链。`replay` 只消费本次上板保存并通过正确性检查的 TIR，核对其 SHA-256；各用例输出模拟报告、block 0 的节点流水 TSV 和全 grid 的驻留 TSV。`comparison.json` 并列给出模拟周期、按配置核心频率换算的微秒数和两种实测 event 中位数。

比较时先对齐 ACU 和 TIR 的 grid、block、kernel 数、指令和搬运字节，再比较时间。把计算吞吐、依赖链和带宽微基准用于定参，预留 `gemm_multiwave` 与 `vector` 检查调整后的结果。整算子时间相近时，也要检查流水是否靠相反方向的误差抵消。
