# PPU 1.5 单卡校准 benchmark

新增 [DeepSeek v4.1 同源算子实卡套件](docs/DEEPSEEK_V41.md)：固定版本 DeepSeek
官方参考中的全部 6 类 TileLang kernel，12 个已有流水对齐/覆盖点和20个模型尺寸测试点。
它们不是“阿里官方 shape”，也不覆盖整模型全部 PyTorch/通信节点。测试原样使用参考
kernel，导出实测同一次编译的 device TIR、输入、正确性结果与耗时供 GTSim 比较。

```bash
python3 deepseek_v41.py plan --suite all
python3 deepseek_v41.py run results-dsv41-first --cases gemm_fp8
python3 deepseek_v41.py run results-dsv41-smoke --suite smoke --acu
python3 deepseek_v41.py run results-dsv41-model --suite model --acu
```

在机房已配置的 SAIL 容器内运行，完整环境、shape来源、频率/缓存口径和失败处理见上述文档。
`run.py --suite all` 不自动包含此独立套件。

在真武 M890P 上测 GTSim 所需的设备资源、持续带宽、矩阵和向量吞吐，并用四个 TileLang kernel 对照实机与仿真。八卡服务器上每次只占用一张指定卡。设备端无需 GTSim、外网或管理员权限。脚本沿用机器当前的频率策略。

## 新增：ISSUE 7 与存储层级补测

完整步骤、公式、失败处理及参数登记见 [内网校准手册](docs/INTRANET_CALIBRATION.md)。本轮约束是**数据不可外传**：所有原始数据、PASM、TIR 和分析结果均留在内网，下面的打包仅用于内网备份/转移。源码可离线带入；不要上传结果 ZIP。

新增 `supplement.py` / `calibration_probe.cu`：FP16/32/64 向量、FP16/BF16/FP8 矩阵的独立吞吐与依赖链；TSM 纯读/纯写/混合；TSM/缓存依赖读取、SHFL 和 barrier。新增 `paired_sweep.py` 检查同源 TileLang 的执行粒度、MNK 边界、tiling、EXP 和分支汇合。运行器保存失败、哈希、原始样本和命令；标准库分析器只生成候选参数，不修改配置。

```bash
# 在已分配的容器、单卡中运行。
python3 -m unittest test_runner test_supplement
python3 supplement.py run cal-smoke --suite smoke --smi-id 0
python3 supplement.py analyze cal-smoke
```

新增设备代码尚待内网 PPU SDK 编译与冒烟验证。FP32/FP4 原生 Cube、纯 SFU 结果延迟、AIU 命令隔离及队列深度不能由现有通用入口直接确认；手册明确给出所需证据、实验步骤和停止定值的条件，不把兼容指令或复合 kernel 冒充目标硬件参数。

另外提供纯主机端的 [GTSim ISSUE 7 参数绑定复核](docs/GTSIM_ISSUE7_REVIEW.md)及 `gtsim_issue7_probe.py`，用于检查 Cube 新旧入口和 conversion/cost 配置优先级；这不是实卡测试。

## 复跑 BM 的 HBM、LLC 和 L2 带宽

在已分配单卡的容器内，解压此前含 BM 源码的离线包，进入含 `run.py` 的目录。新补测源码包不重复附 BM 源码。沿用分配的设备映射。以下示例中，`0` 是这张卡在容器内对应的 `ppu-smi` 编号，按实际替换。

```bash
python3 run.py board results-bm-bw --suite bm-bw --acu --smi-id 0
python3 run.py board results-bm-ramp --suite bm-ramp --bm-blocks 2 16 32 64 128 256 --acu --smi-id 0
python3 run.py pack results-bm-bw bm-bw.zip
python3 run.py pack results-bm-ramp bm-ramp.zip
```

离线包已附四份原始 CUDA 源文件。使用 GitHub checkout 时，可在有外网的机器取得上游提交 `9221f6624d7c3525bb8cd31fd1c7f50967accf01`，将其 `BW/test_peak/{dram_bw,llc_bw,l2_bw}.cu` 和 `BW/test_dram_climb/dram_bw.cu` 按原层级放在 `BM/` 下，或通过 `--bm-root /path/to/BM` 指定。每次编译前核对源码哈希。两个套件需 PPU SDK；省去 `--acu` 可先取得普通 event 计时。这些长测单独启动。[上游命令及指标说明](https://gitcode.com/xxy20183691/BM/blob/main/BW/BW.md)

| 用例 | 工作集与启动方式 | 主要比较项 |
|---|---|---|
| `dram-auto-read/write/copy` | 每数组 16 GiB，512 threads，按各 kernel 的 occupancy 选 grid。预热 100 遍，一次正式 launch 扫 20 遍。copy 需要两个大数组。 | 对照上游 4 号卡 read 普通计时 2117 GB/s、ACU full 的 DRAM 读 2118 GB/s。写和 copy 使用相同程序；copy 按读写双向合计。 |
| `dram-fixed-read` | 16 GiB，1024 blocks × 256 threads，显式固定线程数 | 复核文档列出的历史 grid。当前源码省略 threads 会使用 512。 |
| `dram-512MiB-read` | 512 MiB，1248 blocks × 256 threads | 本仓增加的对照点，匹配此前 39 CU 的 scalar stream 测试尺寸和 grid，保留 BM 的 uint4 循环、预热和计时方式。 |
| `llc-read` | 32 MiB，256 threads，自动选择单波 grid，5000 遍且每遍移动起始位置 | 看 LLC fabric 读速率、L2/LLC hit 和 DRAM 读速率。上游 4 号卡为 273 blocks，实测 LLC 约 2.12 TB/s。 |
| `l2-read` | 64 blocks × 512 threads，每 block 独立读 100 KiB，申请 160 KiB shared，5000 遍 | 对照 L2 load 计数器。 |
| `bm-ramp` | 14 GiB，1024 threads，20 遍，分别读和写。去掉 `--bm-blocks` 可跑原始报告全部 40 种 grid、80 个用例。小 grid 的用例耗时较长。 | 横轴为提交的 blocks。历史报告记录了 grid 与线程数，复跑使用固定提交的现行源代码和 100 遍预热。历史爬坡二进制与当前源码的同一性未经验证。 |

每行结果保存在 `bm-bandwidth.csv`。`logical_GB_s` 取未插桩进程打印的有效带宽，`dram_read_GB_s`、`dram_write_GB_s`、`llc_fabric_read_GB_s`、`l2_read_GB_s` 取单独 ACU 运行中的对应计数器，均为十进制 GB/s。未采集的字段留空；打开 `manifest.json` 检查各测项状态。每行的 `.log` 保存实际启动参数，ACU 原始 CSV、报告及命令也一并保存。脚本不改设备频率。上游文档中的 2.77 TB/s 是按 ACU 利用率反推的 DRAM 峰值，跨机器优先比较相同条件下的实测值。

## 在内网运行

**宿主机上只用预置的 `nerdctl` 进入分配给你的 PPU 容器。** 如果已经在容器终端，直接跳到下一步。如果列表里找不到分配给你的容器，向平台管理员确认容器名或 nerdctl namespace；容器和设备挂载遵循机房配置。

```bash
nerdctl ps
read -r -p '输入分配给你的容器名或 ID: ' ppu_container
nerdctl exec -it "$ppu_container" bash
```

**以下操作全部在容器内执行。** 将离线源码包解压到分配的持久化工作目录，进入含 `run.py` 的目录。结果 ZIP 也保存在该工作目录，便于取回。先检查设备和工具：

```bash
ppu-smi
command -v python3 nvcc acu
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-未设置}"
```

原生测试使用 Python 3.8+ 标准库和 PPU SDK 的 CUDA 兼容 `nvcc`。同源算子还需要能在该卡执行的 PyTorch 和提供 `tilelang.ppu.language` 的 SAIL TileLang。JIT 接口按 `cd65f13d` 的 `tvm_ffi` 路径设计。ACU 采集需要 SDK 的 `acu`。如果只有 SDK，先运行 `--suite native`；缺少 ACU 时省去 `--acu` 仍可采集普通计时。[PPU 工具链对应关系](https://help.aliyun.com/en/document_detail/2864731.html) · [TileLang PPU fork](https://github.com/t-head/tilelang-for-sail)

在运行 benchmark 的同一容器和 Python 环境中检查 paired 的依赖：

```bash
python3 -u - <<'PY'
import sys
print("python:", sys.executable)
import tilelang
print("tilelang:", getattr(tilelang, "__version__", "unknown"), tilelang.__file__)
import tilelang.ppu.language as T
from tilelang import tvm
print("language:", T.__file__)
print("target:", tvm.target.Target({"kind": "ppu", "arch": "ppu_15"}))
import torch
print("torch:", torch.__version__, "device available:", torch.cuda.is_available())
PY
```

如果缺少 `tilelang.ppu`，先核对解释器、包路径和机房提供的 SAIL 环境。参考版本的通用 `tilelang.language` 默认导出 CUDA 方言，直接替换 import 会改变编译入口。若检查通过但执行失败，以输出目录中对应 case 的 `.stderr` 完整 traceback 定位 JIT、设备加载或运行阶段。

第一轮使用分配给你的单张空闲卡。程序使用进程内的逻辑设备 0；如果容器或作业系统已经限制可见设备，保留原有映射。`--smi-id` 填**容器内 `ppu-smi` 对这张卡显示的编号**。以下示例假定该编号为 0。若容器暴露多张卡且未设置可见设备，先按平台分配在容器内设置 `CUDA_VISIBLE_DEVICES`。采集后用 `device.csv` 的 PCI 地址与 `inventory.log` 核对监控的是否为执行测试的卡。

```bash
python3 run.py board results-card0 --smi-id 0 --acu
python3 run.py pack results-card0 results-card0.zip
```

**`results-card0.zip` 仅供内网备份和分析，不外传。** 即使部分测试失败也可以打包，`manifest.json` 列出各阶段状态和时间，对应的 `.stderr` 和 `.command.json` 保存原因和原始命令。每次使用新目录，保留之前的数据。只想快速检查环境或分开运行时可用：

```bash
python3 run.py board results-native --suite native --max-mib 32 --smi-id 0
python3 run.py board results-pair --suite paired --cases gemm_aiu_on gemm_aiu_off --smi-id 0 --acu
```

32 MiB 适合快速检查；正式带宽扫描默认到 512 MiB/数组，两数组共约 1 GiB，可用 `--max-mib 1024` 或 `2048` 扩展。原生测试默认编译目标为兼容 ISA `sm_89`，TileLang 目标为 `ppu_15`。可用 `--nvcc /path/to/nvcc` 指定 PPU SDK 编译器。为了核对板间差异，可在另一张空闲卡用相同参数复测，输出到新目录。

**频率对照。** `clocks.log` 保存指定卡的 `ppu-smi dmon` 采样，`--clock-note` 只记录实验标签。若机房已给同一张卡设置 1300 MHz，可以同样命令另跑一轮，加上 `--clock-note site-locked-1300`。记录机房的实际设置，并用监控核对运行工况。启动时的标称时钟和每秒一次的板卡采样均无法证明每个短 kernel 的实际频率。ACU 统一使用 `--clock-control none`。普通计时在无 profiler 的进程中完成。

## PPU 1.5 的 TSM 与 KVD 容量对照

在已有 SAIL SDK 和 ACU 的容器内运行，只用一张已分配的卡。此套件自动启用 ACU，独立于默认套件，无需 TileLang 或 BM 源码。先做 8 个 profile 点，再运行完整的 80 点扫描。两次都显式选择相同的缓存控制策略。

```bash
python3 run.py board results-kvd-smoke --suite kvd-capacity --acu-cache all --tsm-kib 0 160 --kvd-kib 64 128
python3 run.py board results-kvd-full --suite kvd-capacity --acu-cache all
python3 run.py pack results-kvd-full results-kvd-full.zip
```

同一可执行文件、同一 kernel，每个点只启动 1 个 block、128 个线程，排除 block 并驻数变化的影响。每轮按 64 B 步长对全局数组显式执行可缓存的 32 位 load，连续扫 100 轮，每轮结束时同步。工作集大小指扫描的**地址跨度**。每个进程分配相同大小的数组，输入内容和访问偏移一致。非零 TSM 申请只碰末尾一个字节。每个点都会与 CPU 计算的逐线程结果对比。

每个 TSM 申请量（默认 0、64、128、160、224 KiB）和工作集（32、64、80、96、112、128、160、256 KiB）都运行两个独立进程。`default` 使用进程默认偏好，`prefer_shared` 调用 `cudaFuncSetCacheConfig(..., cudaFuncCachePreferShared)`。两组都设置相同的函数动态共享内存上限，然后只通过启动参数改变实际 TSM 申请量。ACU 窗口内仅有一个目标 kernel。运行器核对 ACU 记录的动态字节数，API 返回错误、CPU 校验失败或缺失计数都会记为失败。

查看 **`kvd-capacity.csv`**。固定 TSM 和工作集，看 `hit_delta_from_default_pp`（显式偏好减默认的 KVD 读命中率，单位百分点）；固定偏好，沿工作集从小到大比较不同 TSM 申请量下的命中率曲线。拐点移动时，用 `l2_load_transactions_from_l1` 核对下行读事务是否同步增加，并在相同条件下复测。结果记录为 `failed` 的点需查看 `manifest.json` 和对应 stderr。每个成功点附原始 ACU CSV、报告与含输入基址及函数上限的 `*-check.json`。如果曲线重合，只能说明这个 SDK、固定函数上限和测试访问模式下没有观测到变化；默认偏好也可能已经选中相同策略。

## 各测试能校准什么

| 产物 | 工作负载与口径 | 对应 GTSim 参数 |
| --- | --- | --- |
| `device.csv`、`occupancy.csv` | 记录 HBM 字节、CU、warp、TSM、运行时寄存器计数单位。以 1/8/32/64 条独立链构造 kernel，读回编译器实际分配的每线程寄存器、local 字节；扫描线程数和动态 TSM，查询 occupancy API 的理论 block/CU。 | CU 资源和驻留上限。结合 ACU 的 Block Limit Registers/Shared Mem/Warps 确认约束来源，再换算寄存器字节。API 上限与一次执行中实际并驻的 block 数分别记录。 |
| `stream.csv` | 1–512 MiB 默认工作集，分别测读、写、copy。以设备 event 计时，保存五组原始样本。读 kernel 每线程写一个归约结果，写入字节另列；copy 合计读写两侧。 | HBM/LLC 有效带宽、工作集拐点。表内是程序逻辑字节。用指定工作集单 launch 的 ACU 下游字节、命中率确认流量实际经过哪一级。分不出 CE 私有容量时保留观测拐点。 |
| `shared.csv` | 对 32 位共享内存读扫描 lane 间距 0/1/2/4/8/16/32 个 word；0 为同地址广播。用两种循环长度扣除固定初始化和栅栏成本。 | TSM bank 冲突相对代价和读服务率。跨 warp 的并发与循环里的累加也影响绝对值。 |
| `math.csv`、`mma.csv` | 1 warp/1 链测依赖延迟，4 链版本测交错执行递推的效果，每条链内部仍有依赖。FP32 FMA、`exp2(-x)`、`s/(1+x)`、BF16/FP8 warp MMA，32/128/512 步各五组。 | 用于依赖延迟与延迟隐藏；组件吞吐用下面的无递推实验。 |
| `independent-array.csv` | 长 FP32 数组逐元素执行 `out[i]=exp2f(in[i])` 和 `out[i]=exp2f(-in[i])`。不同元素独立；每元素逻辑读 4 B、写 4 B。 | 单看完整 kernel 的结果/s、GB/s 随工作集的变化，再用 ACU 区分 HBM、缓存和计算瓶颈。 |
| `independent-{fma,ex2,div,bf16,fp8}.csv` | 寄存器内产生独立输入，每个结果只进入整数校验和。FP32 目标是 `fma.rn`、`ex2.approx.ftz`、`div.rn`；矩阵 C 操作数来自新输入，下一条不读上一条 D。 | 扫并发度和展开数，测各组件可达到的服务率。用 PASM、执行计数检查指令是否保留、是否被拆分。 |
| `gemm_aiu_on/`、`gemm_aiu_off/` | BF16 GEMM，M/N/K=128/64/96，两个 block，128 线程/block，tile=64×64×32，两级流水，只切换 AIU lowering。 | AIU 与普通搬运的相对表现、计算和数据到达的依赖。可核对源码、动态指令数、缓存字节及耗时差。 |
| `gemm_multiwave/`、`vector/` | 前者用相同 tile 计算 640×512×96，grid 共 80 个 block，可考察 39 CU 上的分配与并驻；后者为 10240 个 FP32 元素计算 `(A+B)*(A-B)`，80 个 block。 | 把小 GEMM 得到的参数用于更多 block 和不同工作负载。80 个 block 的实际波数取决于资源与调度，需查驻留数据。 |

原生 CSV 的时间单位为 µs/launch。`*-summary.json` 保留各组中位数、最小/最大值和逻辑服务率。计算依赖斜率可用 `(t512 − t128) × 实测核心MHz / 384` 周期/递推步，前提是同卡、同操作、同并行度、工况稳定。每步可能包含多条机器指令。内存绝对延迟、独立 AIU 命令启动延迟、各缓存写回策略仍需专门实验；本套件不会自动覆盖设备配置。

## 单独运行吞吐测试

```bash
python3 run.py board results-throughput --suite throughput --smi-id 0 --acu
python3 run.py pack results-throughput results-throughput.zip
```

在容器内用已经分配的空闲卡；此模式只需 PPU SDK 兼容编译器，使用 `--acu` 时还需 profiler。默认数组扫描到 512 MiB/数组，`--max-mib 32` 可缩短首次试跑。保留之前的递推数据用于比较。

寄存器测试扫描全卡 grid 大小（CU 数的 1、8、16 倍，每 block 256 线程），每线程分别执行 128、512 个目标操作，并比较 4、16 路展开。输入由线程号和操作编号产生；计时循环内的目标计算无前次结果依赖，也不从 HBM 流式读取输入。索引生成、整数校验、循环和末尾写出也占用时间。每个形状先取五个样本的中位数，再看增大 grid、展开数或工作量后是否达到稳定平台。`--acu` 采集 8 blocks/CU、16 路、512 操作的参考点；数组采集最大工作集的负指数版本。时钟采样与 `.stderr` 的 CU 数和 PCI 地址一起保存。

| 量 | 如何比较 |
| --- | --- |
| FP32 指令吞吐 | `target_ops_per_second / (CU 数 × 实际核心频率 Hz)` 得到元素/CU/周期。高层 `exp2f` 与近似 EX2 分开解读，`div.rn` 也可能展开为多条机器指令。 |
| MMA 吞吐 | 计数单位为每 warp 一个矩阵片段。BF16 m16n8k16 按 4096 FLOP/片段、FP8 m16n8k32 按 8192 FLOP/片段，与 GTSim 不同形状的指令先用 FLOP/s 对齐。 |
| 实测可达峰值 | 在有效计数和稳定频率下，取已饱和形状的中位吞吐。若仍随并发度上升或有 spill、辅助指令瓶颈，就只记录已达到的下界。理论峰值另由硬件规格给出。 |

## 同源算子与流水图核对

指定 `--acu` 后，原生测试还会对 8/64 条 FMA 链的寄存器压力变体和 BF16/FP8 的四累加链 MMA 各采一个正式 launch。寄存器实验与 `occupancy.csv` 的 256 线程、0 B 动态共享内存行对应；以编译器实际分配值和报告中的限制资源解释驻留数。

每个 TileLang 用例只编译一次，普通计时直接运行该 JIT 对象，同时从其 `artifact.device_mod` 保存 `device.json`，从该对象保存 `kernel.ppu.cc`。运行前逐元素对照独立 CPU 参考答案，然后在已分配的输入输出上测量。每个用例有七组热态时间，以及七组间隔 256 MiB 写扫后的单次时间；写扫在计时区间之外。这是驱逐尝试，是否已冷却 LLC 由 ACU 字节数判断。Python event 计时可能包含设备等待 host 提交的空隙，尤其是很短的 kernel。

`--acu` 为 TileLang 各用例另起进程，在 profiler start/stop 内只发起一次目标调用，并核对 profile 与普通计时的 TIR 和生成源码哈希。同时对 stream 的 32 MiB（较小扫描上限时取其上限）及最大工作集采集读、copy 报告。`.acurep` 可用官方 GUI 查看，`acu-*.csv` 可直接分析。默认 `--acu-cache none` 仍可能刷新 L1/L2；若需跟冷启动模拟对照，可在另一轮使用 `--acu-cache all`。官方工具报告 warp stall、PASM 计数、寄存器及流量；这些统计约束流水图，无法恢复每一条指令的逐周期时间戳。[ACU 时钟与缓存控制](https://help.aliyun.com/zh/document_detail/3012044.html) · [ACU 指标和 GUI](https://help.aliyun.com/zh/document_detail/3030030.html)

将 GTSim 源码和构建环境带入内网，在内网仿真机器分析结果。以下示例假设 `PPU-benchmark` 与工作版 `GTSim-ppu` 是相邻目录；其他布局可传 `--adaptor`，smoke 还可传 `--fixtures`。

```bash
python3 run.py smoke smoke-out --sim ../GTSim-ppu/build/gtsim_ir
python3 run.py replay results-card0 sim-out --sim ../GTSim-ppu/build/gtsim_ir
```

`smoke` 用 GTSim 仓内历史 AIU 开/关样例检查工具链。`replay` 只消费本次上板保存并通过正确性检查的 TIR，核对其 SHA-256；各用例输出模拟报告、block 0 的节点流水 TSV 和全 grid 的驻留 TSV。`comparison.json` 并列给出模拟周期、按配置核心频率换算的微秒数和两种实测 event 中位数。

比较时先对齐 ACU 和 TIR 的 grid、block、kernel 数、指令和搬运字节，再比较时间。把计算吞吐、依赖链和带宽微基准用于定参，预留 `gemm_multiwave` 与 `vector` 检查调整后的结果。整算子时间相近时，也要检查流水是否靠相反方向的误差抵消。
