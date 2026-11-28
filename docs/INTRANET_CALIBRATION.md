# PPU 1.5 补测操作手册：在内网完成采集、分析和参数登记

更新：2026-11-28。对应 GTSim ISSUE 5、6、7。**原始数据、PASM、TIR、报告、设备信息和分析结果全部留在内网。** 本文不要求上传或回传结果。只将本仓源码带入内网；需要在内网比对模拟时，将对应版本的 GTSim 源码也带入。

本次提供的新原生 kernel 尚未在 PPU SDK/实卡验证。外网仅检查 Python、分析公式和用例生成。必须先通过下面的环境、编译、正确性和指令计数检查，再采用数据。某条兼容指令不支持时记录失败，不自动换 dtype、换算子或改用 CUDA TileLang。

## 0. 执行顺序与本次交付范围

按以下顺序执行，不要求内网助手临时发明测试方法：

1. §1 保存环境信息，运行无设备的脚本自检。
2. §2 跑 `smoke`，检查 `manifest.json` 和正确性；失败项按 §3 处理。
3. 跑 Vector、Cube、TSM；先无 ACU 测量，再仅对代表点采 ACU/PASM。
4. 按 §4 的斜率方法分析；按 §5–9 对应检查表决定值是否可以登记。
5. 跑缓存/驻留补测及已有套件；不把它们的未知项混进计算参数。
6. 按 §13 做同源 TileLang 边界/依赖验证，再填写 §14 的参数记录。不得自动覆盖 GTSim 配置。

| 编号 | 目的 | 本仓入口/状态 |
|---|---|---|
| V-01 | FP16/32/64 add、mul、FMA 可达吞吐和 FMA 依赖间隔 | **新增可执行** `supplement.py --suite vector`；按 dtype 隔离编译 |
| V-02 | 200/256 等长度的实际执行粒度；区分逻辑 tile 与 warp 记录 | **新增可执行** 原生 mask 扫描 + `paired_sweep.py vector`；PASM 判据见 §5 |
| V-03 | SFU 原始操作 vs `exp2f` 编译展开；FP16 是否提升到 FP32 | 已有 `throughput` + **新增** TileLang exp2；不能把高层复合操作斜率当纯 SFU 延迟 |
| C-01 | FP16/BF16/FP8 Cube 独立吞吐和累加依赖间隔 | **新增可执行** `--suite cube`，兼容 ISA 的 m16n8 分型须对照原生 PASM |
| C-02 | 各 MNK 边界、tiling、输入精度与累加精度 | **新增可执行** `paired_sweep.py cube`；原生 FP32/FP4 路径仍需内网 SDK 能力确认，不伪造支持 |
| S-01 | TSM 纯读、纯写、混合是否共用带宽 | **新增可执行** `--suite tsm` |
| S-02 | TSM 依赖读取、SHFL、block barrier 有效延迟 | **新增可执行** `--suite latency`，结果含循环/地址开销 |
| M-01 | KVD/L2/LLC/HBM 的依赖读取成本和工作集变化 | **新增可执行** 随机 `global-chain`；命中层级须 ACU 判定 |
| M-02 | 容量/TSM carveout、带宽与并发数 | 已有 `kvd-capacity`、`bm-bw`、`bm-ramp`，补充分析规则 |
| A-01 | AIU 开关、专路/共享端口、命令启动与队列 | 已有 paired 开关 + **新增** 大 shape；启动/队列隔离实验需 SDK 原生命令，详见 §10，不能从普通 GEMM 直接定参 |
| R-01 | 寄存器单位、驻留、spill、block/CU 上限 | 已有 native/ACU，补充单位和上下界判据 |
| W-01 | 写分配/写回、在途/队列、拓扑共享范围 | 补充实验步骤与可识别边界；不能仅从一个吞吐拐点反推物理深度 |

“可执行”表示源码和运行入口已提供，不等于已经上板通过。“需 SDK”表示公开兼容接口无法保证对应原生命令语义，应按规定停止该参数的定值，继续其余已支持测试。

## 1. 环境、频率、隐私和重复性

所有编译、测试、ACU、分析都在已经分配的容器内运行。宿主只使用预置 `nerdctl` 进入容器，不安装 Docker，不改驱动/全局环境，不自行锁频，不使用未分配的其他卡。

```bash
# 已在容器内则不要再执行这两条。
nerdctl ps
nerdctl exec -it 已分配的容器名 bash

# 以下均在容器内、含 run.py 的源码目录执行。
command -v python3 nvcc acu
ppu-smi
python3 -m unittest test_runner test_supplement
```

沿用容器的 `CUDA_VISIBLE_DEVICES`，所有程序使用逻辑设备 0。`--smi-id` 是同一张卡在容器监控工具中的编号，必须用 PCI 地址核对，不假定它等于逻辑 0。保存 SKU、SDK、编译器、TileLang、CU 数、功耗/温度/时钟、源码/二进制哈希。所有结果使用新目录，不能覆盖旧数据。

新运行器保存编译命令、每点 stdout/stderr、源码/二进制哈希、`inventory.stdout`、`device-info.stdout`、`clocks.log` 和每一步状态。不修改时钟。`nominal_core_khz` 是属性值，**不能直接代替测量期间频率**。若机房允许并设置固定 1300 MHz，记录机房设置，核对运行期监控；ACU 一律 `--clock-control none`。未确认核心频率时只报微秒、GB/s、FLOP/s，不填写周期参数。

稳定性：每点至少 7 个普通计时样本，报告中位数与范围。出现温降频、别的作业、local spill、样本范围超过中位数 5% 时复跑；5% 和后面的 3%、R²=0.98 是本次分析筛选阈值，不是硬件常数。短 kernel 用多个循环长度拟合，不能靠大量重复 launch 的平均数消除所有提交开销。

## 2. 可以直接复制的运行命令

### 2.1 新原生补测

```bash
python3 supplement.py plan plan-smoke.json --suite smoke --cu-count 39
python3 supplement.py run cal-smoke --suite smoke --smi-id 0
python3 supplement.py analyze cal-smoke

# smoke 检查通过后依次执行；不要同时开多个套件争用同一卡。
python3 supplement.py run cal-vector --suite vector --smi-id 0
python3 supplement.py run cal-cube --suite cube --smi-id 0
python3 supplement.py run cal-tsm --suite tsm --smi-id 0
python3 supplement.py run cal-latency --suite latency --smi-id 0

# 只有确认本轮确实为 1300 MHz 时才加该选项；不是设置频率。
python3 supplement.py analyze cal-vector --core-mhz 1300 --frequency-note site-fixed-and-verified
python3 supplement.py analyze cal-cube --core-mhz 1300 --frequency-note site-fixed-and-verified
python3 supplement.py analyze cal-tsm --core-mhz 1300 --frequency-note site-fixed-and-verified
python3 supplement.py analyze cal-latency --core-mhz 1300 --frequency-note site-fixed-and-verified
```

若已生成 `analysis.json`，重新分析必须传新 `--out analysis-1300.json`。`--timeout 600` 可用于单点较慢的内网环境；默认每命令 180 秒。超时只影响该点并保存失败原因。`all` 是全部扫描，点数较多，优先分套件执行。程序不自动将失败当成功，也不自动下调工作量隐藏错误。

### 2.2 只 profile 代表点

先在普通测量中选择：最低并发、达到平台的并发、最长循环、纯读/纯写/混合各一点、边界长度的两侧。每个 case 的 ID 和参数在 `plan.json` 中。

```bash
# 查看可选 ID（输出都留内网）。
python3 - <<'PY'
import json
for c in json.load(open('cal-tsm/plan.json')):
    if c['params']['steps'] == 4096:
        print(c['id'], c['params'])
PY

# 将下方 CASE_ID 替换为上一条输出的完整 ID；可传多个。
python3 supplement.py run cal-tsm-acu --suite tsm --smi-id 0 \
  --case CASE_ID --acu --acu-cache none
```

ACU 与普通计时是两个独立运行，使用同一个二进制和同一组参数；程序通过 profiler start/stop 只圈一个目标 launch。报告中核对只有目标 kernel。`--acu` 对所有选中点采全量指标，耗时很大，不建议直接全量 profile。ACU 仍可能 replay，缓存策略按版本确认；`none` 不等于“所有缓存保证热”，`all` 不等于“此模型的空有效字节缓存状态已复现”。

### 2.3 复用已有测试

```bash
python3 run.py board cal-native --suite native --smi-id 0 --acu
python3 run.py board cal-throughput --suite throughput --smi-id 0 --acu
python3 run.py board cal-kvd --suite kvd-capacity --acu-cache all
python3 run.py board cal-bm --suite bm-bw --smi-id 0 --acu
python3 run.py board cal-bm-ramp --suite bm-ramp --bm-blocks 2 16 32 64 128 256 --smi-id 0 --acu
```

BM 套件需要此前离线包里的 BM 源码，普通 Git checkout 不自带；新 supplement 不依赖 BM 或 TileLang。32 MiB 只做冒烟，HBM 稳态沿用 512 MiB、1/2 GiB 与 BM 16 GiB 点，先确认空闲显存（copy 需两份）。不要因为内网环境缺少 TileLang 就阻塞 SDK 原生测试。

## 3. 失败项和证据检查，不依赖助手猜测

1. `manifest.json` 中 `failed/blocked` 的点不进入参数表；打开同名 `.stderr` 和 `.command.json`。
2. 某 dtype 的编译失败只影响该二进制，其他 dtype/内存测试继续。FP64 编译失败不等于硬件没有 FP64，也不能用 FP32 的时间代替。
3. CPU reference 失败：停止采用该点，先排查编译器支持、边界、类型/布局；不能放宽到掩盖错误。新原生程序验证所有已写线程结果；paired 保存容差和最大误差。正确性通过不证明每条目标指令都保留。
4. 打开 ACU GUI 的目标 kernel PASM 页面。也可用 SDK 已有的反汇编命令；先看该 SDK 的 `--help`，本仓不假定 NVIDIA `cuobjdump/nvdisasm` 能解析 PPU 二进制。
5. 每个代表点登记：循环内实际 opcode、计算 dtype、每循环动态条数、作用粒度（thread/warp/warpgroup）、是否有分支、转换、地址计算、校验、load/store、spill。找不到动态计数时保留静态循环体和迭代次数，标为条件推断，不能虚填。
6. 判断副作用：编译器合并了独立操作、删除循环、把 FP16 提升为 FP32、FP32 Cube 改 TF32、把 EXP 拆成多条时，重新标记其语义，不把源语言操作数当成原生指令数。

**复合函数规则**：`exp2f(-x)` 的范围检查、选择、乘法与 SFU 指令作为整体可以测“高层 EXP 有效成本”；仅在核对到原生 `v.exp2` 并隔离其他资源时才能填 SFU 的原始服务率。`s/(1+x)` 同理。取负可能折叠为操作数修饰符，不能凭源代码额外加一条 VALU，也不能凭高层 API 就删掉实际存在的修正指令。

## 4. 统一分析公式与自动结果含义

新套件同一参数组合固定 grid、threads、dtype、slots、工作集，仅将 steps 设为 256、1024、4096。每点取中位时间 `t(S)`，拟合：

```text
t_us(S) = a_us + b_us_per_step × S
cycles_per_step = b_us_per_step × core_MHz
每 CU 每周期服务量 = work_per_step / (cycles_per_step × 归一化CU数)
GB/s = 字节数 / (微秒 × 1000)
TFLOP/s = FLOP数 / (微秒 × 1e6)
```

`a` 包含启动、固定初始化/输出和流水首尾，不直接等于某条指令 latency。看相邻两段斜率是否相近（建议差异 ≤10%），并要求正斜率和 R²≥0.98；脚本检查整体拟合，人工还需看两段。固定开销大的近乎水平数据不能给出可靠的 L/II（此前 FMA 的“0.5 周期/依赖步”不能直接采用）。

`analysis.json` 保留原点、拟合、local 存储标记、工作量和归一化结果，**始终标为候选，不自动认证峰值，不改配置**：

- 单 block 的执行归一化为一个 CU；可能没有填满该 CU，只是可达下界。
- 多 block 用设备可见 CU 数归一化，是全卡平均；需 ACU 确认活跃 CU、分布和驻留，不能凭 grid=CU 数认定正好每 CU 一个 block。
- 寄存器内独立吞吐测试的输入不依赖前一结果，但实际 PASM 的输出寄存器复用、辅助指令、编译器排布仍可能限制吞吐。扫描 slots/grid，检查 spill，不能仅见“独立”就认定达到峰值。
- 依赖链 slope 是实际循环的依赖间隔。只有循环内一条目标指令、辅助开销可忽略/已隔离时才可视作该路径 L；矩阵累加转发间隔不一定等于任意消费者可读结果的延迟。
- 日志中的 `local_bytes` 不一定全是 spill；它是复核信号，需编译器报告和 PASM local 流量确认。

**饱和判据**：在无 spill、工况一致时，提高 grid 和 slots 后连续两个较高并发点的服务率变化 ≤3%，增加 steps 的斜率稳定，且目标 FU 的利用率/动态计数支持瓶颈判断，才称“该测试已达到稳定可达吞吐”。它仍不是数学证明的硬件理论峰值；未饱和则登记下界，禁止降低峰值配置到该数。

## 5. Vector：dtype、吞吐、粒度和依赖

### V-01 操作 × dtype 的服务率

`cal-vector` 覆盖 FP16/FP32/FP64 add、mul、FMA；grid=1、CU、8×CU、16×CU，threads=256，slots=1/4/16，steps 三点。FP16 使用打包 f16x2，两个半精度元素都计数；FP32/64 每 lane 一个元素。每循环目标操作数：

```text
有效元素次数 = blocks × active_elements_per_block × slots
FMA有效FLOP = 有效元素次数 × 2
add/mul有效FLOP = 有效元素次数
```

对完整激活的 block，FP16 默认 512 元素，FP32/64 默认 256。脚本的 `source_elements` 只补齐 f16x2 的两个元素，不擅自按 CU 的 g=256 再补齐。先用对齐点校准 R，再做 V-02 推断 g。

参考待核值：FP16 256、FP32 128、FP64 32 元素/CU/周期。不能仅凭 FMA 规格强制 add/mul 相同，应分别登记测量。`vector-chain` 只用一个 warp、一个累加链做有界 FMA 递推，不能用它测峰值；多链吞吐不能反推出单链 L。FP16 的两个打包半精度是两条并行元素链，不将 warp 元素数乘进单链斜率。

### V-02 执行粒度：先区分哪一级的粒度

原生扫描固定 256 threads/block，FP16 f16x2，active=63/64/65、127/128/129、199/200、255/256/257、300、511/512。它检查 warp 激活与打包指令的边界，**并非直接证明一个 TileLang 行必须按 256 取整**。后者用以下同源测试：

```bash
for n in 127 128 129 199 200 255 256; do
  python3 paired_sweep.py vector paired-v256-$n --dtype float16 --op add --rows 312 --tile 256 --valid "$n"
done
for n in 255 256 257 300 511 512; do
  python3 paired_sweep.py vector paired-v512-$n --dtype float16 --op add --rows 312 --tile 512 --valid "$n"
done
```

每行一个 block、128 threads，`tile` 是编译时循环范围，`valid` 是有效元素数，两者分别记录。先保持 tile 不变观察谓词和指令数，再比较 tile=256 与512。313/624等 rows 可用于改变并发验证，不能在长度对比中同时改变 grid。

分析步骤：

1. 读取 `input.tir/device.tir/kernel.ppu.cc`，记录每个逻辑行 lower 成几个 warp/vector 操作及其 mask。
2. 从 PASM/动态执行计数核对是部分 lane 激活、打包操作、每 warp 一条，还是整行固定执行数。相同耗时但 HBM 受限的两个形状不能证明相同计算量。
3. 候选 g 分别计算 `ceil(E/g)×g`，对照**执行指令数和资源工作量**的跳变，不仅拟合一个总时间。多个 g 都能解释数据则不定值。
4. 当前模型约定 FP16 行粒度 256；200/256 都按256、300按512。如果实测只支持其他层级粒度，记录差异和该 lowering，不将256当作已测事实。
5. 不允许把同一逻辑行拆成的每个 warp 记录都补到256，造成重复收费；也不允许把真实独立命令任意合并。padding 的无效 lane 不自动增加全局搬运字节。

### V-03 SFU 与复合操作

已有 `cal-throughput` 提供原始兼容 `ex2.approx` 和长数组 `exp2f(x)/exp2f(-x)`。新增：

```bash
python3 paired_sweep.py vector paired-exp16 --dtype float16 --op exp2 --rows 312 --valid 256 --tile 256
python3 paired_sweep.py vector paired-exp32 --dtype float32 --op exp2 --rows 312 --valid 256 --tile 256
python3 paired_sweep.py vector paired-fork --dtype float32 --op forkjoin --rows 312 --valid 256 --tile 256
```

FP16 调用若提升到 FP32 并增加转换，登记“FP16输入的复合函数成本”，不直接认证 FP16 SFU。`forkjoin` 是 `(A+B)*exp2(A)`：add 与 exp2 无数学依赖，mul 等二者；比较 TIR/DAG 是否保留这点，ACU 聚合指标不能恢复逐周期的两条指令重叠。

SFU 参考吞吐为16元素/CU/周期（FP16/FP32）；需原生指令确认。纯 EXP 结果反复喂自身会溢出，不能直接用无界 `x=exp2(x)` 长链测 L。优先使用 SDK 文档给出的有界原生链（例如已确认可折叠取负的 EXP(-x)）；若实际多出 mul/csel，则记录整条链成本，不将57周期直接填成 SFU L。本次不伪造一个可从复合链唯一分离的纯 SFU 延迟。

## 6. Cube：分型、II、结果延迟与 dtype

### C-01 原生基础测试与兼容指令映射

`cal-cube` 新增 FP16，并保留 BF16/FP8。独立模式的 C 来自常数，前一个 D 不作为下一条 C；链模式则明确 D→C。普通测试验证所有线程输出。

**当前兼容 PTX 形状为 m16n8k16（FP16/BF16）或 m16n8k32（FP8），不能直接称为 PPU 的 m16n16 分型。** 一条兼容指令如何映射原生 TC02，必须查看 PASM。不能只因 FLOP 总量相同就假定是一条/半条原生分型。

```text
每循环兼容指令数 = blocks × (threads/32) × slots
FP16/BF16 FLOP/兼容指令 = 2×16×8×16 = 4096
FP8 FLOP/兼容指令 = 2×16×8×32 = 8192
```

先报告兼容指令吞吐与 FLOP/s。若 PASM 证实每循环原生分型数为 H，且已饱和、每 CU 只有一条聚合 Cube 资源，则 `II = cycles_per_step×活跃CU数/H`。H 是全 launch 的每循环计数，不要再额外乘 warp 或 lane 数；若 profiler 返回按 thread 的计数，先换成相同口径再使用。

链模式斜率反映累加相关的最短可观察间隔，可能包括转发。**它不自动等于 Cube→VALU 消费结果的 L**。后一项需在 SDK 原生测试中用最后 D 作为后续 VALU 的输入，再让后续工作无法越过，改变链长度与独立工作隐藏程度；如果没有可靠的设备计时/依赖隔离，结果延迟保留可配置假设，不靠吞吐倒推16周期。

### C-02 MNK 和 tile 边界

模型初值采用用户提供 `(m,n,k)`：FP32=(16,16,8)，FP16=(16,16,16)，FP8=(16,16,32)，FP4=(16,16,64)。假设 II=1 时每 CU 每周期 FLOP 分别为4096/8192/16384/32768。FP16×39×1.3GHz对应415.3344TFLOP/s。输入类型与 FP32 accumulator 必须分开记录。

```text
Q_tile = ceil(M/m) × ceil(N/n) × ceil(K/k)
吞吐需求 = Q_tile × II
有效FLOP = 2MNK；执行FLOP = Q_tile × 2mnk
独立连续投入时最后结果 = L + (Q_tile−1)×II
```

可执行的同源边界测试（先 FP16，BF16 作为既有路径对照）：

```bash
# 固定 block tile=64×64×32，单独改变一个有效维度。
for x in 15 16 17 31 32 33 63 64 65; do
  python3 paired_sweep.py cube cube-m-$x --dtype float16 --m "$x" --n 64 --k 64
  python3 paired_sweep.py cube cube-n-$x --dtype float16 --m 64 --n "$x" --k 64
  python3 paired_sweep.py cube cube-k-$x --dtype float16 --m 64 --n 64 --k "$x"
done
# 固定完整算子形状，改变编译期 tile，比较真实 lowering。
python3 paired_sweep.py cube cube-tile64 --dtype float16 --m 129 --n 129 --k 65 --bm 64 --bn 64 --bk 32
python3 paired_sweep.py cube cube-tile32 --dtype float16 --m 129 --n 129 --k 65 --bm 32 --bn 64 --bk 32
```

注意：编译器若无论 M=15/16/17 都执行完整 BM=64 的计算，三者原生分型数相同是合理的。**模型应对真实执行 tile 计数，不是对逻辑矩阵的边缘有效尺寸盲目套 Q。** 再检查它是否消除无效 MMA 或只 mask 访存。跨64会改变grid，需分清 block 数变化与分型边界；小 GEMM 的总时间主要用于正确性/计数，不用于峰值定参。还需在较大 M/N（如512/1024）上做吞吐验证。

FP32 Cube：可用 `paired_sweep.py cube ... --dtype float32` 探测编译能力，但 PASM 必须确认真实输入精度和 k=8；若是 TF32、SIMT 或报错，登记 unsupported/不同路径，不能认为已校准 FP32 Cube。

FP4 和原生 FP32/FP8 分型边界：本仓未编造 SDK intrinsic，也不以 INT4/FP16 代替 FP4。内网执行者先在现有 SDK 示例/头文件查 `tc02`、`mma`、`f4/e2m1`、`f32/tf32`，保存真实接口及 ISA 支持条件；按已提供的 `cube_probe` 框架，仅替换一条原生计算及输入打包、CPU参考，独立/依赖链、steps/grid/slots 和计数方法保持一致。找不到官方接口时该行保留待确认，不能由0.5字节类型名推断真实算力。FP4 打包搬运字节与计算元素数分别统计。

## 7. TSM：读写预算、访问延迟、SHFL 与 barrier

### S-01 共享还是独立读写端口

新 `tsm-mix` 固定256线程/8个warp；reader warp 数0/2/4/6/8，其他warp只写。各warp访问独立地址，读写不互相依赖；初始化和末尾检查在循环之外。Width=1/4/16 为每线程每步访问的32位word数，**不是保证一条v4/v16机器指令**，以PASM为准。steps三点扣除固定初始化、栅栏、检查和输出。

```text
read_bytes_per_step = blocks × reader_warps ×32×width×4
write_bytes_per_step = blocks × (8−reader_warps) ×32×width×4
```

先选每个方向已饱和的配置求 Br、Bw，再预测混合点：

```text
独立端口：T ≈ max(R/Br, W/Bw)
共享总预算并可有方向上限：T ≈ max(R/Br, W/Bw, (R+W)/Btotal)
```

若 Br≈Bw≈128B/周期/CU，均衡混合总吞吐仍约128，支持共享；若混合可接近256，支持独立或更高总预算。读写单向宽度不同则用上面的完整公式，不能只比较50/50点。需要至少三种混合比例、两种并发/width下结果一致，并排除 LSU发射、整数累加、barrier、bank冲突等瓶颈。普通读写的证据不能自动证明 AIU 写口也共用同一资源。

TSM参考方案为共享128B/周期/CU、约30周期访问开销，均保留可配置。实测支持其他模型时登记实测，不强行使图符合初始假设。

### S-02 延迟与跨warp归约

- `tsm-chain`：1block/1warp、4KiB共享数组，下一条load地址来自上一条结果，32lane不同起点，循环中无其他warp隐藏依赖。`b×MHz` 是依赖访问的有效周期，包含循环/地址开销；检查是否为TSM向量load、是否有bank冲突。不是带宽测试。
- `shuffle-chain`：1warp，连续 XOR shuffle，下一条依赖前一结果。结果校验lane置换；斜率给有效交换间隔。若SDK引入其他指令，按实际序列记录。
- `barrier`：1block，32/128/256/512线程都按相同次数到达barrier。斜率是该线程数、无额外工作时的同步循环成本；不能解释成任意不均衡到达情况下的固定等待时间。barrier必须全部线程参加。
- 归约：保留已有 SHFL→VALU 串行层级。跨warp用同一block内 TSM store→barrier→load→VALU；不用跨CU/NoC通路。单独测出的load、shuffle、barrier开销不可在同一流水时间段重复相加。

**store 延迟不能直接用 store event 的斜率测出来**：它可能只表示写入被接收。需要 SDK 定义的可观察消费事件，例如同block另一warp在合法同步后读取并产生后续依赖。这个实验测到的是store→同步→load路径，不能无依据减去两个独立平均值就宣布为纯store L。当前对称读写30周期仍是明确假设。

### S-03 请求发射能力

保留前级发射限制是合理的；字节服务与命令发射分别定参。要测命令II，固定总字节量，改变实际机器命令数/每命令字节数；保持地址、并发、缓存状态一致。以PASM确认1×128B与8×16B是否真对应不同命令序列，不能按DAG节点数代替。

当前 Width 扫描能发现趋势，但若编译器全部展开为相同32位load/store，它只改变指令数量，并未改变每命令大小，不能唯一分离字节带宽和发射率。此时沿用可配置发射参数，记录无法识别；不要删除前级以强迫总时间相等。

## 8. 缓存：容量与依赖读取 latency

### M-01 新的随机依赖访问

`global-chain` 的工作集为16KiB至128MiB，64B地址槽内只读取4B索引，CPU生成随机单环，32lane以不同起点跟随索引，1block/1warp。每点先完整遍历环预热，再测多个steps。记录的是**地址跨度**，不是有效load字节数；有用数据、缓存实际事务字节、地址跨度不能混用。

1. 按工作集画/列 `cycles_per_step`，以ACU的KVD、L2、LLC命中和DRAM字节判断平台对应哪层，不按预设64KiB/1.5MiB/32MiB给曲线自动贴标签。
2. 核对PASM为预期向量全局load而非KSD/标量路径。`ld.global.ca` 的兼容名字不保证PPU每层分配策略。
3. ACU圈定整kernel，不圈定循环的某一小段；本probe warmup在profile范围外，但ACU可能清缓存/replay。profile与普通计时热状态不一致时不能拿同一数字相除，先确认命中率。计时每次从相同起点开始，短prefix反复命中可能偏热；最长steps、不同工作集与ACU必须交叉核对，必要时增加steps（不超过CLI限制）。
4. 32lane是同warp的32条地址链，可能产生事务并行和合并；这测得的是该访问形态的有效依赖成本，不是单个物理DRAM原始时序。
5. 如果平台随steps变化、仍有混合命中或PASM不符，只登记观测，不给某层纯latency。

**层级延迟如何填模型**：观察到的 L_KVD、L_L2、L_LLC、L_HBM 是从线程发出到值可用的端到端路径延迟。若模拟器沿各层累加，不能把四个端到端值分别写进四层再相加。先确认模型字段表示总路径还是增量；只有在一致访问形态、测量误差允许且模型路径可分解时，才拟合相邻层的非负增量，并用所有冷热路径同时验证。负差值或结果不稳定时保留待校准。

### M-02 容量、作用范围和带宽

已有 `kvd-capacity` 对TSM申请量/缓存偏好进行固定1block对照，是容量联动的主要测试；两者容量都可变不证明物理共用SRAM。保持函数动态共享上限、block/threads和地址模式一致，观察命中拐点是否可重复移动，并用下一级事务增加佐证。

L2容量需要确认活跃CE及共享范围。没有CU→CE放置证据时，只能写“此调度/地址模式的有效容量”，不能把单block平台乘4当作每CE物理容量。`avg×10=sum`只证明该指标有10个聚合实例，需硬件/工具文档关联实例与CE。

BM bandwidth使用同源码、同grid/threads、同工作集/遍历次数。原生scalar归约读约780GB/s不能替代HBM峰值。核对计数器侧字节再换算峰值：

```text
counter_peak_Bps = counter_measured_Bps / (pct_of_peak_sustained_elapsed / 100)
若只有事务数 n：measured_Bps = n × bytes_per_transaction / 同一ACU窗口秒数
每实例B/cycle = 全部实例Bps / (实例数 × 该服务域频率Hz)
```

**禁止把“事务数/利用率”当作“事务/秒”**，还缺采集窗口时间；此前小GEMM TSM反推的31GB/s结论不能采用。若计数器为 `.avg`，不能又当 `.sum` 再除实例数。百分比4.77%应输入0.0477。利用率低时四舍五入误差会放大，但数值异常也可能先是单位/窗口错误，不能一律归咎固件。

read/store/copy 的计数器若共享峰值分母，百分比相加可吻合total，这只说明工具的归一化口径一致，不足以独立证明物理读写必共享；还需混合流量饱和实验。LLC约2.66TB/s、L2面向KVD读约1024B/cycle/CE和写256、L2向fabric读256属于不同接口，不能互相覆盖成一个local_cache速率。

## 9. 驻留与寄存器

复用 `occupancy.csv` 和相应ACU测项，登记 threads/block、实际寄存器/thread、静态/动态TSM、local bytes、API理论block/CU、ACU限制因素和活跃block。核对131072 runtime units是否为32位寄存器：将regs/thread×threads×驻留blocks与阶跃对照，不能把units直接当B。

确认block/CU硬上限时，选择很小的寄存器和TSM使用，扫描32/64/128/256线程，grid足够大，并检查warp槽、寄存器、TSM都不是先到的上限。观察到28个block只给出硬上限至少28，不能直接写max_blocks=28。若API值与计数器不一致，分别登记，不臆造实际并驻。

编译选项不变时，提交更多block通常体现为排队，不意味着编译器运行时重新分配更少寄存器。改变launch bounds/maxreg等编译选项属于另一个二进制，需另存hash并检查spill。超容量“最多驻留1个”是GTSim既定近似，不等于真机无spill或允许超额物理分配。无须用不确定的内存带宽反推所有寄存器spill成本。

## 10. AIU：路径、命令与消费事件

先运行已有paired及新增大shape：

```bash
python3 run.py board cal-pair --suite paired --cases gemm_aiu_on gemm_aiu_off --smi-id 0 --acu
python3 paired_sweep.py cube aiu-large-on --dtype bfloat16 --m 1024 --n 1024 --k 1024 --aiu on
python3 paired_sweep.py cube aiu-large-off --dtype bfloat16 --m 1024 --n 1024 --k 1024 --aiu off
```

先比较源码/PASM：如果开关没有改变生成的搬运路径，标为“开关未生效”，**不能拿时间相同证明AIU无收益**。核对ldgsts类、普通KVD load、LSU shared store、TSM AIU写口及L2/LLC/DRAM流量。AIU跳过寄存器中转不等于绕过全部cache；缓存命中也不需要每笔实际访问HBM。

要分离三个参数，内网使用SDK已存在的AIU示例分别做：

| 参数 | 必须的实验结构 | 可接受的解释 |
|---|---|---|
| 命令启动/结果延迟 | 1block，发起一次拷贝→官方wait→依赖读取TSM；tile从256B、1/4/16/64KiB扫描；同来源缓存层级 | `T(D)=a+D/B`的a是整条可观察路径截距，不自动等于纯命令setup；源访问与wait开销要按模型定义归属 |
| 可持续搬运率 | 多块独立目的缓冲，源已知热/冷，多命令在途，消费者保证数据被使用；总量足够大 | 按受限源端口、AIU自身、TSM目的端共同解释，不将HBM峰值当AIU独占速率 |
| 与LSU的共享关系 | AIU写不与LSU访问同一地址；LSU独立读/写TSM与AIU分别单跑、同时跑 | 使用§7混合资源公式判断目的端争用；同下游L2限制也会造成争用，须保持源层级并检查计数器 |

**这三项不能用当前兼容 `cudaMemcpyAsync` 替代**：它可能是主机/设备copy engine，语义不是block内global→TSM。也不能猜测 `cp.async` 在PPU必然映射目标AIU。缺少原生接口/合法wait时只做已有paired路径核对，命令setup、II、纯TSM写带宽保持待确认。

## 11. 写回、队列与缓存策略（保留未知参数，不强行定值）

W-01 写策略分四组：新分配未读目标的冷store、同址重复store、先load再store、写工作集从小于LLC到明显超过LLC。每组用独立进程和相同store指令/范围；输出必须有消费者或CPU正确性检查，记录ACU是否包含回收/读取验证。普通计时与profile分开。

- 看冷store是否有下游读流量，区分实际读改写、cacheline回填、输入/校验流量；不能把全部DRAM读算作C写分配。
- kernel结束时LLC已有脏数据而HBM写字节较少，支持写回/延迟可见的可能；下一次已知会驱逐的流量或显式可观测的完成事件补足字节才可对账。不能把kernel结束等同于所有数据已经写到HBM。
- 改变数据内容与地址复用、保持总字节不变，检查写合并/压缩等影响。当前有效字节模型不要求拟合cacheline事务放大，离散请求合并仍留后续阶段。

队列/在途：固定单请求大小、层级和CU数，增加每线程独立地址链/warp数，在同一kernel内控制未等待请求数，记录吞吐/等待/stall。只能在排除带宽、寄存器、warp、发射瓶颈后把拐点作为在途能力的条件下界。`B×L`可给带宽延迟积的量级，不是物理MSHR/回写队列的直接测量。回写缓冲若无可观测ack或独立stall计数，本阶段保留配置并做敏感性扫描，不能以LLC容量代替。

CU/CE共享范围和NoC：只有可确认放置/计数实例时，才比较同CU、同CE跨CU、跨CE。缺少亲和性控制时记录调度条件和实例分布，不按block编号推断物理位置。NoC不作为本轮计算模型校准的阻塞项。

## 12. 不混淆原始latency、吞吐与发射能力

填写参数时逐条问：

1. 测试是独立工作还是依赖链？前者可找R/II，后者主要看可观察依赖间隔。
2. 一次循环实际做多少条原生命令，属于哪个FU/端口？源代码一行不一定是一条指令。
3. 是否存在多个线程、warp或CU隐藏了依赖？是不是把全卡数除错了范围？
4. 时间包含哪些阶段？把同一个30周期测量既填LD/ST tail又填TSM latency会重复计算。
5. 是否只从一个总kernel时间拟合了多个未知参数？若多种参数组合都解释得通，登记不可唯一识别，增加独立实验或保留假设。

对新配置的要求是按操作×dtype给出执行粒度g或m/n/k、R或II、L及来源，公共FU/端口共享归属保持一致。不是每种dtype复制一套FU，也不是将所有节点结果延迟顺序相加。

## 13. 同源对照和实际流水的可验证范围

新 paired_sweep 保存相同编译对象的 `input.tir/device.tir/device.json/kernel.ppu.cc/hardware.json`。在内网从同一device.json导入GTSim，不重新手写相似算子。保存adaptor和simulator提交、配置hash、DAG、节点timeline、驻留表。

```bash
# 在内网同时有 GTSim-ppu 源码和已构建模拟器时；路径按实际目录替换。
python3 ../GTSim-ppu/scripts/gtsim_tilelang.py paired-v256-200/device.json \
  --sim ../GTSim-ppu/build/gtsim_ir --timeline sim-v200.tsv --timeline-blocks 0 \
  --residency sim-v200-residency.tsv --report sim-v200.json
```

新paired_sweep的hardware.json是独立格式，**不要交给旧版 `run.py replay`**；直接导入并按本节对比。若当前adaptor仍不支持FP16 EXP/FP64，记录明确失败，不能修改成FP32后声称原case通过。

ACU代表点可手工圈定（与普通case相同参数，只更换输出目录并加profile）：

```bash
acu --clock-control none --cache-control none --profile-from-start off --set=full \
  --target-processes all -o paired-v200-acu \
  python3 paired_sweep.py vector paired-v200-profile --dtype float16 --op add --rows 312 --tile 256 --valid 200 --profile-one
acu -i paired-v200-acu.acurep --page raw --csv --csv-file paired-v200-acu.csv
```

先核对两个`hardware.json`的device.json和kernel.ppu.cc哈希一致；否则不能把profile计数拼到普通时间上。按顺序比较：kernel数→grid/block→寄存器/TSM→有效/执行计算量→逐层搬运字节/冷热状态→依赖→总时间。缓存热态计时不能直接校准当前冷启动模拟；需要同一缓存状态或者只比较不受该差异影响的计数。

普通ACU聚合统计不能证明每个节点的逐周期流水图精确；可用它否定不可能的流量、发射次数、驻留和重叠上限。保留未用于定参的验证集：FP16长度200/300、另一种tiling、forkjoin、softmax/归约、多波GEMM。不能用同一个GEMM调完所有参数再以该GEMM误差小作为独立验证。

## 14. 内网参数登记与决策表

复制本目录 `parameter-record.example.json`，所有记录留内网。每行至少包含：参数名、数值/区间、单位、作用范围、操作/dtype/分型或粒度、II与L口径、原始case ID、源码/二进制hash、核心频率依据、PASM核对、缓存层级/并发条件、拟合误差、状态及不能解释的部分。

| 状态 | 使用规则 |
|---|---|
| measured_effective | 特定路径/形状的有效成本，不能直接称为硬件原始规格 |
| inferred_sustained_peak | 多并发平台、动态计数和频率支持的可持续峰值推断，保留条件 |
| lower_bound | 未饱和但有效；不能用它把已有理论峰值下调 |
| assumed | 无可识别数据，保持可配置与来源说明 |
| unsupported | 当前编译入口未支持，不等于硬件一定不支持 |
| inconclusive | 数据有效但无法唯一识别参数，继续补测或保留假设 |

最终在内网生成一张参数表，分列“原模型值、观测值/区间、建议值、状态、证据、修改理由”。禁止自动写配置。结果无须出网；若需要外部协助，仍应遵循数据不能外传的约束，不要求提供原始结果、源码、截图或汇总数值。
