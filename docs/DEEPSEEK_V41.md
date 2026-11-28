# DeepSeek v4.1：同源算子的实卡测试与 GTSim 对照

本套件用于**单个算子从输入已就绪到输出完成的耗时**比较。覆盖固定版本
`inference/kernel.py` 中全部 **6 类 TileLang kernel**，包含 FP8 量化、FP4
量化、FP8 GEMM、FP8×FP4 GEMM、稀疏 attention、HC/Sinkhorn。
不需要下载模型权重，不做八卡整模型推理，也不把各行耗时直接加成模型 token 延迟。

## 1. 来源和覆盖边界，先明确

| 内容 | 来源 / 判断 |
| --- | --- |
| 算子实现 | **DeepSeek 官方** [DeepSeek-V4.1-Flash-1017/inference/kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/kernel.py)，不是阿里编写的 PPU 专用实现。 |
| 固定版本 | `1bae0e1088993b9b82de08f59de5be5a34e42170`。源码、config 和 MIT LICENSE 原样随仓库提供；每次运行检查 SHA-256。 |
| PPU 编译器 | 阿里平头哥的 [tilelang-for-sail](https://github.com/t-head/tilelang-for-sail)。历史仿真使用 `cd65f13dbffffef1d72fed6620c74886cb1ab877`；实卡使用机房提供的兼容版本，记录真实版本和文件路径。 |
| shape | **不是阿里官方 benchmark shape 表**。小尺寸由本项目选择；5120、2304、1280、64 heads、512 head_dim、128 window、512 index_topk、4 HC streams、20 Sinkhorn iterations 来自上述模型 config。M、batch、query 数、KV 长度由测试设定。 |
| 阿里 PPU 示例 | 当前公开仓有 GEMM、flash attention、FP4/FP8 等例子，不能将这些不同实现自动视为 DeepSeek v4.1 的等价性能基准。 |
| 未覆盖 | `model.py` 中 PyTorch 实现的 RMSNorm、RoPE、路由/top-k、SiLU、索引生成、部分 einsum/线性层、embedding，以及视觉、Engram、DSpark 的其余逻辑和跨卡通信。它们没有在这 6 类 TileLang kernel 中，不能宣称已测全模型所有节点。 |

测试不改上游数学公式、tile、流水级数、warp 划分或 pass 配置；仅在生成
PrimFunc 时将 `T.symbolic` 的 shape 绑定为记录的常量，并显式指定 `ppu_15`。
沿用原文件的 `tilelang.language`，这是已有 DeepSeek 仿真入口，不能与自写
`paired.py` 所需的 PPU 方言导入混为一谈。`--aiu on/off` 是单独标记的编译选项实验，
默认 `default` 保留上游配置。FP4 GEMM 仍然是**FP4 解包/转 FP8，再做 FP8 MMA**；
不能为了跑通换成直接 FP4 cube GEMM。

## 2. 具体测哪些 shape

`python3 deepseek_v41.py plan --suite all` 不需要 TileLang/卡，输出所有精确尺寸。

**smoke：12 点**（除 packed FP4 量化外，覆盖之前 11 个 DeepSeek 流水图场景）：

| 测试名 | 输入/输出逻辑 shape 与参数 |
| --- | --- |
| `act_fp8`、`act_fp8_qdq` | X=[32,128]，group=32，E8M0 scale；分别输出 FP8、量化再反量化到 BF16。 |
| `act_fp4_qdq`、`act_fp4_packed` | X=[32,128]，group=32，E8M0 scale；分别输出 BF16、打包 FP4。 |
| `act_fp4_latent_qdq` | X=[32,128]，group=16，E4M3 scale，BF16 输出。 |
| `gemm_fp8`、`gemm_fp4` | M/N/K=32/128/128，group=32，C 为 BF16；B 的逻辑 shape 为 [N,K]，计算 A×Bᵀ。FP4 B 的物理存储为 [N,K/2] 字节。 |
| `sparse_window` | Q=[1,1,64,512]，KV=[1,1024,512]，topk=128；有重复索引及 -1 padding。 |
| `sparse_compressed` | 同上，topk=128+512=640。 |
| `mhc_sinkhorn` | 1 token，mixes=[1,24]，hc=4，20 次迭代，eps=1e-6。 |
| `act_fp8_hidden` | X=[32,5120]，group=32。 |
| `gemm_fp4_expert` | M/N/K=32/2304/5120，group=32。 |

**model：20 点**：M=1/32/256 分别测试 query GEMM (N=1280,K=5120)、
expert up (N=2304,K=5120)、expert down (N=5120,K=2304)、hidden 量化、Sinkhorn；
attention 用 (batch,queries)=(1,16)/(8,1)，KV 长度8192，topk=128/640；
另有全 -1 attention 正确性测试。head 数按 TP=1 的64取值，**不是 TP=8 的单卡
head shard**。M=1 是边界覆盖，M=32 用于已有流水对齐，M=256 看更大计算量和并驻。
这些点是代表场景，不能声称覆盖所有可能 shape、所有专家路由分布或真实请求轨迹。

## 3. 内网怎么跑

遵守机房要求：宿主机只用现成 `nerdctl` 进入已分配的容器，不安装 Docker，
不改全局依赖或频率。下面均在容器内、分配的工作目录和既有 SAIL Python（3.10+）环境执行。
结果、TIR、二进制、PASM 和报告全部留在内网。

```bash
git pull --ff-only origin main
python3 deepseek_v41.py plan --suite all > deepseek-plan.json
python3 -m unittest test_deepseek_v41 -v
python3 - <<'PY'
import sys, torch, tilelang
from tilelang import tvm
print(sys.executable, torch.__version__, tilelang.__file__)
print(tvm.target.Target({'kind':'ppu', 'arch':'ppu_15'}))
print('cuda available:', torch.cuda.is_available())
for name in ('float8_e4m3fn', 'float8_e8m0fnu'):
    print(name, getattr(torch, name))
PY
```

先选一张空闲卡，不要八张卡一起跑同一设备。保留平台已有 `CUDA_VISIBLE_DEVICES`
映射；程序使用逻辑设备0。若平台让你自行选择，先设置分配卡的可见性。
`ppu-smi` 编号与可见性映射不一定相同，先核对 PCI/UUID。

先运行一个小 GEMM，确认完整实卡编译和校验，再全量运行：

```bash
python3 deepseek_v41.py run results-dsv41-first --cases gemm_fp8
python3 deepseek_v41.py run results-dsv41-smoke --suite smoke --acu
python3 deepseek_v41.py run results-dsv41-model --suite model --acu
```

每次目录必须是新目录。每个 case 是独立进程；错误/超时后继续测其他点，最终返回
非零状态，不能把 failed 行当成功。默认每点30分钟超时，可用 `--timeout 3600`。
ACU full 很慢时，先去掉 `--acu` 收齐计时，然后选择失败/关键 case 单独补采。
`run.py --suite all` **不会**自动启动本套件，须使用上面的独立入口。

频率监控要与测试同时运行。下面的卡号0仅是示例，必须换成已核对的 `ppu-smi` 编号：

```bash
mkdir -p results-dsv41-monitor
ppu-smi -q > results-dsv41-monitor/inventory.log
ppu-smi dmon -i 0 -s pucv > results-dsv41-monitor/clocks.log 2>&1 &
dsv41_monitor_pid=$!
trap 'kill "$dsv41_monitor_pid" 2>/dev/null || true' EXIT
python3 deepseek_v41.py run results-dsv41-all --suite all --acu --clock-note site-locked-1300
kill "$dsv41_monitor_pid" 2>/dev/null || true
wait "$dsv41_monitor_pid" 2>/dev/null || true
trap - EXIT
```

`--clock-note` 只记备注，不锁频；仅在机房确实已锁1300MHz时使用该标签。默认值
`not_verified` 不代表频率已核实。每点保存 `ppu-smi -q`、SDK 编译器版本和可取得的
TileLang git commit；板级采样不能代替短 kernel 的实际频率指标。

有多张空闲卡时，可在不同容器/不同可见卡上把 `--cases` 拆分，结果目录也分开。
不要在同一张卡上同时跑本套件和其他 benchmark；否则带宽和频率对比失去意义。

## 4. 如何计时、校验和采 ACU

默认使用 CUDA-compatible Graph，将50次相同 kernel 调用捕获后用设备 event
计时，除以50；重复7组。输入创建、CPU参考、JIT、初次设备加载和图捕获都在计时区间外。
这是 **hot/repeated** 时间，会复用缓存。默认预热20次。

另测每次写256MiB清扫缓冲后的单次 kernel，保存 `after_scrub_us`。
**它只是尝试制造冷缓存，不能保证清空所有层级**。完整 kernel 的极短时间也含图执行/
调度开销，ACU 的目标 kernel duration 可作补充，不能将图或 Python host 时间
当成 FU 延迟。

若 SDK 不支持 Graph，明确报错，不静默换计时方式。可重跑新目录并指定
`--timing event`，它会含 Python 提交可能带来的设备空闲；不能与 graph 结果无条件混比。

CPU 使用独立数学参考：FP8/FP4 舍入与 scale、带 scale 的 GEMM、含 sink 和
无效索引的 attention、Sinkhorn。量化要求精确一致；GEMM/attention/Sinkhorn
容差写入 `validation.json`，必须保留最大误差。FP4 nearest-even、低/高 nibble
及 E8M0 bytes 有独立单测。输入包含零量化组、随机非均匀 scale、索引重复/-1，
每点计时前默认3次重新污染输出并校验，计时后再次检查结果。
可用 `--check-repeat 20` 加强 attention 重复验证。不接受“只检查不崩溃”。

`--acu` 在普通计时进程结束后另开 profiler 进程，关闭 profiler 自动锁频，
默认 `--cache-control none`，窗口内恰好调用目标 kernel 一次。窗口前依然预热，
所以其流量**不是 after_scrub 的流量**。自动核对 profiler 与计时进程的 device IR、
生成源代码、索引输入哈希；不同则标为 `timing_passed_profile_failed`。
`--acu-cache all` 表示让 profiler 使用自己的缓存控制，不能声称是已证实的全层冷启动。

也可以只补一个 case 的 ACU（用新输出目录）：

```bash
mkdir -p results-dsv41-acu
acu --clock-control none --cache-control none --profile-from-start off \
  --set=full --target-processes all -o results-dsv41-acu/sparse_window \
  python3 deepseek_v41.py case results-dsv41-acu/kernel \
    --cases sparse_window --profile-one
acu -i results-dsv41-acu/sparse_window.acurep --page raw --csv \
  --csv-file results-dsv41-acu/sparse_window.csv
```

## 5. 必须留下哪些数据

| 文件 | 用途 |
| --- | --- |
| `plan.json`、每点 `request.json` | shape、dtype/scale/quant变体、AIU选项、seed、静态化绑定和来源。 |
| `manifest.json`、`summary.csv`、每点 `.log` | 成败清单、原始编译警告/traceback、总体耗时摘要。不可只留成功行。 |
| `result.json` | 所有原始计时样本、每样本 launch 数、运行环境、编译配置、哈希与检查结果。 |
| `input.tir`、`device.json`、`device.tir`、`kernel.ppu.cc` | **实测的同一次编译**产物，仿真必须使用这里的 device.json。 |
| `tensors.pt`、`inputs.json` | 本次合成输入原样留档；后者包含 attention 的全部实际 `topk_idxs`，供导入器确定 gather 地址/分支。不是模型私有权重。 |
| `kernel.so`、`binary.json` | 尝试导出当前已编译 executable，供 SDK 反汇编/资源分析；若版本不支持导出，明确记录原因，不伪造 PASM。 |
| `*-acu.csv` / `.acurep`、`.status.json` | kernel duration、launch grid/block、寄存器/TSM、spill、实际时钟、缓存hit和各级读写字节、FU计数。缺少的指标保留未知。 |
| `clocks.log`、`environment.json`、SDK/TileLang logs | 与实际卡的身份、频率及编译版本对应。 |

用机房现有 SDK 对 **kernel.so 或它包含的设备二进制**做 PASM/资源提取。保存原始
命令和输出；不能另编译一个“相似 kernel”来声称这是本次实测的指令或寄存器数。
内网 agent 如果遇到 API/类型缺失，应保存失败点和版本，禁止将 FP8/FP4 改成
BF16、将稀疏 attention 换成 dense attention、降低 shape 或放宽容差后沿用原标签。

## 6. 与 GTSim 配对，怎么判断是模型问题

后续将 GTSim 带入内网（不用把数据外传），例如：

```bash
case_dir=results-dsv41-smoke/sparse_window
python3 /path/to/GTSim-ppu/scripts/gtsim_tilelang.py "$case_dir/device.json" \
  --inputs "$case_dir/inputs.json" --sim /path/to/gtsim_ir \
  --dag "$case_dir/replay.dag" --timeline "$case_dir/timeline.tsv" \
  --residency "$case_dir/residency.tsv" --report "$case_dir/gtsim.json"
```

先核对输入 IR hash、shape、dtype、layout、tile、grid、threads、编译选项和 attention
索引，再比较性能。历史 fixtures 即使 shape 相同也不能直接代替本次 device.json。
整数 inputs 用于复现索引和编译绑定，不是要求 GTSim 做完整数值功能仿真。

每个算子最终整理一行：case、IR hash、实测频率、实测中位数及波动、模拟 cycles
和按同频率换算的 us、比值、grid/block、register/TSM、cache各级读写量、已知限制。
**cycles / MHz = us**；例如1300周期 / 1300MHz = 1us。
GTSim 当前按单算子冷启动时，优先看 after_scrub 候选与真实流量，不能无条件拿
hot 重复时间作冷启动误差。缓存状态不能核实的点先标记，保留两组数据。

CPU 参考/输入准备不包含在实测时间，设备 kernel 自身的 FP4 转换、scale处理、
索引 gather、softmax/Sinkhorn等包含在内。框架 wrapper 额外的 contiguous、copy、
head padding、索引生成、通信和后续消费算子不在这里；因此这是算子级端到端，
不是模型请求级端到端。

## 7. 当前验证范围与已知编译提醒

本次在已有隔离主机环境完成32/32点的 CPU 数据/参考准备和 PPU device IR lowering，
新增9项主机单测全部通过（含低精度存档和错误结果不得标为成功的检查）。
本次开发环境没有 PPU 实卡 SDK。可验证来源、输入参考和 PPU device IR lowering，
**不能宣称机器码编译及32点实卡执行已通过**，第一轮内网结果仍须验收。
已有主机 lowering 对 sparse attention 给出 ThreadSync 警告，内容涉及条件分支中
同步外提后冲突可能仍存在。必须保留该警告并核查实卡多次正确性，必要时使用 SDK
提供的竞态检查；不能把“成功生成 TIR”当成没有竞态的证明。若需要调整上游源码，
另存有明确差异的新变体，并重新生成真机与模拟两端的输入。
