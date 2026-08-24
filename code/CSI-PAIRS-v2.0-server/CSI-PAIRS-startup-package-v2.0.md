# CSI-PAIRS 论文启动包 V2.1（V6 协议修订）

> 打包边界：`build_server_bundle.sh` 生成内部实验交付包，不能作为匿名 supplementary
> 提交。匿名投稿包必须使用 `formal_v2/scripts/build_anonymous_supplement.sh`；它会排除内部
> Git provenance，以及没有下游再分发许可的第三方文件。

> 兼容性说明：文件名保留 `v2.0` 是为了不破坏冻结审计路径；数据/产物 schema 和打包根保留 V2.1，正式配置与运行门为 V2.3 V6。

> 日期：2026-08-08（Asia/Shanghai）
> 版本性质：正式实验执行版，不是新增科学结果
> 当前状态：`CODE_READY_FOR_FORMAL_INPUT`
> 当前科学结论：`SCIENTIFIC_EVIDENCE=NOT_ASSESSED`；归档 fixture 仍为 `scientific_use=FORBIDDEN`
> 工程基线：合作者 GitHub `origin/main@eef3040c13264829cda1f4398009f691b52038ae`；审计代码提交 `4074e98fe3c1d1ddccee79672f312b9111185992`；旧本地实现不作为代码来源
> 科学设计约束：冻结 Idea V6；归档 No-X/null/shortcut 失败仅作历史追溯

## 0. 一页结论

V2.1 按冻结 V6 实现数据再生成门、资格门、patch F/P、严格四臂、两城市定位、统一评测、风险、路径、资源控制和证据汇编代码。当前没有运行正式实验，代码存在不等于 gate 通过。

本版解决了 V1.x 最重要的五个阻塞：

1. 新增 qualified RT/实测数据入口，强制引擎版本、配置哈希、许可、材质、坐标、相位规范、clean target 和重复测量。
2. 每个 bank 随机化 natural anchor 与 bit-to-primitive mapping；No-X 仍读取真实 source map、radio、patch mask/query 和 typed signed edit，不再用零图代理。
3. 正式训练前先验收 repeat noise、physical-only primary route、独立 teacher sensitivity strata、teacher reconstruction、oracle-x、No-X、action-swap、null safety 与 variant-ID；teacher 不参与 raw-CSI primary inclusion。
4. Endpoint/Alignment/Response/Full 使用同一 PyTorch 架构、初始化、batch、mask、步数和完整前向合同；四臂参数量和前向分支一致。
5. 定位只读取共享 state；目标 support/query 按唯一 receiver position 隔离，最高独立统计单位为 base-map-cluster，并对 seed、draw、cluster 做配对汇总。

V2.1 的 dry run 只能证明软件执行。fixture qualification 必须以退出码 `1` 和认证状态
`DRY_RUN_FAIL_NOT_EVIDENCE`、`passed=false`、`scientific_use=FORBIDDEN` 结束；外层 verifier
仅在完整复核这一预期 fail-closed 状态后返回 `0`。这不是 qualification PASS。fixture 在
元数据、gate 和报告中永久标记 `FORBIDDEN`，其任何数字都不得进入论文。

## 1. 本版交付物

| 交付 | 文件 | 状态 |
|---|---|---|
| 正式数据合同与独立再生成门 | `formal_v2/DATA_CONTRACT.md`、`formal_v2/formal_dataset.py`、`formal_v2/formal_data_verification.py` | 代码已实现；未运行正式 verifier |
| 冻结正式配置 | `formal_v2/configs/formal_v2.json` | 完成；读正式数据前再由负责人签字冻结 |
| 非科学小配置 | `formal_v2/configs/formal_v2_smoke.json`、`formal_v2/formal_fixture.py` | 完成；只测代码 |
| 六条件 wrong-map | `formal_v2/formal_wrong_map.py` | 代码已实现；必须复用资格门冻结 teacher；ridge 不冒充外部 baseline |
| Stage-0 与 Response 资格 | `formal_v2/formal_qualification.py`、`formal_v2/formal_teacher.py` | 代码已实现；未运行 |
| patch F/P 与严格四臂 | `formal_v2/formal_model.py`、`formal_v2/formal_factorial.py` | 代码已实现；未运行 |
| 两城市定位与统计 | `formal_v2/formal_factorial.py`、`formal_v2/formal_statistics.py` | 代码已实现；city-level k 和 V6 多层估计量 |
| CGS/Response/q_comp/p_fail/path | `formal_v2/formal_evaluation.py`、`formal_v2/formal_risk.py`、`formal_v2/formal_path.py` | 代码已实现；未运行 |
| 资源、scene-ID、外部模型控制 | `formal_v2/formal_controls.py`、`formal_v2/formal_scene_id.py`、`formal_v2/formal_external.py`、`formal_v2/external_adapters/` | shuffled-pair、retention、scene-ID 与五个资源控制已有首方可执行实现和认证清单；Wi-GATr 是当前唯一 C1 合格模型，WiSER 为 C1 不合格的 style-controlled 实现；均无正式结果 |
| 外部论文与 Sionna 设施 | `formal_v2/WAIBU_INTEGRATION.md`、`formal_v2/sionna_scene_export.py`、`formal_v2/sionna_facility.py` | 10 项资源哈希登记、表征/地图基线、formal world 到 PLY/XML 的 exporter 和 G8 adapter 已实现；G8 active/null 均按 cluster CI 判门；无正式结果 |
| 命令入口 | `formal_v2/formal_cli.py` | 完成；上游失败会阻断四臂 |
| V2 自包含工具 | `formal_v2/formal_io.py`、`formal_v2/formal_baselines.py` | 完成；不再导入冻结 V1 `experiments` 代码 |
| 依赖与脚本 | `formal_v2/requirements-lock.txt`、`formal_v2/scripts/setup_formal_v2.sh`、`formal_v2/scripts/run_formal_v2.sh` | 完成 |
| 服务器打包/自检 | `formal_v2/scripts/build_server_bundle.sh`、`formal_v2/scripts/verify_server_bundle.sh`、`README.md` | 完成 |
| V2 论文源稿 | `paper_v2/main.tex` | 完成；V1.26 论文未改动 |
| Claim-Evidence | `artifacts/v2_0_claim_evidence_contract.json` | 完成；所有科学 claim 仍 blocked |
| V2 测试 | `formal_v2/tests/test_formal_v2.py` | 完成 |
| 最终验证报告 | `artifacts/v2_0_verification.md` | 完成；汇总测试、dry run、PDF 和 V1 冻结锚点 |

## 2. 继承且不得删除的失败事实

- V1.1 scene 3 的 82/82、scene 4 的 84/84 No-X null units 超过 0.018 tolerance，两处均为 100%。
- `map_feature_hurts`、`variant_id_signal`、`oracle_x_not_helpful`、`beam_readout_insensitive_to_copy` 四个 warning 必须在 V2 输出继续出现；未评估不能写 PASS。
- 在新的非 fixture gate 通过前，禁止 `first`、`outperform`、`calibrated`、`label-efficient`、`synergy`、`method efficacy`、`localization improvement` 和 `real causal grounding`。
- V2 的 relative-edit 修复是新假设，不是既有失败已经被修好。只有 untouched selection banks 能决定它是否有效。

## 3. 正式 gate 顺序

| Gate | 输入 | 通过标准 | 失败退出 |
|---|---|---|---|
| G0 文献与资源 | 投稿前检索和资源记录 | 可核查、未过期、许可完整 | 删除首创性表述 |
| G1 RT/repeat/route | clean CSI 与独立 repeats | 每个 bank 的 physical primary active/null 覆盖达标，teacher sensitivity 作为独立 audit stratum 通过一致性门 | 修 RT、gauge 或资产；不能用 teacher 改写 primary inclusion，也不能放宽 held-out 阈值 |
| G2 teacher 与 No-X | source-train teacher、method-selection | teacher/readout、No-X、null、shortcut 全通过 | 停止四臂 |
| G3 单分支 | 冻结 probe 与 target-free response | Alignment/Response 分别成立且 null 安全 | 删除失败分支结论 |
| G4 联合价值 | 四臂及五个资源/concat 控制 | 七个子门全部 PASS | 不写 synergy；NOT_ASSESSED 不是 PASS |
| G5 两城市 | target support/query，k=0/8 主检验 | 两城市不反向退化；median/P90 与 bank CI 同报 | 删除跨城/label-efficiency/improvement claim |
| G6 风险 | source calibration 与 target k=0 common support | q_comp/p_fail、符号、校准、AURC 全通过 | 删除 calibrated risk |
| G7 路径机制 | persistent paths 与 no-op retrace | 四档、平衡、趋势和零路径等效通过 | 删除机制解释 |
| G8 外部有效性 | 第二引擎或受控实测 banks | active 方向和 null 稳定性独立通过 | 只写 simulator-defined/consistent |

独立数据再生成门、G1 和 G2 是四臂硬前置。CLI 会认证 dataset/config/teacher 哈希；`--allow-nonscientific-fixture` 只允许软件链路，不会改变 `scientific_use=FORBIDDEN`。

## 4. 数据负责人现在要准备什么

1. 七个互斥 source role 都必须有独立 banks：encoder-train、method-selection、probe-train、probe-selection、calibration-fit、calibration-selection、final-unseen-bank。
2. 至少两个 target cities；每城建议至少 4 个独立 banks。每 bank 至少 128 个 support-pool positions，另有完全不重叠的 query positions。
3. 每个 bank 的完整 `2^d` canonical worlds、共同 receiver positions、双向 Hamming-1 edges、每 bank 独立 primitive permutation 与 natural anchor。
4. 至少 3 个独立 observation repeats；确定性 RT 另存 clean target。重复噪声不能在 paired worlds 间复制。
5. 引擎 name/version/config 文件及 SHA-256，地图/材质许可，BS/坐标定义，CSI 单位、real/imag 布局和 pair-consistent phase/gauge 规则。
6. 第二引擎或受控实测数据若可用，必须作为实际 `external_validation` banks 放入 archive；只有 metadata 声明会被拒绝。

数组和 metadata 的精确字段见 `formal_v2/DATA_CONTRACT.md`。

## 5. 两个实验怎样运行

### 实验一：六条件 wrong-map

内置 `controlled-relative-map-ridge-diagnostic` 只负责验证六条件、相同 CSI、相同 denominator 和配对评分链路：

1. correct；
2. paired active alternative；
3. paired null alternative；
4. wrong city；
5. geometry destroyed；
6. empty。

它会输出 `wrong_map/per_sample_results.csv` 与 `summary.csv`。正式 C1 仍要求把同一条件合同接到至少两个准确命名、许可明确的外部 map-conditioned models；内置 ridge 的结果不得写成领域现象。

### 实验二：Response 资格

source-encoder-train 只拟合 CSI-only masked teacher、冻结 readout、归一化和资格 probes；source-method-selection 决定：

- raw physical target 是否可学；
- oracle-x 是否胜 copy；
- No-X 是否逐 bank 胜 copy、no-action、fixed-checkpoint action-swap；
- null predicted delta 是否留在 0.018 dead zone；
- randomized variant ID 是否仍有异常信号；
- absolute source-map 是否再次伤害泛化。

输出在 `qualification/`。最重要的是 `gate.json`、`response_gate.csv`、`null_safety.csv` 和 `shortcut_audit.json`。

## 6. 四臂实现边界

`formal_v2/formal_model.py` 实现继承完整二维 CSI teacher encoder 的 F、多通道地图 encoder、radio+BS-pose token、cross-attention fusion、typed signed-edit encoder、query embedding，以及 latent/physical 双输出。下游定位只读取保留的 F state；teacher、predictor、edit encoder、route 和 readout 都不进入定位输入。

四臂复用同一 teacher、初始化、StepPlan、bank/edge/direction/mask/query 顺序和 checkpoint 规则，arm 只改变 loss factor。代码记录 forward、FLOPs、梯度范数和 wall time；资源差异必须由 equal-FLOP 与三类 concat 控制实际验证。

这是正式可执行 baseline architecture，不代表最终最优 backbone。若换成 CSI-MAE/Transformer，必须保持同输入 allowlist、四臂同结构、同 forward 预算和 checkpoint 规则，并重新跑全部测试。

## 7. 启动命令

### 7.1 建环境

```bash
formal_v2/scripts/setup_formal_v2.sh /unused/path/csi-pairs-v2-env
```

如本机 Python 的 CA 证书异常，先修复证书，不要把 `--trusted-host` 写入长期实验脚本。

### 7.2 先做代码检查（不运行实验）

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m formal_v2.scripts.check_python_syntax formal_v2
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s formal_v2/tests -v
```

这只证明静态导入和已覆盖语义测试通过，不证明资格、四臂或科学 gate。

### 7.3 检查正式数据

```bash
PYTHONDONTWRITEBYTECODE=1 /unused/path/csi-pairs-v2-env/bin/python -m formal_v2.formal_cli inspect-data \
  --config formal_v2/configs/formal_v2.json \
  --dataset /path/to/csi_pairs_formal_v2_1_v6.npz \
  --output /unused/path/data-inspection
```

### 7.4 独立再生成后准备正式运行

```bash
CSI_PAIRS_PYTHON=/unused/path/csi-pairs-v2-env/bin/python \
CSI_PAIRS_FORMAL_DATASET=/path/to/csi_pairs_formal_v2_1_v6.npz \
CSI_PAIRS_FORMAL_OUTPUT=/unused/path/formal-run \
CSI_PAIRS_COMPUTE_PLAN=/path/to/authorized-compute-plan.json \
CSI_PAIRS_VERIFIER_MANIFEST=/path/to/independent_rt_verifier.json \
CSI_PAIRS_EXTERNAL_ADAPTER_MANIFEST=/path/to/external_adapters.json \
CSI_PAIRS_EXTERNAL_VALIDITY_MANIFEST=/path/to/external_validity.json \
CSI_PAIRS_LITERATURE_RESOURCE_MANIFEST=/path/to/literature.json \
CSI_PAIRS_RT_CALIBRATION_MANIFEST=/path/to/rt_calibration.json \
CSI_PAIRS_FULL_RUN_PHASE=prepare \
  formal_v2/scripts/run_formal_v2.sh
```

`prepare` 在任何正式后期训练前认证资源、运行时、许可、预算、G0、独立 RT、数据再生成与 G1/G2，然后写入 `approval/request.json` 并停止。允许的 LLM-as-judge（`codex` / `claude-code` / `cursor`）复核该 request 及早期 gate 后，在 run root 外签发一次性 approval：

```bash
PYTHONDONTWRITEBYTECODE=1 /unused/path/csi-pairs-v2-env/bin/python -m formal_v2.formal_cli create-run-approval \
  --request /unused/path/formal-run/approval/request.json \
  --output /path/to/formal-run-llm-judge-approval.json \
  --judge codex:bound-session \
  --expires-utc 2027-01-01T00:00:00Z \
  --attest-llm-judged
```

使用完全相同的输入环境，把 `CSI_PAIRS_FULL_RUN_PHASE` 改为 `run`，并设置 `CSI_PAIRS_LLM_JUDGE_APPROVAL_MANIFEST=/path/to/formal-run-llm-judge-approval.json`。runner 只恢复同一份已认证的 prepare root；旧 run、布尔 approval、输入/运行时/gate/预算变化、跨 run 重放和已消费 approval 均被拒绝。完整 compute-plan 字段和所有默认 manifest 见根 `README.md` 第 5 节。每次正式 run 使用新的输出目录；脚本拒绝覆盖 stage 目录。

## 8. 论文 V2 图表与结果槽位

`paper_v2/main.tex` 当前包含三个不使用实验数字的协议图，以及两个明确标记为 `PLANNED` 的结果 schema：

| 图/表 | 当前内容 | 允许填入结果的条件 |
|---|---|---|
| Figure 1 | 六条件 paired-intervention audit protocol | 图本身是协议；C1 结论仍需两个 C1-eligible 模型的正式 rows |
| Figure 2 | CSI/map、shared F/P、target-only loss 路径与 retained module allowlist | 方法/信息预算图，不是效果证据 |
| Figure 3 | 七个 source roles、target support/query 隔离和 G0-G8 fail-closed 顺序 | 协议图，不是 gate 通过证据 |
| Table 3 | Endpoint/A/R/Full 的 active CGS、null gap、Response NMSE、FLOPs schema | 对应 G3/G4 正式 rows 生成并复核后逐格填写 |
| Table 4 | 两城 k=0/8/32/128 median/P90 与 frozen k=0 risk schema | G5 与独立 risk gate 的正式 rows 生成并复核后逐格填写 |

正式结果 panel 在出现经过认证的非 fixture rows 前不实例化。任何 fixture、smoke、单测或 schema PASS 都不能替换 `PLANNED` 单元格。

每个最终 caption 必须写：输入/对照、独立 scene banks 数、训练 seeds、label draws、误差区间、单位和冻结门槛。表中的 `NOT RUN` 不能用 fixture 数字替换。

## 9. 论文状态和写作动作

- V2 论文是独立目录 `paper_v2/`；`paper/main.tex` 与 V1.26 PDF 未改动。
- 当前稿件是 pre-experiment protocol freeze；摘要不包含完成式实验结论，不得把软件 ready 改写成 method effective。
- G2 失败：论文转向数据/readout qualification，不再写 Response 方法。
- oracle 通过而 No-X 失败：删除 deterministic Response headline，转概率 response 或 paired compatibility。
- G4 interaction 失败但 Full 胜两单支：最多写 complementary，不写 synergy。
- Full 未胜两单支：取消 CSI-PAIRS Full headline，围绕更强单分支或 audit 重新定题。
- G8 未通过：结论止于 simulator-defined/consistent。

## 10. 目前仍需外部资源

以下内容不能由代码生成，也没有在工作区被发现：

1. qualified RT engine 与真实版本化配置；
2. 许可明确的场景、地图、材质与 BS 坐标；
3. 第二引擎或受控实测 paired CSI；
4. 足够多的独立 source banks 和两个 target cities；
5. 至少两个不同且满足 C1 资格规则的外部 map-conditioned models；当前仓库只有 Wi-GATr 合格；
6. 正式 GPU 预算与训练时长。

因此 V2.1 的准确裁决是：**A + R1 已冻结，`PAPER_PROTOCOL_GO=GO`；允许执行 fixture smoke 与正式输入的只读 preflight。外部输入、许可、CUDA 和预算未关闭前，`FORMAL_GO=NO-GO`、`SCIENTIFIC_EVIDENCE=NOT_ASSESSED`，不能启动正式训练或论文主结果写作。**
