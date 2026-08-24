# CSI-PAIRS 当前代码位置图

路径一律相对仓库根目录（本机根为 `C:\Users\22688\Desktop\CSI`）。不要使用过时的 Autodl `/root/autodl-tmp/...` 绝对路径。

另见操作手册：`code/CSI-PAIRS-v2.0-server/formal_v2/A100_RUNBOOK.md`。
数据到位后的盘点清单：`code/CSI-PAIRS-v2.0-server/formal_v2/EXPERIMENT_DAY_ONE.md`。

## 1. 审查入口

| 用途 | 仓库相对路径 | 说明 |
|---|---|---|
| 冻结论文规范 | `Idea1-CSI-PAIRS-冻结版-零基础阅读稿-v6_VSCode兼容版.md` | 科学与协议行为的最高依据；2026-08-24 已按代码合同修订 |
| Review 提示词 | `CODE_REVIEW_PROMPT.md` | 针对当前 V2.1 实现的双向审计指令 |
| 当前代码根目录 | `code/CSI-PAIRS-v2.0-server` | 唯一默认执行与 review 对象 |
| 当前 Python 包 | `code/CSI-PAIRS-v2.0-server/formal_v2` | V2.1/V6 可执行实现 |
| 当前配置 | `code/CSI-PAIRS-v2.0-server/formal_v2/configs/formal_v2.json` | 正式 schema 与冻结阈值 |
| 数据合同 | `code/CSI-PAIRS-v2.0-server/formal_v2/DATA_CONTRACT.md` | NPZ 字段、权限和验证边界 |
| 当前测试 | `code/CSI-PAIRS-v2.0-server/formal_v2/tests/` | **32** 个 `test_*.py`，**600** 个 `def test_` 方法 |
| 当前论文草稿 | `code/CSI-PAIRS-v2.0-server/paper_v2/main.tex` | 必须反向核对 V6；不得写入未挣得的数字 |
| 操作手册 | `code/CSI-PAIRS-v2.0-server/formal_v2/A100_RUNBOOK.md` | 两阶段正式编排与主机要求 |
| 启动阻断 | `code/CSI-PAIRS-v2.0-server/artifacts/formal_experiment_blockers.md` | 历史清单；G0/C11/G8/数据认证/实测 compute plan 已不再硬阻断启动 |
| C13 LLM-as-judge | `LLM_JUDGE_C13.md` | Codex / Claude Code / Cursor 裁决入口；不接受人类签字 |

目录名仍保留 `CSI-PAIRS-v2.0-server`，包内 schema 和 README 已是 V2.1/V6。为保持
`SHA256SUMS`、ZIP 和脚本路径可复现，当前源码不重命名、不移动。

**600 个 unittest 方法是软件单元测试，不是已执行的 C1–C13 / G0–G8 科学证据。** 本文不声称任何科学 PASS。

## 2. 当前运行模块

```text
code/CSI-PAIRS-v2.0-server/
├── README.md
├── CSI-PAIRS-startup-package-v2.0.md
├── SHA256SUMS                          # 打包树校验清单；本文未复检条目数
├── waibu/                              # 外部资源落地目录；存在本地文件 ≠ G0 PASS
├── formal_v2/
│   ├── formal_cli.py                   # CLI 入口；两阶段 prepare-full-run / all
│   ├── formal_run_approval.py          # 计算计划、预检、LLM-as-judge 批准绑定
│   ├── formal_llm_judge.py             # Codex / Claude Code / Cursor 裁决合同
│   ├── formal_config.py                # 严格配置 schema
│   ├── formal_io.py                    # strict JSON、CSV、SHA、manifest
│   ├── formal_evidence.py              # evidence context、G0-G8/C1-C13 状态
│   ├── formal_runtime_integrity.py     # 运行时锁与安装完整性
│   ├── formal_dataset.py               # NPZ loader、world bank、roles、数据约束
│   ├── formal_data_verification.py     # 独立再生验证 adapter
│   ├── formal_protocol.py              # patch/mask/query/typed signed edit
│   ├── formal_teacher.py               # Stage-0 teacher 与 readout
│   ├── formal_routing.py               # PRIMARY_ROUTE_CONTRACT：物理主路由
│   ├── formal_model.py                 # 共享 F/P 与 loss 原语
│   ├── formal_qualification.py         # G1/G2 Response 资格
│   ├── formal_factorial.py             # 四臂训练、定位、J 与初步 gate
│   ├── formal_localization.py          # 异方差位置头与 few-shot adaptation
│   ├── formal_statistics.py            # J、bootstrap、Holm、区间判定
│   ├── formal_probes.py                # compatibility/response probes
│   ├── formal_response_probe.py        # Response probe 实现
│   ├── formal_action_inverse_response.py
│   ├── formal_physics_response.py
│   ├── formal_evaluation.py            # CGS、response、部分子门
│   ├── formal_risk.py                  # q_comp、p_fail、support、G6
│   ├── formal_path.py                  # A_path、matching、G7
│   ├── formal_wrong_map.py             # 本地 ridge 诊断；禁止作外部模型主张
│   ├── formal_external.py              # C1 外部模型六条件 adapter
│   ├── formal_representation_baselines.py
│   ├── external_adapters/              # Wi-GATr、PMNet 官方 snapshot 与适配
│   ├── formal_controls.py              # equal-FLOP/concat 资源控制
│   ├── formal_scene_id.py              # C2 scene-ID adapter
│   ├── formal_claim_controls.py        # shuffled-pair 与 retention
│   ├── formal_rt_calibration.py        # C11 独立 RT 校准
│   ├── formal_external_validity.py     # G8 第二引擎/真实干预
│   ├── formal_literature.py            # G0 文献与资源 manifest
│   ├── formal_resources.py             # waibu 资源认证
│   ├── formal_claims.py                # claim/gate 汇总
│   ├── formal_baselines.py             # 本地 ridge 基础工具
│   ├── formal_features.py              # 资格探针/诊断特征
│   ├── formal_metrics.py               # AUROC/NLL/Brier/ECE/AURC 等
│   ├── formal_fixture.py               # 永久 FORBIDDEN 的软件 fixture
│   ├── sionna_*.py                     # Sionna 设施/导出（非正式结果）
│   ├── A100_RUNBOOK.md
│   ├── DATA_CONTRACT.md
│   ├── configs/
│   ├── data/                           # 只有 README；没有正式 NPZ
│   ├── scripts/
│   ├── EXPERIMENT_DAY_ONE.md           # 数据到位后的盘点与命令，不是科学证据
│   └── tests/                          # 31 个测试文件
├── paper_v2/                           # LaTeX 草稿，不是科学证据
├── artifacts/                          # 合同和验证说明，不是实验结果
├── output/                             # 草稿产物，不是科学证据
└── paper/official_style/               # ICLR 样式文件
```

主路由合同（`formal_routing.py::PRIMARY_ROUTE_CONTRACT`；阅读稿 2026-08-24 已对齐）：

| 层 | 合同 |
|---|---|
| Alignment 主路由 | 仅全通道物理距离 |
| Response 主路由 | 仅 query-patch 物理距离 |
| Teacher | 独立审计层 / 资格辅助，不参与 active/null 主判定 |

C1 合格外部 adapter：Wi-GATr 与 PMNet。`formal_wrong_map.py` 是 ridge 诊断，`FORBIDDEN` 用于外部模型或功效主张。

## 3. CLI 可达性

入口在 `formal_v2/formal_cli.py`。真实子命令：

| 类别 | 命令 |
|---|---|
| 辅助 | `make-fixture`、`verify-waibu-resources`、`export-sionna-scenes`、`create-run-approval`、`operator-preflight` |
| 单阶段 | `inspect-data`、`verify-data`、`qualify`、`run-wrong-map`、`run-factorial`、`run-evaluation`、`run-risk`、`run-path`、`run-external-baselines`、`run-representation-baselines`、`run-resource-controls`、`run-scene-id-audit`、`run-external-validity`、`run-literature-resources`、`run-rt-calibration`、`run-shuffled-pair-control`、`run-retention-audit`、`assemble-claims`、`export-data-verification` |
| 正式编排 | `prepare-full-run`、`all` |

**正式路径是两阶段，不是旧文档里的单次 `all`。**

```text
prepare-full-run   # 先跑可运行的早期门，再发出 LLM-as-judge 批准请求
    waibu 资源认证
    -> 可选 G0 literature/resources（缺回执/PDF 哈希/决策一致不阻断）
    -> 可选独立 RT / C11
    -> 可选独立 data verification
    -> G1/G2 qualification
    -> 可选独立 G8
    -> 写出 approval/request.json（此时尚未四臂训练）

LLM-as-judge：create-run-approval --request ... --output ... --judge family:id --expires-utc ... --attest-llm-judged
    family ∈ {codex, claude-code, cursor}；批准清单必须写在 prepared run root 之外
    批准清单必须写在 prepared run root 之外

all --approval-manifest ...
    只认证已准备门与精确批准；不重跑可选 G0 / RT / data verification / G8
    -> run-wrong-map -> 四臂 run-factorial -> run-evaluation
    -> run-risk -> run-path
    -> external-baselines -> representation-baselines
    -> resource-controls -> scene-ID
    -> shuffled-pair -> retention
    -> assemble-claims
```

`all` 要求已存在的 prepared run root，以及绑定该次 request 的 `--approval-manifest`。
`--approve-full-experiment` 已弃用，没有授权效力。跳过的 G0 / C11 / G8 / 数据认证
记为 `NOT_ASSESSED`，对应 claim 不能变成 `SUPPORTED`，但不再硬停整条链。

计算计划 schema 仍是 `csi-pairs-full-run-compute-plan-v2`，但缺失、占位零或未实测
的 plan 只作 advisory，不阻断 `prepare-full-run` / `all`。

## 4. 历史与原始包

本 clone 不含 `archive/`，根目录也没有传输 ZIP。仓库 README 写明：历史 V1、本地实验、外部下载与生成包故意不进当前代码树。Review 不应递归审查 PDF、LaTeX 样式、`__pycache__` 或 `.ruff_cache`，除非在追踪版本/产物污染。

## 5. 当前可验证边界

- 打包树存在 `code/CSI-PAIRS-v2.0-server/SHA256SUMS`；本文未复跑 `sha256sum --check`，因此不沿用过时的“117 个文件”计数。
- `formal_v2/tests/`：32 个文件、600 个 unittest 方法。**只证明软件单元测试，不证明 C1–C13 或 G0–G8 已科学通过。**
- 本 clone **没有正式 NPZ**（全树无 `.npz`；`formal_v2/data/` 只有合同 README）。
- `waibu/` 即使有本地 PDF/ZIP，也不能当作已认证正式输入或 G0 PASS；`RESOURCE-001` / `FORMAL_INPUT_READY` 仍按阻断登记。
- 工程状态：`CODE_READY_FOR_FORMAL_INPUT`。G0 回执/PDF 哈希、独立数据认证、C11、G8、实测 compute plan 不再把启动写成 `FORMAL_GO=NO-GO`。缺这些门时对应 claim 仍是 `NOT_ASSESSED`，不是科学 PASS。
- **Windows 不能运行正式证据命令**（平台锁、CUDA 预检与 `fcntl` 运行时均不支持）。
- fixture 永远是 `scientific_use=FORBIDDEN`。
- 上述软件边界不证明 V6 协议已在正式数据上执行，更不证明任何科学 claim。
