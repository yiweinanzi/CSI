# CSI-PAIRS V6 全链路完成 Goal（GPT-5.6-sol / ultra）

```text
你是 CSI-PAIRS V6 正式实验的总执行负责人。本任务运行于 GPT-5.6-sol，reasoning effort=ultra。你的职责不是给方案，而是接管现有现场，持续实现、验证、运行、监控、恢复和验收，直到正式 evaluation、全部下游阶段、claims、审计和最终报告全部完成。

第一步处理持久 Goal：若当前线程已有同一目标的 active Goal，读取并继续它，不得重新创建或从头开始；否则立即调用 create_goal，objective 为“彻底解决 CSI-PAIRS V6 evaluation 近 O(N^2)、无断点和无真实进度的问题，在科学语义与来源可追溯性不变的前提下复用既有训练 checkpoint，完成正式 evaluation 和全部下游链路”。不要设置 token budget。只有全部成功标准满足时才能把 Goal 标记 complete。

## 最终目标

彻底解决旧 evaluation 极慢、没有中间产物、不能恢复、没有完成百分比的问题；证明优化实现与旧版科学语义等价；建立合法的新 source/run/migration/approval 身份；受控切换并用两张 A100 跑完正式 evaluation；验收后按固定顺序完成 risk、path、全部 baselines、controls 和 claims；交付真实指标、gate、SHA、审计证据、性能修复及详细中文报告。

## 权威位置

- 冻结协议：/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/Idea1-CSI-PAIRS-冻结版-零基础阅读稿-v6_VSCode兼容版.md
- 旧固定仓库：/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server
- 旧固定提交：9850fffe0b34f16b45066973308f18b10555ca5d
- 旧正式 run root：/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server/runs/formal-v6-gpu-9850fff-20260825T071800Z
- 正式 dataset：/root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz
- dataset SHA-256：060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac
- checkpoint context SHA-256：89c7ea09678778638dfa16d73266b31383fdcf8ed0ddcbbf81fb780069b64ef5
- 优化 worktree：/root/xunlian/Futaoran/CSI_EVALUATION_LINEAR_RESUME_20260830/code/CSI-PAIRS-v2.0-server
- 优化分支：perf/evaluation-linear-resume-v1，基线 9850fff
- 旧 smoke oracle run：/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/runs/smoke-full-v6-p8-clusterfix-20260826T230312Z
- 旧 smoke fixture：上述 run 同级的 smoke-full-v6-p8-clusterfix-20260826T230312Z.fixture.npz
- 旧进程监督目录：/root/xunlian/Futaoran/formal_external_inputs/supervision/formal-v6-gpu-9850fff-20260825T071800Z

## 已知现场，但必须重新核验

- 3 seeds x 4 arms x 20000 steps 已全部 COMPLETE；不得重训、覆盖或删除 checkpoint。
- factorial 已完成，真实状态为 FAIL / CANDIDATE_NOT_CLAIM，G5=FAIL；其他未评估 gate 不得提升。
- 旧 evaluation PID 曾为 66858，启动时间 2026-08-29 10:52:49，启动 ticks 466542136，身份 SHA bcc357b38dcc6fc737a1b4a04c933663160e47713e53960831803845a96fbc09。
- 2026-08-31 00:02 时旧进程已运行约 37 小时，CPU 约 200%，RSS 约 132 GiB，evaluation 正式文件仍为 0；监督器 PID 曾为 108485。
- GPU0 基本空闲，GPU1 约 16%、1.6 GiB；可用内存约 539 GiB，磁盘约 158 GiB，.pyc=0。机器没有 OOM、驱动故障或锁冲突。
- 优化 worktree 已有未提交改动：formal_evaluation.py 的稳定线性 pair 分组；formal_evaluation_resume.py 的身份、原子 shard、SHA、锁、resume 和 progress；相应测试正在形成。先审查和集成现有工作，禁止覆盖后重新写。

## 不可改变的科学与证据约束

1. 不修改冻结协议、route、数据划分、seeds、训练步数、loss、scheduler、模型数学定义、指标定义、统计方法、阈值或 gate。
2. 不把训练 loss、工程成功、脚本退出 0、文件存在、fixture、局部抽检或候选数据包装成科学 PASS。
3. 保留真实 FAIL、BLOCKED 和 NOT_ASSESSED；科学 FAIL 是结果，不是工程阻塞，应记录并继续后续允许阶段。
4. 旧训练 checkpoint 只能作为不可变 legacy upstream 使用；不得改写 payload、manifest、SHA、旧 gate、旧 approval 或旧 run root。
5. 优化 evaluator 不能冒充 9850fff。必须有新 commit、新 source tree SHA、新 run identity、新 migration receipt 和新 LLM-as-judge approval。
6. 审批只能由 Codex、Claude Code 或 Cursor 完成。不得恢复人工审批，不得自行制造 ACCEPTED；judge 必须审阅实际 diff、测试、benchmark 和迁移证据。
7. 不重装环境，不重新下载已有数据，不提交 dataset/checkpoint/CSV 等大文件，不清理有效证据。
8. 所有 Python 命令使用 PYTHONDONTWRITEBYTECODE=1 和 python -B；持续验证 .pyc=0。
9. 同一输出 shard 只能有一个 writer。禁止两个完整 evaluation 指向同一输出目录。
10. 不因 GPU 利用率低、日志暂时不增长、科学 gate FAIL、可恢复 warning 或 ETA 波动而停止全链路。

## 协作方式

使用最多 4 个并行 agent 加速，但每个文件和模块必须有明确 owner，禁止多人同时覆盖同一文件。建议并行职责为：线性算法与 oracle；resume/progress/locking；migration/provenance/approval；主代理负责协议阅读、集成、正式运行和最终验收。主代理必须亲自审查所有 diff 和关键协议，不能仅采信子代理结论。独立读取和测试可并行；依赖前序证据的切换、审批和正式运行必须串行。

## 阶段 0：接管并冻结现场

读取协议、AGENTS.md、现有 Goal、运行手册、git/worktree 状态、现有 agent 改动、监督器、日志和产物契约。只读重新核验 PID 66858 的 cmdline、cwd、exe、启动 ticks、启动时间、父 PID、CPU 时间和身份 SHA；若身份不符，标记 PID_REUSED_OR_MISMATCH，绝不触碰。若进程仍匹配，继续由低开销监督器保护。记录旧 run 的提交、source/runtime/config/dataset/approval、qualification/factorial、checkpoint index 及 12 个 checkpoint SHA。

不要把时间浪费在重复排查 NVIDIA 驱动；只有 nvidia-smi、CUDA 调用或内核日志出现新证据时才重开硬件调查。旧进程仍运行时禁止启动第二份完整旧 evaluation，也禁止 strace、debug attach、renice、pause、模糊 pkill 或清理目录。

## 阶段 1：完成通用性能修复

先整合现有线性分组和 resume 改动，再修改主 evaluation。必须消除 _response_effect_rows、_compatibility_effect_rows、_paired_score_differences 和相关路径中“每个 pair ID 重扫完整数组”的行为。使用稳定排序/group boundaries 或等价索引，保持旧 np.unique 的字典序、重复键验证、首行选择、tie-break、NaN/Inf 行为、异常文本、输出字段和顺序。禁止物化 N x N 矩阵。

批处理 _response_probe_dataset 与 _native_mask_cover_metrics 中兼容的 model.state/predict 调用；缓存 immutable map/radio/action/mask/patch 输入。根据显存实测选择批量大小，允许自适应缩小 batch 处理 OOM，但不得丢样本、降精度或改变顺序。优先按 checkpoint 或互不重叠 shard 分配两张 A100；worker 完成顺序不得改变最终 canonical 顺序。

将每个 checkpoint 的中间结果按 (seed, arm, substage, scene, edge) 或经证明更安全的独立单元写入 shard。probe fit 是每个 (seed, arm) 的 barrier；保存并认证 probe 状态后才能并行预测。每个 shard 先写同目录临时文件，flush/fsync、schema/计数/SHA 校验通过后原子 rename。最终以 checkpoint index 的 seed-major、arm 顺序 endpoint/alignment/response/full，再按旧 bank/pair/route/bin 顺序流式合并；CSV fieldnames 必须保持旧 write_csv 首次出现顺序，禁止按 worker 完成顺序合并。

实现强身份 resume：manifest 至少绑定新 commit/source tree、config、dataset、output schema、legacy checkpoint SHA、seed、arm、substage、输入范围和输出 SHA。恢复时只跳过身份完全匹配且重新校验通过的 COMPLETE shard；半写、损坏或过期 shard 只重算自身。全局 writer lock 必须非阻塞拒绝第二 writer；final manifest 只能在所有 shard 验证和 canonical merge 后生成。

实现原子 evaluation_progress.json 和只读 status 命令。进度必须由启动前确定的 total_units 与已认证 completed_units 计算，至少包含 percentage、seed、arm、substage、shard、吞吐率、开始/更新时间和 ETA；每完成 shard或最长 60 秒更新一次。不得伪造百分比。

## 阶段 2：测试与语义等价验收

先运行定向单元测试，再跑受影响回归。覆盖空输入、单样本、乱序、重复 pair ID、tie、非整除分块、NaN/Inf、错误 pair、半写文件、损坏 shard、过期身份、重复 resume、双 writer、原子 progress、最终合并顺序和无 N x N 分配。

使用现成旧 smoke evaluation 作为 oracle，不重新运行完整旧算法。逐文件、逐行、逐字段比较所有 evaluation 输出：非浮点字段、主键、字段顺序、行顺序和计数必须完全一致；浮点优先 bitwise 一致，只有冻结协议明确已有容差时才使用该容差并记录最大误差。再从正式 dataset 选取一个不改变原数据的真实小子集，用旧版与新版端到端对照。任一未解释差异都阻止切换。

做 N、2N、4N benchmark，记录墙钟、CPU 时间、GPU 利用率、峰值 RAM/VRAM、输出 SHA 和经验增长指数。证明 pair 聚合已脱离近 O(N^2)，目标指数 <1.5；检查内存没有转化为新 O(N^2)。对测试子进程做中途终止、恢复、损坏 shard 和 stale identity 测试；绝不能用旧 PID 66858 做恢复实验。根据真实 benchmark 给出正式 evaluation 的 ETA 和输出空间上界。

切换门槛必须同时满足：LINEAR_GROUP_TEST=PASS、SMOKE_ORACLE_EQUIVALENCE=PASS、REAL_SUBSET_EQUIVALENCE=PASS、SCALE_TEST=PASS、RESUME_TEST=PASS、CORRUPT_SHARD_TEST=PASS、STALE_IDENTITY_TEST=PASS、LOCK_TEST=PASS、PROGRESS_TEST=PASS、DISK_BUDGET=PASS。

## 阶段 3：合法迁移与新审批

现有证据合同不允许新 source 直接复用旧 run root 或旧 approval，但 checkpoint payload 的科学身份允许作为不可变 legacy input。实现集中式、严格的 migration contract，不要在各消费者中散落宽松旁路。

新增版本化 migration request/receipt 和统一 upstream resolver。迁移证据必须绑定：旧 commit/source/runtime/config/dataset/protocol/approval；qualification gate、manifest、teacher；factorial gate、manifest、normalization、frozen pilot、checkpoint/resume index；12 个 checkpoint 文件 SHA 与 payload identity；旧 evaluation 冻结收据；base-to-new diff；oracle、真实子集、N-2N-4N、resume、corrupt/stale/lock/progress 测试报告；新 commit/source/new root/run nonce/compute plan。旧 factorial FAIL/G5 FAIL 必须原样继承。

默认 require_stage_manifested_gate 等严格路径不得弱化。只有 migration receipt 和新 approval 完全匹配时，统一 upstream resolver 才能从只读 legacy root 提供 qualification/factorial/checkpoints。evaluation、risk、path、external baselines、resource controls、claim controls 和 claims 必须使用同一个 resolver，不能各自猜路径或复制旧 gate。

修复 formal_claim_controls 的 checkpoint schema bug：正式 checkpoint 实际包含 source_tree_sha256、requirements_lock_sha256、runtime_provenance_sha256、runtime_provenance。应复用版本化 canonical checkpoint validator，对 legacy/current schema分别严格认证；禁止简单删除 exact-key 检查或忽略未知字段。测试缺字段、额外字段、篡改 provenance、legacy 正常 checkpoint、current checkpoint，以及 shuffled-pair/retention 的真实加载路径。

提交优化代码、测试、benchmark 和迁移说明，但不提交大型产物。生成新的 approval request，由独立 Codex/Claude Code/Cursor judge 审查后通过官方 CLI 创建 accepted receipt。若 judge 拒绝，保留拒绝证据，修复后重新提交；不得改写历史或伪造接受。

## 阶段 4：受控切换

如果旧 evaluation 在优化门槛完成前自然结束，先按固定版合同验收其产物；若 PASS，优先使用旧结果，不做无意义切换。如果旧进程仍运行，只有全部技术门槛、新 migration receipt 和 NEW_LLM_APPROVAL=ACCEPTED 后才允许切换。

本 Goal 明确授权一个窄范围例外：满足全部切换门槛后，冻结旧 PID 的身份、命令、提交、输入 SHA、运行时间、CPU/RAM/GPU、日志、锁及完整产物清单，然后只对启动 ticks 和身份 SHA 仍完全匹配的具体 PID 发送正常 SIGTERM，并等待它释放资源。禁止 pkill、模糊匹配、SIGKILL、删除旧目录或把旧 partial output 混入新 run。写出终止与切换收据。若身份不匹配，绝不触碰该进程。

旧 PID 退出后必须停止旧 supervisor 的自动续跑能力。迁移 coordinator 对旧 run 的 `.csi-pairs-operation.lock.guard` 持只读共享 flock，同时对新 run 持正常独占 operation lock，防止旧 CLI 在新 evaluator 读取 legacy checkpoint/factorial 期间重新写旧 root；共享锁不得改写旧 lock JSON。migration receipt 必须绑定 freeze 时间、旧 lock 状态和 legacy inventory SHA。

## 阶段 5：正式优化 evaluation

创建独立新 run identity 和输出目录，从认证过的 12 个 legacy checkpoint 只读运行新 evaluator，不重新训练。使用现有两张 A100，按实际 profile 提高 batch 和并行度；目标是最短可信墙钟时间，而不是为了显示高利用率制造争用。每张 GPU 只承担明确 shard，CPU worker、内存和 I/O 设置必须有界。启动耐久、幂等、可恢复 supervisor；Codex 断线后仍应继续，重连后从收据恢复状态。

持续监控 progress、checkpoint/shard、GPU、CPU、RAM、磁盘、锁、日志和 .pyc。异常退出时先验证已有 shard，再从最后合法 shard 恢复，不删除证据后从头跑。瞬时错误可按仓库既有且不改变语义的规则重试并登记；OOM 只能缩小 batch 后恢复相同 shard。磁盘预算必须包含预计最终输出、临时 shard、合并空间和安全余量。

evaluation 完成后先验收：正常完成日志；旧合同要求的 9 个 CSV、gate.json 和 canonical manifest；schema、字段顺序、主键、行数、样本范围和跨文件守恒；无未契约化 NaN/Inf、截断、重复、漏样本、临时文件或半写文件；config/dataset/checkpoint/source/migration/approval 身份一致；全部 SHA 对账；官方验证器和指定测试通过。resume/progress/shard/tmp/lock 必须位于正式 evaluation inventory 之外。只有 OUTPUT_ACCEPTANCE=PASS 才能继续下游。

## 阶段 6：跑完全部下游

严格按仓库 Goal/正式 all 的固定顺序执行，不自行改序：risk -> path -> external baselines -> representation baselines -> resource controls -> scene-ID -> shuffled-pair -> retention -> claims；若正式 all 还包含 external validity、literature resources、RT calibration 或其他协议阶段，也按仓库固定顺序纳入，不能静默遗漏。

每个阶段执行“前置输入验收 -> 既定命令 -> 原始日志 -> 官方验证器 -> 审计收据”。科学 FAIL 记录为真实结果并继续所有协议允许的阶段；工程失败、输入身份错误或产物不完整必须在本阶段通用修复并从合法 checkpoint/receipt 恢复。不得跳样本、放宽参数、写占位文件或把 optional warning 提升为 blocker。

claims 必须建立逐项 claim-to-evidence 映射，只引用验收 PASS 的产物，并写清 metric、split、comparison、condition、sample unit、statistics、限制及 gate 状态。factorial/G5 的真实 FAIL 不得被后续工程成功覆盖。缺失证据保持 NOT_ASSESSED。

## 阻塞与继续规则

以下才是硬阻塞：协议存在不可推断的科学歧义；PID 身份不匹配；oracle 或真实子集不等价；来源/审批身份不合法；输出损坏或缺样本；磁盘无法保留安全余量；官方验证器失败且无法通用修复。遇到硬阻塞时先尝试至少三种不改变实验语义的可验证解决办法；确实需要用户提供不可推断信息时才提一个最小问题。

以下只记 warning 并继续：科学 gate FAIL、GPU 未满、CPU 阶段不使用 GPU、日志短时不增长、ETA 波动、可恢复瞬时故障、非必需 optional gate 缺失、不会影响论文结果的非关键检查。不得制造奇怪 blocker，也不得跳过任何会影响真实论文指标、完整性或可追溯性的工作。

## 状态汇报与持久运行

开始后先报告：旧 PID 身份、旧进程是否仍运行、优化 worktree 改动、当前测试状态、迁移方案、两卡状态和下一门槛。此后每个主要里程碑或最长 30 分钟汇报一次真实证据；不要用主观总百分比掩盖未知工作量。每次汇报必须包含：

OLD_EVALUATION=<RUNNING|EXITED|TERMINATED_AFTER_VALIDATION|PID_MISMATCH>
OPTIMIZED_COMMIT=<sha|NOT_READY>
LINEAR_GROUP_TEST=<PASS|FAIL|IN_PROGRESS>
ORACLE_EQUIVALENCE=<PASS|FAIL|IN_PROGRESS|NOT_RUN>
REAL_SUBSET_EQUIVALENCE=<PASS|FAIL|NOT_RUN>
SCALE_TEST=<PASS|FAIL|NOT_RUN>
RESUME_TEST=<PASS|FAIL|NOT_RUN>
PROGRESS_REPORTING=<PASS|FAIL|NOT_RUN>
MIGRATION_RECEIPT=<PASS|FAIL|NOT_READY>
NEW_LLM_APPROVAL=<ACCEPTED|REJECTED|NOT_RUN>
CUTOVER=<NOT_READY|READY|COMPLETE|NOT_NEEDED>
FORMAL_EVALUATION=<RUNNING percentage/ETA|PASS|FAIL|NOT_STARTED>
DOWNSTREAM_STAGE=<当前阶段和状态>
GPU=<两卡利用率/显存/任务归属>
DISK_FREE=<数值>
PYC_COUNT=<数值>
FIXED_VERSION_CONTAMINATION=<NO|YES>

## 最终完成标准与交付

只有以下全部满足才可 update_goal(status=complete)：性能修复有独立 commit；所有等价性/规模/resume/lock/progress 测试 PASS；migration 与新 approval 合法；正式 evaluation OUTPUT_ACCEPTANCE=PASS；全部下游按固定顺序执行并验收；claims 完成或因真实科学证据明确保持 NOT_ASSESSED；旧固定 run 未污染；.pyc=0；最终审计和报告完成。

最终交付必须包含：优化 commit/source SHA；旧现场冻结证据；migration request/receipt/new approval；profile、oracle、真实子集、N-2N-4N benchmark；resume/故障恢复说明；新 run root；正式 evaluation 和全部下游真实指标；全部 C1-C13、G0-G8 状态；dataset/config/protocol/runtime/checkpoint/manifest SHA；每个阶段的验证命令和收据；未解决限制；以及 /root/xunlian/Futaoran/8月30全部进度汇报.md。报告必须非常通俗、详细说明过去工作、为什么旧版慢、怎样证明新版可信、最终结果能和不能支持哪些论文结论。

持续执行，不要停在建议、计划、局部代码、单元测试或“等待用户确认”。除非触发真正硬阻塞，否则自行解决问题并继续下一阶段。全链路未完成时不得发结案式回复。
```
