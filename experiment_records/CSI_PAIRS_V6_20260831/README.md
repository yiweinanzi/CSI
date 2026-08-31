# CSI-PAIRS V6 实验记录快照（2026-08-31）

本目录是 CSI-PAIRS V6 当前正式实验的可公开、轻量级记录快照。它保存源码之外的日志、审批、门状态、清单、小型指标、smoke oracle、修复回归记录和历史进度文档；不包含正式数据集、shard、checkpoint、模型权重、运行环境或其他大文件。

## 归档身份

- 快照时间：`2026-08-31T15:37:13+08:00`
- 固定正式代码：`9850fffe0b34f16b45066973308f18b10555ca5d`
- 优化比较器基线：`ac83473543251f1f0b2a73b02e2ab953b84e8407`
- 正式 run：`formal-v6-gpu-9850fff-20260825T071800Z`
- 正式 dataset SHA256：`060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac`
- dataset 文件本身：未归档

当前集成分支还包含三个独立验证过的提交：fixture streaming 身份构建器、迁移证据 runner、旧 evaluation 退出后证据冻结合同。提交 SHA 以本分支 Git 历史为准。

## 内容

| 目录 | 文件数 | 字节数 | 内容 |
| --- | ---: | ---: | --- |
| `formal_external_inputs/` | 385 | 5,893,796 | 正式 launcher、审批、监督日志、OOM/退出证据、C11/C13 小型记录、迁移请求与测试收据 |
| `formal_run/` | 44 | 1,879,640 | 正式 qualification、wrong-map、factorial 的小型 JSON/CSV/清单和 gate |
| `smoke_oracle/` | 272 | 17,013,405 | 已有 smoke 全链路 oracle 的小型输出；科学用途为 `FORBIDDEN` |
| `resume_repair/` | 226 | 3,933,955 | resume 修复测试和小型 fixture 记录 |
| `baseline_repair/` | 222 | 3,761,576 | baseline 修复测试和小型 fixture 记录 |
| `legacy_handoff/` | 5 | 4,832,560 | 旧交接说明、工作树补丁、文件清单和 SHA 清单 |
| `training_top_level_logs/` | 14 | 202,570 | “正式开始训练”目录顶层的小型日志、JSON 和 SHA 收据 |
| `root_documents/` | 3 | 64,750 | 历史进度、Goal 和实验数据说明 |

## 排除规则

归档只复制普通文件，单文件必须小于 `5 MiB`。以下类型无论大小均排除：

- 数据和数组：`.npz`、`.npy`、`.bin`
- checkpoint/权重：`.pt`、`.pth`、`.ckpt`、`.onnx`
- 压缩包和论文附件：`.zip`、`.tar`、`.tar.gz`、`.tgz`、`.pdf`
- 依赖和缓存：`.whl`、`.pyc`、`.pyo`、venv、runtime、cache

正式 run 因此排除了 47 个文件，共 `4,147,956,757` 字节，其中包括 checkpoint 和数 GB 的逐样本 CSV。`formal_external_inputs` 排除了 7 个 PDF，共 `38,154,542` 字节。正式 dataset 为约 2.0 GB，所在 dataset 目录约 4.2 GB，均未进入 Git。`private_audit/` 依照仓库公开边界保持私有，只保留公开产物中的可验证引用。

## 快照时科学状态

- 四臂训练：`12/12 × 20,000 steps` 完成。
- Factorial：interaction `-0.1129042110218661`，bootstrap 95% CI `[-0.3622278912898419, 0.12138785993190301]`，`G5=FAIL`，`scientific_use=CANDIDATE_NOT_CLAIM`。
- 旧正式 evaluation：因内存增长最终 OOM 退出，没有形成可验收的正式 evaluation 文件。
- 优化 evaluation：快照时仍在做冻结旧实现与新实现的真实子集等价验证，不能标为 PASS。
- `risk/path/baselines/controls/claims`：快照时尚未正式续跑。

这些文件是工程证据和历史记录，不自动构成论文科学结论。Smoke 结果、fixture、dry-run、候选数据、工程 `PASS` 和未验收迁移证据不得写成正式科学主张。后续合法产物会以新的提交追加到同一分支。

## 2026-08-31 runtime 追加

应用户要求，本分支随后追加了最终集成 runtime 的四项修复：process-safe migration controls、shuffled/retention checkpoint 全量预验、受 guard 保护的 post-exit stale-lock 归档，以及 raw/canonical config SHA 合同。对应 runtime 分支远端提交为 `8d489b2387e7bb6c988a41d9e0d57b8a6cffc4d4`，formal source-tree SHA-256 为 `aa5b1d6a1062d68bb5de045b40f042e1a453d14a8d73632ad0be86dc7b07caff`。

集成聚焦回归为 `131 passed`；三个独立组件回归分别为 `344 passed, 6 skipped`、`101 passed` 和 `59 passed`。GPU 隐藏的最终全量回归为 `879 passed, 12 skipped`，无失败，耗时 566.6 秒。详细绑定见 `formal_external_inputs/github_uploads/CSI_PAIRS_V6_RUNTIME_UPDATE_20260831.json`。

## 完整性和安全检查

- Git 本身记录每个归档文件的 blob 身份；提交 SHA 是此快照的内容地址。
- 归档前扫描了 GitHub token、AWS key、私钥、Bearer/JWT 和 URL 内嵌凭据等高置信模式，未发现命中。
- `.pyc/.pyo` 数量为 `0`。
- 源目录仅被读取；复制过程没有移动、清理或改写正式 run、日志、锁、checkpoint 或活跃比较器产物。
