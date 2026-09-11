> 本文为前一轮状态。后续接线与修复见 [2026-09-09-paper-completion.md](2026-09-09-paper-completion.md)，当前入口为 paper_run。

# 六张实验表：实现、证据与剩余缺口

**后续代码清理已落实：** 已实际删除 58 个废弃文件，旧审批、迁移、claims 与安装环境闭包认证入口已移除。
当前支持范围和最终检查见 [代码清理报告](2026-09-09-code-cleanup.md)。下文涉及旧认证链的描述为清理前记录，不再代表当前可用命令。

当前修改基于 main@1121a0e。本次没有访问服务器、加载正式 NPZ 或运行新的正式训练。历史数字来自 c591701 快照；不要当作本次修复后的成绩。

## 追加落实的底层精简（以本节为当前状态）

用户追问后，进一步实现了独立训练/评测路径，而不只是在旧后端外套调度器：

- `paper_train.py` 直接进行源域 teacher、四臂训练及可选定位。使用原模型、损失、同种子配对批次和定位代码；新路径没有 LLM 许可、安装环境闭包校验、迁移认证、claims、risk、path。配置、数据、模型摘要和实际环境版本仍记录，恢复时核对关键输入。
- `paper_core.py` 将 alignment、response、native 分成独立任务，各自完成后保存结果。运行 alignment 不会启动 response/native。核心特征提取不再生成捷径特征或四个 response 变体；旧的 17 模型协议只在历史入口保留。附加对照尚未全部迁移到轻量路径。
- `paper_minibatch.py` 定义明确的新协议：每个 optimizer update 只读一个小批次。默认候选为 100/400/1600 次更新、每批 1024 行；只在 `source_probe_selection` 比较候选。所有臂采用相同候选集合。这个预算尚未通过正式数据验证，不能说已证明足够或与旧 2000 次全数据更新等价。
- `paper_arrays.py` 将 NPZ 解包为经过摘要检查的只读 NPY 内存映射；保持路由数据原有精度。源特征逐场景保存为 float32，再以磁盘数组拼接，小批次按需读取。核心特征数值已与旧实现逐项比较一致，float32 与旧神经网络输入精度一致。
- 数据验证的重复观测残差改成逐场景计算；有限值检查也逐场景进行，通道标准差改用分块合并矩，避免 NumPy 方差实现临时复制完整 CSI。仍保留噪声独立性和有限值检查。

**还不能说内存问题全部消失**：路由、teacher/source corpus、定位逐样本统计仍会占内存；内存映射增加解包磁盘空间与 I/O。没有正式 NPZ/GPU 的峰值内存、吞吐和完整恢复测量。

新的一条命令（数据在服务器执行）：

```bash
cd code/CSI-PAIRS-v2.0-server
python -B -m formal_v2.paper_train --dataset /path/to/formal.npz --output /path/to/new-run --device cuda:0 --evaluate
```

仅复用已有 encoder checkpoint 做核心评测：

```bash
python -B -m formal_v2.paper_core --dataset /path/to/formal.npz --factorial-root /path/to/factorial --teacher /path/to/teacher.pt --output /path/to/core-evaluation --device cuda:0 --tasks alignment response native
```

`--tasks alignment` 可单独运行核心 AUROC/null 指标，不等待任何附表任务。新/旧 probe 协议不能混合导出。新训练输出中的 `factorial` 和 `core/evaluation` 会被出表器识别。

合成数据端到端测试已覆盖四臂训练、恢复、定位，以及三个核心评测任务分别运行/汇总；这些测试结果永久标记为 FORBIDDEN，不是正式实验指标。下面关于旧调度器的限制仍成立；主表外部基线后端与附表完整迁移尚未完成。

本轮相关回归结果：**159 passed，8 skipped**。跳过项为 6 项 CUDA 测试和 2 项需要 Windows 符号链接权限的测试。覆盖新增轻量路径、出表、探针、定位头、四臂训练恢复、数据契约、原批量评测与评测恢复；`git diff --check` 通过。没有把 CPU/合成数据测试当成正式 GPU 实验验收。

## 已经实现

1. `formal_v2.paper_suite`：独立的 plan / run / export 入口。配置包含两张主表和四张附表，可按表选择依赖，已完成阶段校验后复用，分别输出 CSV、Markdown、LaTeX 和缺失清单。计划与导出只用标准库，不触发训练环境校验或 LLM 审批。
2. 定位头参数选择修复：每个 loss 对应实际参与该次 forward 的参数；额外检查最后一次更新。原实现用更新前 loss 选择更新后参数，可能保存过冲状态。这个修复会改变新评测结果，历史结果必须保留原版本标识。
3. 单探针断点恢复：保存模型、Adam 状态、已完成步数、输入/初始化/训练预算标识。每 100 步及最后一步落盘；主 streaming evaluator 已接入。恢复输入不同、非有限模型/优化器状态会报错。CPU 完整训练与中断恢复已验证逐参数相同。
4. 源域 pilot 的诊断记录补充：采样记录保留编码器上的 endpoint、实际加权 alignment/response 梯度范数及三组余弦，避免把未加权均值当作实际训练影响。此诊断仅在传入 loss_trace 时启用。
5. Windows 原子写入兼容、只读 evaluation-status 解耦训练环境检查。POSIX 仍保留目录 fsync，Windows 保留文件 fsync 和原子替换。

## 现在实际能出什么

| 表 | 范围 | 当前历史快照可导出内容 |
| --- | --- | --- |
| main1 | 四臂的 alignment / null / response 指标 | 缺完整评测汇总；明确显示 MISSING |
| main2 | 四臂消融 + Wi-GATr / PMNet，2 城市 × 4 个 k，Median/P90 | 四臂 64 个真实数值；外部对比 32 个指标格缺失 |
| s1 | 因子交互、成本、资源匹配控制 | 训练成本和部分原始统计已有；资源匹配与完整区间缺失 |
| s2 | 捷径、响应变体、打乱配对控制 | 缺完整对应输出 |
| s3 | 外部基线六种地图条件诊断 | 缺完整对应输出；与主表 few-shot 对比口径不同 |
| s4 | k=0 风险/校准，joint / d_only / u_only | 缺完整对应输出 |

生成六个文件并不等于完成六组实验。`coverage.json` 中 COMPLETE 只表示该数据格/阶段的覆盖状态，不表示优于基线或支持论文主张。科学 FAIL 可以出表，程序异常不能充当完成。

## 使用

在仓库的 `code/CSI-PAIRS-v2.0-server` 目录执行：

```bash
# 无需 NPZ / GPU：导出仓库中明确指定的历史快照
python -B -m formal_v2.paper_suite export --archive --output ../../docs/results/paper-tables-20260909

# 查看服务器既有运行目录还差什么；只生成计划，不启动训练
python -B -m formal_v2.paper_suite plan --run-root /path/to/run --dataset /path/to/formal.npz --output /path/to/paper-report

# 调度当前已实现的阶段，最后导出六张表和缺失清单
python -B -m formal_v2.paper_suite run --run-root /path/to/run --dataset /path/to/formal.npz --output /path/to/paper-report

# 可按需单独选择表，依赖会自动展开
python -B -m formal_v2.paper_suite run --run-root /path/to/run --dataset /path/to/formal.npz --output /path/to/paper-report --tables main1 s2
```

输出不完整时入口返回 2。这不是把缺失填零或宣告实验失败。`--archive` 不能与当前运行目录混用，当前实验的空格不会用历史数字自动补齐。

配置在 `formal_v2/configs/paper_suite.json`。当前 `run` 是对已有阶段后端的调度：需要原运行目录的 data_verification/qualification、正式数据和对应运行环境；没有消除后端所有认证门槛，也没有重新实现数据生成/资格训练。Wi-GATr 仍需其专用环境。

## 仍未完成，不能称为全量一键闭环

- **外部方法的主表后端**：现有两个官方代码适配器产出六条件 power-to-location 诊断，没有与四臂相同的城市 support/query、三个种子、十次 label draw 的 k=0/8/32/128 评测。配置中的 `baseline_localization` 明确 BLOCKED，不是一个能执行的空壳命令。导出器的 comparison_protocol 校验只是接口约束，尚无自动生产该协议和完整基线结果的后端。
- **评测内存/时长**：旧入口保留 17 模型和全数据更新；新核心入口已经拆分、磁盘存储并采用小批次协议，具体见追加精简一节。正式数据上的资源改善幅度仍未验证。
- **自动研究配方**：旧 source pilot 的固定硬件/128 步限制尚未全面改造。当前增加了原因诊断，没有凭猜测将正式损失替换成宣称更强的配方。
- **服务器验收**：尚未测量实际显存、峰值 RAM、吞吐、完整数据上的恢复一致性或新指标；本机 CUDA 测试跳过。

## 为什么 Full 目前没有优势

确定的现象：最新快照的 Full 在两个城市、四个预算的全部八格都差于 Response-only。主表中的数据如实保留。

| 证据 | 可以判断什么 | 不能据此判断什么 |
| --- | --- | --- |
| 三个 Full 的末步 fixed active alignment loss 为 0.099987–0.099992，接近两个 0.05 hinge 的无区分参考值 0.1 | Alignment 的有效辨别信号值得优先检查 | 不能仅凭 loss 证明特征完全坍塌或认定唯一原因 |
| 记录的 alignment 梯度范数 0.00394–0.00582，response 为 0.394–0.463 | 旧记录提示两分支影响很不平衡 | 这些是有限采样摘要，不是全程实际加权梯度比 |
| Full 的 response log variance 约 -1.19，对应 exp(-s) 约 3.3 | 联合训练实际响应权重发生变化，应与 static_sum 做源域对照 | 不能预判关闭不确定性权重必然更好 |
| Alignment/Response 梯度余弦均值约 0.0004–0.0100 | 不能从该均值支持“强正协同” | 也不能据此证明普遍负向梯度冲突 |
| 定位头 loss/state 对应错误 | 确定的实现缺陷，已修复并测试 | 修复对 Full 与其他臂的相对收益未知 |

后续方法验证需要在 `source_method_selection` 上比较固定/效果相关 margin 与 static_sum/uncertainty weighting，再用多个源域种子确认；记录每臂结果、有效梯度和成本。选定配方后冻结，再运行目标域完整评测。不能用目标测试集反复挑选到 Full 排第一，也不能通过删除强基线让 Full 获胜。

历史导出见 `docs/results/paper-tables-20260909/main2.md`。上面的工程修复和证据链是本次已完成内容；“方法达到 SOTA”和“所有表完整跑通”仍没有完成证据。
