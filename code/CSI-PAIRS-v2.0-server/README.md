# CSI-PAIRS experiments

从服务器已有 NPZ 开始，一条命令运行两张主表和四张附表：

```bash
bash formal_v2/scripts/run_paper.sh /path/to/formal.npz /path/to/new-run --device cuda:0
```

脚本使用服务器现有 PyTorch、NumPy、SciPy 环境；首次使用时自动安装 Wi-GATr 的独立官方环境。无需审批许可、安装回执、环境闭包认证或干净 Git 工作区。
如果环境已经准备好，也可以直接运行：

```bash
python -B -m formal_v2.paper_run --dataset /path/to/formal.npz --output /path/to/new-run --device cuda:0
```

| 表 | 自动运行的内容 |
| --- | --- |
| Main 1 | 四分支的对齐 AUROC、null 误判率、统一响应探针与原生响应指标 |
| Main 2 | 四分支 + Wi-GATr/PMNet 适配，Boston/Seattle，k=0/8/32/128，中位误差和 P90 |
| S1 | 因子交互置信区间、计算量、五个资源对照、旧损失配方对照及配对打乱定位结果 |
| S2 | 六个捷径探针、响应输入消融、copy/no-action/action-swap、源域全局配对打乱 |
| S3 | SigMap、Wi-GATr、PMNet、WiSER、RFIR 的六种地图条件诊断 |
| S4 | 四分支 k=0 的 joint/d-only/u-only 风险校准，全查询与源域支持范围分别报告 |

结果写入 OUTPUT/tables/{main1,main2,s1,s2,s3,s4}.{csv,md,tex}，coverage.json 列出缺项。负结果照常出表，不因没有超过基线而阻止完成。
重复同一命令即可恢复；定位按模型保存、诊断按单元保存、探针每 25 次更新保存、主模型与基线保存优化器状态。PMNet 按 epoch 恢复。单个地图方法失败后，其余地图方法仍继续。

仅查看精确任务命令（不需要数据或 GPU）：

```bash
python -B -m formal_v2.paper_run --dataset /server/formal.npz --output /server/new-run --plan
```

可用 --stages training probes controls maps risk tables 选择阶段；例如 --stages tables 仅重新出表。--wigatr-python 可以指定已有的 Wi-GATr Python。各模块 paper_core、paper_controls、paper_maps、paper_risk 也支持独立运行。
新协议请使用新输出目录，数据或配置发生变化时不要覆盖旧实验。运行记录保留版本和输入摘要，不扫描或认证整个 Python 安装环境。旧调度器已移除；paper_suite 只负责历史结果的只读导出。

当前默认配置为 formal_v2/configs/paper_formal.json 和 paper_all_probes.json。新配方在源域 pilot 上按共享编码器梯度归一化辅助损失，Full 使用固定加权和；定位头的特征与坐标尺度只在源域拟合。旧配方作为独立对照保留。探针和教师读出层的步数均为小批次更新次数。

**解释对比结果时必须保留的区别：** Wi-GATr/PMNet 原本预测功率，主表使用“原生逆定位估计 + 共同位置头”的适配；S3 报告原生逆定位，不能拿 S3 冒充少样本主表。SigMap/WiSER/RFIR 保留现有受控实现标注。所有主表方法使用同一支持样本和查询点，目标测试标签不参与选择。
S1 资源匹配报告实际参数/FLOP 比例，concat 使用共同位置头而不再叠加额外瓶颈。S2 是源域全局配对打乱，不冒称旧的场景内分桶打乱。S4 按每个 bank 实际可用类别采样并输出 mixture.csv；不存在的 gray/null 等类别不会被伪造，无法估计的共同支持指标写 N/A 和原因。

本地 CPU 测试验证软件功能；正式 NPZ、大规模 GPU 运行和性能提升尚需服务器实测。代码修复与“已经达到 SOTA”是不同结论。历史 checkpoint 和负结果未删除；合成测试结果只写 fixture_tables，永久标记 FORBIDDEN。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest formal_v2/tests -q
```
