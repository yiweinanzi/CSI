# CSI-PAIRS 训练、选择、校准和测试数据说明

## 先说结论

这份数据不能把所有 CSI 行混在一起以后随机切成“80% 训练、20% 测试”。

原因很简单：同一个场景 bank 里有 4 个只差一个地图干预的 sibling worlds，
每个 world 又共享 256 个接收机位置，并有 3 次观测重复。如果把这些近亲样本随机拆开，
测试集就会出现训练集场景的近重复，最后的精度会虚高。

因此，ZIP 只保存一份数据字节，训练、选择、校准和测试通过
`scene_roles`、`position_roles` 和本目录中的
`FORMAL_MAIN_SPLIT_LEDGER.json` 进行选择。不要复制出多份 NPZ 再手工改名。

## 两个源城市如何分

Austin 和 Chicago 各有 7 个 bank。两个城市使用相同的 bank 编号规则：

| bank 编号 | 数据角色 | 零基础解释 |
| ---: | --- | --- |
| 00 | `source_encoder_train` | 真正训练 Stage-0 teacher、主编码器和四个方法臂 |
| 01 | `source_method_selection` | 调参数、选结构、定阈值和 checkpoint 规则，不能回头训练最终模型 |
| 02 | `source_probe_train` | 在已经冻结的表示上训练统一 probe |
| 03 | `source_probe_selection` | 选择 probe 类型和正则，不能继续拟合 probe |
| 04 | `source_calibration_fit` | 拟合温度、风险校准器和支持范围模型 |
| 05 | `source_calibration_selection` | 只选择校准方案，不能把这部分再并回拟合集 |
| 06 | `source_final_unseen_bank` | 源城市内部最终测试，只允许看一次 |

所以每个角色都有两个独立 bank：Austin 一个，Chicago 一个。训练数据、调参数据、
校准数据和最终测试数据互不重叠。

## 两个目标城市如何分

Boston 和 Seattle 各有 8 个 target bank，每个 bank 有 256 个唯一接收机位置：

- 偶数位置索引，共 128 个，属于 `support_pool`，只可用于预注册的少样本适配。
- 奇数位置索引，共 128 个，属于冻结的 `query`，只可用于最终目标城市评测。
- 标签预算 `k=0/8/32/128` 指唯一接收机位置数，不是 CSI 行数。
- 一个 support 位置只能提供一条预注册的正确或自然观测；不能把它的 4 个 sibling
  worlds 算成 4 个标签。
- 某个位置一旦被选入 support，它的全部 sibling worlds、edges、masks 和 queries 都要从
  最终定位、风险、CGS 和 interaction 统计中排除。
- `query` 绝不能用于梯度更新、早停、归一化、阈值、超参数或 checkpoint 选择。

## 另外两个城市如何用

Denver 和 Miami 各有 2 个 `external_validation` bank。它们用于提前登记的外部城市或
压力测试，不能参与训练、归一化、校准、阈值选择、模型选择或 checkpoint 选择。

## 外部公开数据如何分

DeepMIMO、UrbanMIMOMap、RadioMapSeer 和 DeepSense 不属于上述 34-bank 主数据，
也不能与主数据按行拼接。它们各自遵守 `docs/SPLIT_POLICY.json`：

- DeepMIMO 按完整 scenario 分组。
- UrbanMIMOMap 当前只有 map 0，按 transmitter/configuration 组分开。
- RadioMapSeer 按完整 city map 分组，Loc/ToA 等派生视图必须跟原 map 在同一侧。
- DeepSense 按 acquisition sequence 和 location 分组，不能随机拆相邻帧。

WWM 原始数据不在包内。DeepSense Scenario 8/33 只是明确标注的非等价公开替代，
不能写成“使用了原始 WWM”。

## 当前仍不能做什么

这份 split 账本解决的是“谁可以读取哪部分数据”，不等于论文实验已经获准开始。
CPU 34-bank 数据仍是 `CANDIDATE_NOT_CLAIM`，A100 Sionna 数据仍是
`fixture=true`、`scientific_use=FORBIDDEN`。正式训练还需要在预批准的 Linux/A100
运行时重新生成并通过全部资格门和 LLM-as-judge 批准。
