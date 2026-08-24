<!-- encoding: UTF-8 -->

# CSI-PAIRS 冻结版研究方案（v6.0，零基础阅读稿）

## Paired Alignment and Intervention-Response Supervision

版本：v6.0  
日期：2026-08-04  
修订：2026-08-24，对齐已冻结代码合同 `PRIMARY_ROUTE_CONTRACT`、C9 四道风险门、Scene-ID 定位位移方向、Alignment 主分数 `random_75`。主路由只看物理 CSI；teacher 只做资格审计，不再参与 active/null 判定。  
目标：ICLR 2027  
方法名：**CSI-PAIRS（Paired Alignment and Intervention-Response Supervision）**

这是一份实验开始前的研究方案，不是实验结果。除非明确引用了已有论文，文中写到“预期”“应当”“如果成立”的地方，都不能在投稿时改写成已经证实的结论。最终论文中的每一条结果性表述，都要通过第 14 节的 Claim-Evidence Gate，也就是“证据够不够支撑这句话”的检查。

### 读前先认识几个词

| 词 | 这里是什么意思 |
|---|---|
| ICLR | International Conference on Learning Representations，机器学习领域的重要国际学术会议。ICLR 2027 是这项工作的投稿目标，不代表论文已经达到录用标准。 |
| CSI | 信道状态信息。可以先把它理解成无线信号从基站传到接收机以后留下的一张详细“指纹”，里面包含不同天线、不同频率上的复数测量值。 |
| 地图条件模型 | 模型同时看 CSI 和地图，再做预测或定位。 |
| RT | Ray Tracing，射线追踪。给定地图、基站、接收机位置和无线配置后，用物理规则计算信号如何反射、绕射和传播。 |
| 世界状态 | 同一个场景经过某种明确编辑后的地图版本，比如某栋建筑存在或不存在。 |
| 干预边 | 两个只相差一项地图编辑的世界之间的连接。比如只把一栋建筑从“无”改成“有”。 |
| scene-level world bank，场景级世界库 | 为同一个场景预先生成一组地图版本，并让这组版本服务许多接收机位置。这样模型不能靠某张地图的独特纹理直接查出位置 ID。 |
| active、gray、null | 一次地图编辑后，主路由只看物理 CSI：变化足够大叫 active，只落在噪声范围内叫 null，其余叫 gray。冻结 teacher 另做资格审计，不参与这三类判定。三类样本承担的监督不同。 |
| CGS | 用完全相同、预算固定的小探针检查冻结表征能否分清正确 CSI-地图配对与同一条 active 边上的备选配对。它只在这类配对范围内衡量可读出的几何一致性，不能当成全局地图真伪分数。 |
| Endpoint 与四臂 | Endpoint 是“同一个世界、正确地图、零编辑”的共同基础预测任务。四臂是在相同数据和骨干下，分别训练 Endpoint、Alignment-only、Response-only 和两种监督都开启的 CSI-PAIRS full。 |
| encoder，编码器 | 把原始 CSI、地图等输入压成一组供后续任务使用的数字特征。本文把最终保留的编码器记作 $F$。 |
| representation，表征 | 编码器产出的数字特征。可以把它理解成模型对当前 CSI 与地图的内部摘要。 |
| latent，隐藏特征 | teacher 从 CSI 中提取出的中间特征。它不是人工可读的物理量，所以后文还会单独检查它是否保留了真实 CSI 信息。 |
| predictor，预测器 | 根据编码器输出、地图编辑动作和查询位置，预测目标世界的隐藏特征与物理 CSI patch。本文记作 $P$。 |
| teacher，教师模型 | 只看 CSI、提前训练好并冻结的参考模型。冻结表示后续训练不能再改它。本文记作 $T_{\mathrm{CSI}}$。 |
| patch | 从完整 CSI 中切出的一小块，通常对应一部分天线和子载波。 |
| probe，探针 | 冻结编码器后，再训练一个结构简单、训练预算固定的小模型，用来检查表征里是否含有某类信息。 |
| calibration，校准 | 把模型分数转换成有明确概率含义的数值，并检查“报 20% 风险”的样本是否大约真的有 20% 失败。 |
| arm，实验臂 | 一套受控实验中的一个模型版本。不同实验臂只改变指定的监督开关，其余条件按方案保持一致。 |

---

## 0. 先把研究问题和结论边界说清楚

### 0.1 我们到底担心什么

一个同时看 CSI 和地图的模型，在“CSI 与地图正确配对”的数据上表现很好，仍可能没有真正理解局部几何。它也许只从地图里认出了“这是哪座城市”“这是哪类街区”或“这里大概是视距传播”，却不知道移走一栋建筑以后，当前信道应该保持不变，还是应该朝哪个方向变化、变化多少。

### 0.2 CSI-PAIRS 怎么处理这个问题

CSI-PAIRS 在同一条可检查的场景干预边上做两件事：

- **paired alignment** 检查当前 CSI 更支持边的哪一侧，也就是哪张地图相对更能解释它。
- **intervention-response** 根据 RT 在编辑后重新计算出的目标，学习 CSI 应该朝什么方向变化，以及变化多少。

两项监督共同更新同一个地图条件编码器 $F$。真正做下游定位时，不再调用动作编码器、预测器或教师模型，只留下这个共同训练过的 $F$。

### 0.3 论文要检验的中心命题

正确配对训练可能让地图条件 CSI 模型对局部几何仍不敏感。CSI-PAIRS 用可跨位置重复使用的场景级配对干预，同时训练相对地图-CSI 对齐与带方向的 RT 目标响应。最后要看的是：联合训练能否在跨城市定位和限定配对场景内的风险校准上，同时胜过只做 alignment 和只做 response 的版本。

原论文可以使用下面这句英文作为中心命题，但它仍是一条等待实验检验的命题：

> Correctly matched training can leave map-conditioned CSI models insensitive to local geometry. CSI-PAIRS uses reusable scene-level paired interventions to supervise both relative map-CSI alignment and signed target response, and tests whether their shared representation improves calibrated unseen-city localization beyond either supervision alone.

### 0.4 两条监督为什么要放在同一套机制里

```text
alignment：当前 CSI 更支持干预边的哪一侧？
response：沿这条边做完编辑，CSI 应朝哪里移动，移动多少？
```

alignment 只告诉模型一个相对顺序。它能把不匹配的一侧推远，却没有告诉模型应该靠近哪个真实目标。response 给出方向和幅度，却不自动保证模型能判断当前 CSI 与地图是否对齐。两条监督解决的问题不同，而且必须作用在同一条干预边、同一个物理单元和同一个最终保留的编码器上。

“作用不同”还不能证明它们放在一起会互相帮助。这个结论只能靠四臂实验、交互效应和机械拼接对照来验证。

### 0.5 从 V4.1 和 V5 各保留什么

| 内容 | v6 的决定 | 为什么 |
|---|---|---|
| 错图现象 | 保留为论文开场 | 它能直接暴露“正确配对表现好”和“局部几何响应正确”之间的缺口。 |
| CGS 与统一探针 | 保留，但只在 active paired edit 的适用范围内解释 | 任意一张错图都不能自动当成绝对不相容。 |
| 显式 compatibility | 保留为相对 alignment | 只比较同一个物理单元、同一条 active 干预边的两侧。 |
| 每个位置单独生成 K 个世界 | 删除 | 每个位置都有独一无二的地图变体时，模型可能直接从地图认出位置编号。 |
| scene-level world bank | 采用 | 一组地图和编辑要在许多位置上重复使用，从而切断“只看地图就查到位置”的捷径。 |
| 所有非对角配对都做 ranking | 删除 | 不同地图可能在别处产生相似 CSI，gray 和 null 也不该被强行推远。 |
| RT 重追踪 target-response | 升为核心分支 | 它直接规定编辑后应靠近的真实模拟目标，以及变化方向和幅度。 |
| 风险校准 | 保留，但收紧适用范围 | $q_{\mathrm{comp}}$ 与 $p_{\mathrm{fail}}$ 分开拟合，只用源城市数据，拟合后冻结再迁移。 |
| 跨城市 zero-shot 和 few-shot 定位 | 保留为最终价值检验 | 方法不能只在自己构造的 compatibility 或 response 小任务上证明自己。zero-shot 指目标城市不给定位标签，few-shot 指只给很少的位置标签。 |
| path-incidence 与 LoS/NLoS | 保留 | 用来检查收益是否真的出现在编辑影响传播路径的位置。LoS 是视距，NLoS 是非视距。 |
| $I(x)$ 可辨识性分析 | 降为 P1 | 它仍有价值，但不能抢走严格四臂主实验的 P0 资源。P0 是必须先完成，P1 是主线成立后再做。 |

### 0.6 哪些话不能说过头

- $(H_{i}, M_{j})$ 不能叫“全局绝对负样本”“物理不可能组合”或“错误世界”。全文统一叫 **paired active alternative**，也就是同一条确认生效的干预边上的另一侧候选。
- 场景级超立方体世界库不能叫独立同分布的“可交换世界”。准确说法是“节点均衡、边对称的场景级世界库”。
- $q_{\mathrm{comp}}$ 只表示指定配对干预范围内的相对 compatibility，不能解释成任意地图在整个地图空间里为真的概率。
- 随便换成另一座城市的地图只能算压力测试。校准数据若没有包含这类扰动，就不能把输出分数解释成经过校准的概率。
- 没有真实受控干预或独立物理引擎证据时，只能说结果与当前模拟器一致，也就是 `simulator-consistent`，不能说模型理解了真实世界因果关系。
- ranking、Transformer、MAE、JEPA、对比学习和多任务加权都不是本文单独声称的新算法。MAE 是“遮住一部分输入，再让模型重建”的学习方法；JEPA 是“在内部特征空间预测目标特征”的方法。
- 本方法不能叫 world model，也不能声称模型会做长时间 rollout 或成为通用信道模拟器。
- 最终保留的 $F$ 如果没通过地图信息保留审计，只能说一次性的 response 或 alignment 预测头学会了任务，不能说下游表征获得了 geometry grounding，也就是对几何的可靠扎根。
- full 如果没有同时胜过 alignment-only 和 response-only，就删除“联合方法优于单分支”的主张。交互效应如果没有通过预先设定的门槛，只能写“统一的 paired multi-task supervision”，不能写 synergy，也就是不能声称两条监督产生了额外协同。

---

## 1. 论文叙事从“换错地图”开始

### 1.1 先看现有模型到底怎么用地图

现有地图条件模型通常只在正确地图上训练和测试。这样的成绩回答不了一个问题：局部几何变化后，模型会不会相应改变输出。模型也可能只用了城市风格、场景类别或粗略的 LoS 先验。

至少选择两个能正常运行的地图条件模型，统一测试下面六种输入：

| 条件 | 给模型的地图 | 这项测试在看什么 |
|---|---|---|
| A：correct | 真正生成当前 CSI 的地图 | 常规基准。 |
| B：paired-active alternative | 位置、无线配置和其他物理条件不变，只换成相邻世界；主路由按物理 CSI 判定这次编辑生效。teacher 是否敏感只作审计，不改条件标签 | 检查模型是否对局部、确实生效的几何变化有反事实响应。 |
| C：paired-null alternative | 地图确实编辑过，但当前信道差异和教师目标差异都落在噪声范围内 | 检查模型会不会只要看到编辑就误报变化。 |
| D：wrong-city map | 换成另一座城市的地图，并按以基站为中心的米制坐标规则对齐 | 只做压力测试，不能自动解释成有概率意义的 compatibility。 |
| E：geometry-destroyed | 保留地图的低阶统计特征，但破坏具体空间结构 | 区分模型用的是几何结构，还是颜色比例、占用率之类的粗统计。 |
| F：empty map | 输入全零地图，或训练前冻结的平均地图提示 | 检查模型是否至少用到了某种地图信息。 |

如果 B 和 A 的结果几乎一样，而 F 明显变差，只能得到一个有限结论：模型用了地图提供的粗条件，但对局部且确实生效的编辑不够敏感。单凭这个现象，还不能断言模型把地图当成了 scene ID，也就是场景身份证。

### 1.2 “地图只是场景身份证”需要额外证据

这仍然是一条有条件的机制假说，需要在源城市内部同时通过两项实验：

1. 在严格留出的新位置上，只给一个 scene-ID prompt，表现能否追平给正确地图的 prompt。
2. 把地图换成城市 $b$，与把 prompt 换成 `ID_b` 时，**定位预测的位移方向**是否一致。主门使用二维定位坐标的方向余弦，不要求再比内部隐状态。

目标城市的 $k=0$ 实验没有任何目标定位标签，不能临时学习新的 scene embedding。它只能使用训练前冻结的 `UNK` 表征，或源城市 scene embedding 的均值。`UNK` 是 unknown 的缩写，可以理解成“遇到训练时没见过的城市时统一使用的未知占位符”。如果两种替换造成的定位位移方向不一致，论文只保留“模型对局部几何不敏感”，删除 scene-ID 机制解释。

### 1.3 从现象走到方法

```text
正确配对时表现好
    不能证明
模型真的按局部几何变化来使用地图
        ↓
用 active 和 null 错图测试，加上 CGS 做诊断
        ↓
alignment 指出干预边上哪一侧相对一致
response 给出 RT 重追踪后的变化方向和幅度
        ↓
训练结束只留下两者共同更新的编码器 F
        ↓
用严格四臂检查跨城定位、风险校准和路径机制
```

论文仍按 V4.1 的顺序展开：现象、指标、方法、结果、分析。V5 不会变成第二套并列方法，它只负责补上 V4.1 缺少的 response 监督。

### 1.4 CGS 在本文里具体测什么

CGS 可以先理解成“表征中的几何一致性可读程度”。做法是冻结编码器，在相同隐藏接收状态和同一条确认生效的编辑上，用所有方法完全相同、训练预算固定的简单探针，检查它能否分清真正生成 CSI 的那一侧和 paired active alternative。

- active 四元组进入 AUROC。AUROC 越高，表示探针越能把两侧排对；它只衡量排序能力，不等于全局概率。
- gray 只报告现象，不给强标签。
- null 只用来检查模型是否过度判别，也就是明明编辑没有产生可分辨影响，模型却硬说两侧不同。
- endpoint、alignment-only、response-only、full 和外部地图条件基线使用同一个探针与同一预算。

$q_{\mathrm{comp}}$ 是 active 干预边内部、离线计算的相对概率。$p_{\mathrm{fail}}$ 是在一个提前冻结的地图比较候选集合下，定位会失败的概率。两者必须分别校准，而且只能用源城市的 calibration 数据拟合。训练用了什么损失，不能反过来改变指标定义。

---

## 2. 每个训练样本必须来自可检查的配对物理单元

### 2.1 什么叫“同一个物理单元”

先固定五件事：独立场景块 $b$、世界状态 $u$、用户位置 $x$、无线配置 $c$，以及除地图编辑外保持不变的物理条件 $\xi_{\mathrm{phys}}$。无线配置 $c$ 包括基站姿态、载频、阵列和天线配置。$\xi_{\mathrm{phys}}$ 包括其他必须固定的传播条件。

不加观测噪声时，RT 输出写成：

$$
H^\star(b,u,x)
=
\operatorname{RT}\!\left(M(b,u),x,c,\xi_{\mathrm{phys}}\right).
$$

$M(b,u)$ 是场景块 $b$ 的第 $u$ 个地图状态，$H^\star$ 是 RT 直接计算出的干净 CSI，星号表示它还没有叠加接收机噪声。把地图、用户位置、基站配置和其他物理条件一起交给 RT，就得到这个世界下的 CSI。比较干预边 $u \leftrightarrow v$ 时，$x$、$c$ 和 $\xi_{\mathrm{phys}}$ 必须一模一样，只能切换一项已记录的地图编辑。

需要研究接收机噪声或测量扰动时，第 $r$ 次观测写成：

$$
H_{\mathrm{observed}}(b,u,x,r)
=
\operatorname{Observe}\!\left(
H^\star(b,u,x),\eta(b,u,r)
\right).
$$

`Observe` 表示观测过程，$\eta (b,u,r)$ 是世界 $u$ 在第 $r$ 次测量中独立抽到的观测噪声。配对世界可以共享真实传播环境 $\xi_{\mathrm{phys}}$，却不能复制同一段观测噪声。$u$ 和 $v$ 的噪声要独立抽样，否则模型可能利用被复制的噪声，得到虚假的 response 可预测性。

P0 主结果使用干净 RT 目标 $H^\star$，把地图编辑造成的变化与测量噪声分开。独立噪声下的重复测量作为稳健性实验，也用来估计 no-edit 或 identity noise floor，也就是“不编辑时自然会波动多大”。真实测量若没有干净目标，就使用提前规定的多次重复测量均值和不确定性区间，不能把一次测量中的噪声差直接当成地图效果。除专门的噪声实验外，后文把 $H^\star$ 简写为 $H$；route、教师目标、alignment 分数和 response 目标默认都由干净 RT 配对计算。

大规模生成数据前还要过 RT 资格门。材质、散射、天线和噪声参数只能在独立校准集上拟合，拟合后冻结。随后在一个既不参与参数拟合，也不参与世界编辑和最终测试的验证集上，提前规定并检查 path loss、delay spread、angular spread 和可见路径数。它们分别是路径损耗、时延扩展、角度扩展和能被 RT 找到的传播路径数量。资格门没通过，仍可把工作写成 simulator-defined 方法研究，但全文不能称这个模拟器已经校准。

### 2.2 一个场景块只建一个可重复使用的 world bank

每个独立场景块选择 $d \ge 2$ 个互相兼容、可以撤销的基本编辑。每项编辑都预先规定两个物理可行的状态，比如某建筑存在或不存在。所有 0/1 组合构成世界库：

$$
\mathcal W_b
=
\left\{
M(b,u)\;\middle|\;u\in\{0,1\}^{d}
\right\},
\qquad
K=2^d\ge 4.
$$

$W_{b}$ 是场景块 $b$ 的世界库，$u$ 是一串长度为 $d$ 的 0/1，$K$ 是这个库里的地图总数。两个可开关的编辑会产生 4 张地图，三个会产生 8 张。每个 bit 的 0 和 1 具体代表哪一侧，要在每个 bank 内随机置换，避免固定编码泄露信息。

每个节点必须根据完整状态说明单独做 canonical render，也就是用同一套确定规则从头渲染。不能先改一张图、保存，再继续改成下一张，因为文件痕迹和编辑顺序可能暴露世界来源。训练只使用两个状态中恰好有一位不同的边：

$$
\mathcal E_b
=
\left\{
(u,v)\;\middle|\;d_{\mathrm H}(u,v)=1
\right\}.
$$

$E_{b}$ 是允许训练的边集合，“只差 1 个 bit”也叫 Hamming distance 为 1。每条边只对应一个明确的基本编辑，模型不会同时面对多个变化却不知道是哪一个起作用。

同一个地图状态 $M(b,u)$ 必须服务于这个场景块中的许多用户位置。先固定 world bank 和所有世界共用的位置集，再对节点、边和位置做完整交叉生成。不能为了某一种动作另挑更容易产生明显变化的位置。

自然原图 $M_{nat}$ 进入四臂共同的 endpoint 或 natural 数据分支，但 alignment 不能总把它当作唯一且身份特殊的“干净锚点”。如果必须把自然原图放进 bank，就要通过 randomized-anchor 和 edit-status XOR 单元测试。XOR 测试看的是：模型能否只靠“这张图像不像原图”和“两种输入的编辑状态”猜出标签。任何能稳定猜出的构造都不能进入 headline alignment。

这组世界不能称为独立同分布的可交换变体。我们实际依赖四条可检查的条件：源数据中的节点使用次数均衡；每条无向边的两个方向出现次数相同；active alignment 四元组在边内精确匹配 CSI 和地图的边际分布；同一个 bank 跨多个位置复用，让地图纹理无法唯一指向某个位置。

P0 原则上一个独立场景块只生成一个超立方体 bank。如果多个 bank 使用完全相同的基础地图，它们必须进入同一个数据 split 和同一个统计 cluster。重复的基础节点要去重，或按连接次数的倒数加权，不能把这些 bank 当成相互独立的新场景来虚增样本量。

$d=2$ 只有在两个基本编辑属于同一编辑族、变化量级相当，而且 geometry-matched wrong-action candidate coverage 通过资格门时，才能用于 headline。这个 coverage 检查错误动作候选是否在几何上足够可比。若不满足，就要在大规模 RT 之前改成 $d \ge 3$，不能看完目标城市结果以后再放松匹配标准。

### 2.3 所有世界共用同一批合法位置

$$
X_b
=
\bigcap_{u\in\{0,1\}^{d}}
\operatorname{FreeSpace}\!\left(M(b,u)\right).
$$

$X_{b}$ 是场景块 $b$ 的共同用户位置集。一个位置只有在所有地图版本中都能合法放置用户设备，才可进入数据，新增建筑不能盖住任何共同位置。

采样顺序固定如下：

1. 均匀选择独立场景块或 bank。
2. 在该 bank 内均匀选择一条无向边。
3. 从它固定的 $X_{b}$ 中均匀选择位置 $x$。
4. 固定无线配置 $c$ 和物理条件 $\xi_{\mathrm{phys}}$，组成完整物理单元。
5. alignment 同时构造边的两侧；response 对 $u \to v$ 和 $v \to u$ 两个方向等频采样。

active、gray、null 只能在采样完成后判定。允许在渲染前按几何角色预注册共同位置，例如反射面可见或分支覆盖；这不是按渲染后的 CSI 效果或 route 标签再筛位置。禁止在看到 CSI、teacher 或 route 之后再删位置，否则动作类型会泄露用户位置，也就是产生不该存在的 $P(x | action)$ 关系。

同一个源样本在训练中组成一个 branch bundle。不同分支必须复用完全相同的 $H_{u}$、源地图、无线配置 $c$、输入遮挡、输出查询和数值增强，只改变动作与目标邻居。每个 source 至少连接两个 Hamming-1 分支，每个分支都要以大于 0 的概率被抽到。有两个结构分支还不够，还要报告“两个目标确实可区分”的 active branching coverage。如果一个 source 永远只对应一个 target，模型可以完全忽略 action，动作信息价值实验也就不成立。

### 2.4 哪些地图编辑可以进入 P0

P0 只使用主模型的地图表示能够看见、RT 也能在两个方向上重新追踪的编辑。

| 编辑族 | 例子 | 必须满足的条件 |
|---|---|---|
| occupancy | 加入或移除建筑、大型临时结构 | 不能覆盖共同位置集。 |
| height | 建筑升高或降低 | 高度必须落在提前规定的物理范围。 |
| material | 在混凝土、玻璃、金属类别之间切换 | 材质电磁参数通过 RT 资格门后冻结。 |
| compound state | 一个世界同时包含两个基本状态 | 每一条训练边仍然只能切换一个基本编辑。 |

窗口、细面片等主模型地图分辨率看不见的编辑不能进入 P0。模型输入如果看不见一项变化，损失函数就不能强迫它预测这项变化。

### 2.5 地图和动作给模型看什么

源地图使用以基站为中心、范围固定、分辨率固定的多通道网格。至少包含 occupancy、height、material 和基站位姿或配置。$G(M)$ 只负责按确定规则把地图栅格化并排成 token；可训练的地图编码器属于 $F$，不能藏在 $G$ 里。

动作编码写成：

$$
e_{uv}=E_{\phi}\!\left(\Delta_{uv}\right).
$$

$\Delta_{uv}$ 是从世界 $u$ 走到世界 $v$ 的带方向编辑网格，$E_\phi$ 是动作编码器，$e_{uv}$ 是它输出的动作特征。每个网格明确记录 add 或 remove、高度增加或降低、材质从哪一类变到哪一类。材质类别编号不能直接做减法，因为“金属编号 3 减玻璃编号 1 等于 2”没有物理意义。

动作输入不能包含 world ID、edit ID、文件名、生成序号、用户位置 $x$、path-incidence、effect bucket、物理差异 $\delta_{\mathrm{phys}}$ 或任何由目标 CSI 推出的量，也不能把完整目标地图再输入一遍。

地图范围、token 数量、token 顺序和 padding 都不能随用户位置改变。主模型禁止使用以用户为中心的裁图、按真实用户连线挑 map token、用户轨迹和 receiver ID。

### 2.6 四种泛化问题必须分开切数据

| 测试轴 | 怎么分组 | 它回答什么 |
|---|---|---|
| unseen-position | 同一位置的全部 sibling worlds、边、遮挡和查询必须一起进入训练侧或测试侧；测试位置的 CSI 不能进入 Stage 0 或归一化统计 | 模型能否处理同一个 bank 中没见过的位置。 |
| unseen-edit/operator | 整个基本编辑家族或实例，以及它专属的世界和 CSI，从 Stage 0 开始全部留出 | 模型能否迁移到没训练过的编辑规律。 |
| unseen-scene-bank | 整个场景块、基础地图、bank、全部位置和 CSI，从 Stage 0 开始全部留出 | 模型是否摆脱了具体世界或 bank 的指纹。 |
| unseen-city | 整座城市的场景块、bank、位置、CSI 和统计量全部留出 | 这是 headline，检查严格跨城泛化。 |

headline 至少要有两个源训练城市和两个相互独立的目标城市，每座城市都要有多个独立场景块。不同训练随机种子只是同一数据上的重复运行，不能冒充独立城市或独立场景。

当 $d \ge 3$ 时，可以单独留出“某个已经见过的基本编辑，放在新的 bit 上下文中形成的边”，作为 compositional operator 诊断。由于这些节点可能通过其他边出现在训练中，这项实验不能冒充独立数据泛化。任何 split 都不能随机拆散 sibling worlds。

### 2.7 一份固定的数据权限账本

除独立的 RT 资格门数据外，所有源城市 scene bank 都按最高层场景块一次性分到下面互不重叠的数据块。一个场景块的 sibling worlds、位置、边、方向、遮挡和查询必须一起移动，不能跨块。

| 数据块 | 只允许做什么 | 明确不能做什么 |
|---|---|---|
| `source-encoder-train` | 训练 Stage-0 teacher 和 readout、四臂的 `F/P`、源域位置头；计算训练用归一化常数 | 不能训练 probe、拟合 calibrator 或参与阈值与模型选择。 |
| `source-method-selection` | 承担所有 source pilot、validation 和 model-selection；冻结 route 的实际效果阈值与噪声阈值、margin 和 $\phi$、loss 权重、mask/query bank、网络结构、资格门和 checkpoint 规则 | 不能更新最终 encoder，不能训练统一 probe，也不能拟合 calibrator。 |
| `source-probe-train` | 在冻结表征上训练统一的 compatibility probe 和 response probe | 不能选择 probe 类型，不能拟合 calibrator。 |
| `source-probe-selection` | 只选择 probe family、正则强度和 probe checkpoint | 不能训练 calibrator，也不能报告最终指标。 |
| `source-calibration-fit` | 拟合 $T_{A}$、风险校准器、特征标准化参数和适用范围估计器 | 不能选择 probe，也不能看目标城市结果。 |
| `source-calibration-selection` | 只在提前列好的候选中选择 calibration 的正则或模型类型，并冻结适用范围分位数 | 不能把数据回流去训练 encoder 或 probe，也不能充当校准最终测试集。 |
| `source-final-unseen-bank` | 只做一次源城市外部审计和最终统计检验 | 不能用于任何训练、阈值、选模或校准。 |

文中若写“source pilot 后冻结”，只允许读取 `source-method-selection`。若写“source train”或“outer-train”，都只能指 `source-encoder-train`。校准候选在 `source-calibration-fit` 上拟合，在 `source-calibration-selection` 上选择。选中后直接冻结，不能再把 selection 数据加回去重拟合。

目标城市还要按 receiver position 分成 adaptation-support 和 frozen query。$k$ 永远表示 $k$ 个不同的接收机位置，不是样本条数。每个 support 位置在 headline adaptation 中只能提供一条提前规定的正确或自然 CSI-地图观测，不能把这个位置的 K 个 sibling worlds 展开，把 $k=8$ 偷换成 `8K` 条有标签样本。

support 位置的所有 sibling worlds、边、遮挡和查询，都要整组从最终定位、风险、CGS 和 interaction 统计中删除。若想用目标城市 sibling worlds 做增广，只能另列为 privileged diagnostic，同时报告不同位置标签数和带标签样本总数。$k=0$ 没有 support 数据；$k>0$ 时，任何 query 位置都不能进入梯度更新、早停、归一化或超参数选择。

---

## 3. 先判断一项编辑到底算 active、gray 还是 null

### 3.1 用一个只看 CSI 的冻结 teacher 提供参考目标

Stage 0 只读取 `source-encoder-train` 中的 CSI，训练 $T_{\mathrm{CSI}}$。它沿用 CSI-MAE 的基本设置：把 CSI 的实部和虚部作为两个通道，按“天线 × 子载波”切 patch，使用二维位置编码，随机遮住 75% patch，并使用不对称的 MAE 编码器和解码器。训练完成后立刻冻结，生成 target 时始终处在评测模式。四个实验臂共享完全相同的 teacher，也共享完全相同的 CSI encoder 初始化。

对完整且没有遮挡的 $H_{k}$，第 $q$ 个隐藏目标写成：

$$
z_k(q)
=
\operatorname{stopgrad}
\!\left(
T_{\mathrm{CSI}}(H_k)[q]
\right).
$$

`T_CSI(H_k)[q]` 是 teacher 在解码器之前产生的第 $q$ 个 patch 特征，`stopgrad` 表示这条目标分支不接收梯度，后续训练无法修改 teacher。$z_{k}(q)$ 还要使用 `source-encoder-train` 的统计量做固定归一化。teacher 只负责把目标 CSI 转成稳定的参考答案，不能跟着四臂一起改变答案。

teacher 的输入函数中不能出现地图、位置、动作或轨迹。实现时必须做单元测试：保持 $H$ 不变，只替换代码里对 $M$、$x$ 和动作 $a$ 的引用，teacher target 在规定的数值精度内必须完全不变。

还要单独训练并冻结逐 patch 的 CSI readout $D_{\mathrm{CSI}}$：

$$
y_k(q)=\Psi_q(H_k),
\qquad
\widetilde y_k(q)=D_{\mathrm{CSI}}\!\left(z_k(q)\right).
$$

$\Psi_q$ 从原始 CSI 中取出第 $q$ 个实部和虚部 patch，并用源训练常数归一化，结果记作 $y_{k}(q)$。$D_{\mathrm{CSI}}$ 尝试从隐藏特征 $z_{k}(q)$ 还原物理 patch，输出记作 `ỹ_k(q)`。隐藏特征本身不能直接读懂，readout 用来审计其中是否还保留了可还原的物理 CSI 信息。

所有差值目标必须使用 pair-consistent phase gauge，也就是配对世界共用同一个相位参考。可以把相位参考理解成两份波形共同使用的“时钟零点”：比较前必须对着同一只钟，不能把两份波形各自拨到最有利的位置。确定性 RT 使用与世界状态无关的共同时间和相位基准，不能分别给 $H_{u}$ 和 $H_{v}$ 做各自最优的相位旋转。分别旋转会把预处理差异误写成 response 的方向。

实测系统如果无法提供共同相位参考，P0 的物理目标和指标就改用不受整体相位影响的表示，比如幅度、协方差或 delay-angle power。delay-angle power 是按信号到达时延和到达角度分格统计能量，不依赖整份 CSI 共同转过多少相位。具体选择只能在 `source-method-selection` 上确定并冻结。

$D_{\mathrm{CSI}}$ 只做审计，不参与 response 的训练目标。预测器会独立输出物理 patch。若直接把 $D_{\mathrm{CSI}}$ 的还原结果放进物理损失，readout 自身的误差可能与隐藏目标冲突。如果 readout 在 `source-method-selection` 上无法可靠还原 patch，就不能再用隐藏目标支撑“模型学到了物理响应”这句话，必须退回固定的原始或物理目标。

### 3.2 alignment 看完整信道，response 看单个 patch

alignment 是完整信道层面的配对任务。先定义两个距离：

$$
\begin{aligned}
\delta^{\mathrm A}_{uv}
&=
d_{\mathrm{phys}}
\!\left(
\Phi(H_u),\Phi(H_v)
\right),\\
\gamma^{\mathrm A}_{uv}
&=
d_z
\!\left(
T_{\mathrm{CSI}}(H_u),
T_{\mathrm{CSI}}(H_v)
\right).
\end{aligned}
$$

$\Phi(H)$ 是实验前冻结的完整信道物理表示，P0 同时使用按源训练常数归一化的复数 CSI 和 delay-angle power 摘要。$d_{\mathrm{phys}}$ 是归一化后的均方根距离，$d_{z}$ 是 teacher token 的归一化均方根距离，上标 $A$ 表示这组量服务 alignment。$\delta ^{\mathrm A}$ 从物理 CSI 看整条信道变了多少，$\gamma ^{\mathrm A}$ 从冻结 teacher 的内部特征看整条信道变了多少。归一化常数只能由 `source-encoder-train` 计算，并在所有实验臂和数据 split 中固定。

response 是 patch 层面的任务。对第 $q$ 个输出 patch，距离写成：

$$
\begin{aligned}
\delta^{\mathrm R}_{uv}(q)
&=
d_{\mathrm{phys}}
\!\left(
\Phi_q(H_u),\Phi_q(H_v)
\right),\\
\gamma^{\mathrm R}_{uv}(q)
&=
d_z
\!\left(
z_u(q),z_v(q)
\right).
\end{aligned}
$$

$\Phi_q$ 默认就是前面的 $\Psi_q$，只看第 $q$ 个物理 patch，上标 $R$ 表示这组量服务 response。一项编辑可能只影响部分天线或频率区域，完整信道明显变化，不代表每个 patch 都明显变化，因此 response 的路由必须逐 patch 判断。

response 的 $d_{\mathrm{phys}}$ 和 $d_{z}$ 要分别使用后面 dead-zone 所用的 physical norm 与 latent RMS norm。RMS 是均方根：把各维差值先平方、取平均，再开平方，用一个数概括整体差异大小。这样“是否超过阈值”和“预测是否超过允许误差”使用相同单位。

主路由只使用物理距离，不再把 teacher 距离 $\gamma$ 写进判定：

$$
\operatorname{Route}(\delta)
=
\begin{cases}
\mathrm{null},
& \delta\le\epsilon_0,\\
\mathrm{active},
& \delta\ge\epsilon_1,\\
\mathrm{gray},
& \text{其他情况}.
\end{cases}
$$

$\delta$ 是物理空间差异；$\epsilon_{0}$ 是噪声上界，$\epsilon_{1}$ 是有实际意义的变化下界。teacher 距离 $\gamma$ 仍按同样阈值另算一层，只用于资格审计：报告物理 active 中 teacher 也敏感的比例，以及物理 null 中 teacher 也判 null 的比例。这两项过低时，隐藏特征指标不能代替真实物理差异，但不能回头改写已经冻结的主路由标签。

alignment 和 response 使用两套独立物理阈值：

$$
\begin{aligned}
0&\le\epsilon^{\mathrm A}_0<\epsilon^{\mathrm A}_1,
&
0&\le\tau^{\mathrm A}_0<\tau^{\mathrm A}_1,\\
0&\le\epsilon^{\mathrm R}_0<\epsilon^{\mathrm R}_1,
&
0&\le\tau^{\mathrm R}_0<\tau^{\mathrm R}_1,\\[3pt]
r^{\mathrm A}(uv)
&=
\operatorname{Route}
\!\left(
\delta^{\mathrm A}_{uv},
\epsilon^{\mathrm A}_0,
\epsilon^{\mathrm A}_1
\right),\\
r^{\mathrm R}(uv,q)
&=
\operatorname{Route}
\!\left(
\delta^{\mathrm R}_{uv}(q),
\epsilon^{\mathrm R}_0,
\epsilon^{\mathrm R}_1
\right).
\end{aligned}
$$

$r^{\mathrm A}$ 是整条信道的 alignment 路由，$r^{\mathrm R}$ 是第 $q$ 个 patch 的 response 路由。不能拿完整信道的阈值直接判断一个小 patch 是否 active。两组低阈值来自无编辑重复仿真或重复测量的 noise floor，两组高阈值来自 `source-method-selection` 上提前规定的 practical-effect floor。所有阈值都要在查看目标城市结果前冻结。$\tau$ 只服务 teacher 审计层，不进入 $r^{\mathrm A}$ 或 $r^{\mathrm R}$。

route 还要满足三条实现规则：

- 换边的方向不能改变 route，必须有 $r^{\mathrm A}(uv) = r^{\mathrm A}(vu)$，以及 $r^{\mathrm R}(uv,q) = r^{\mathrm R}(vu,q)$。
- route 只用于决定损失、审计采样和分层评测，不能作为模型输入。
- 每个数据 split 都要报告 active、gray、null 的原始数量、比例和场景级覆盖，不能先删掉非 active 样本，再宣称 active 覆盖率是 100%。

还要报告两项一致性：物理上 active 的样本中，teacher 有反应的比例；物理上 null 的样本中，teacher 也判 null 的比例。如果这两项太低，说明 teacher 目标与物理任务不一致，隐藏特征指标不能代替真实物理差异。

### 3.3 三类 route 分别承担什么监督

**active** 表示物理 CSI 差异达到生效下界。只有这类样本可以要求“真正生成当前 CSI 的地图分数，比干预边另一侧更高”。

**gray** 表示物理差异落在噪声上界和生效下界之间。它仍然有 RT 重追踪得到的目标 $H_{v}$，所以可以学习具体 target，但不能强制说两侧一定不同或一定相同。

**null** 表示物理差异落在噪声范围内。它不是负样本，也不能要求预测变化精确等于 0，因为真实的 $z_{u}$ 和 $z_{v}$ 仍可能存在阈值以内的小差异。本文使用 dead-zone penalty，也就是设置一圈允许误差，只惩罚超出这圈范围的虚假变化。

### 3.4 输入遮挡和输出查询必须隔离

$m$ 是从源 CSI 中遮住的 patch 集合，$V_{m}(H_{u})$ 是预测器唯一能看到的源 CSI。$q$ 是要求模型预测的目标 patch。常规训练和 target-free 评测都强制 $q$ 属于 $m$，也就是答案所在的 patch 不能提前露给模型。同一个四元组和同一个 branch bundle 必须逐样本复用完全相同的 $(m,q)$，不能让正确地图、alternative 地图或不同动作使用不同遮挡。

Stage 1 的 source mask bank 提前固定三种模式：随机遮挡 75%、按天线块遮挡 50%、按子载波块遮挡 50%。后两种是本研究为下游外推增加的设置，不能误写成 CSI-MAE 原论文的预训练配置。Alignment 主分数只平均 `random_75` 条目；天线块和子载波块遮挡只作诊断，不进入主 alignment 分数。评测使用的 mask/query bank 在训练前冻结，而且要覆盖全部 patch。

`full-H no-x` 只作为可预测性上界。它表示给模型完整 $H$，但仍不给目标世界 CSI。这个版本必须另行训练 $m$ 为空集的受控模型，不能把一个按 masked 输入训练的模型在测试时突然换成完整 CSI，再把这种分布外结果当成可辨识性证据。

无论哪种设置，目标世界 $H_{v}$ 都不能进入源预测器。它只允许用于生成冻结 target、物理标签、effect route 和最终评分。

---

## 4. CSI-PAIRS 的模型与训练目标

### 4.1 两条监督共同训练同一个编码器

源世界 $u$ 在遮挡 $m$ 下的内部状态写成：

$$
C_u(m)
=
F_{\theta}
\!\left(
V_m(H_u),G(M_u),c,m
\right).
$$

$F_{\theta}$ 是参数为 $\theta$ 的共享编码器。它接收可见的源 CSI、按确定规则处理后的源地图、无线配置和遮挡位置，输出状态 $C_{u}(m)$。alignment 和 response 都从这个状态出发，也都必须把训练梯度传回同一个 $F_{\theta}$。$F_{\theta}$ 内部包含 CSI encoder、map encoder 和轻量 cross-attention fusion。cross-attention 可以理解成让 CSI 特征带着问题去查看地图特征，再把相关地图信息合进当前状态；“轻量”表示这部分只承担融合，不另外做一套庞大的主干网络。

下游定位只运行：

$$
r=F_{\theta}(H,M,c),
\qquad
\widehat x=g(r).
$$

$r$ 是最终保留的表征，$g$ 是位置头，$\widehat x$ 是预测位置。定位时不再读取动作编码器、预测器、teacher 或 readout。位置头 $g$ 只用源城市的位置标签训练。目标城市 $k=0$ 时整套网络冻结；$k>0$ 时，所有实验臂必须使用同样数量的位置标签、同一组可更新参数和相同步数。

为了报告模型自身的 compatibility 和 response 能力，可以在单独的诊断路径保留冻结的 $P$、$T_{\mathrm{CSI}}$ 和 $D_{\mathrm{CSI}}$，但它们的输出不能传给 $g$。因此，下游定位若有改善，只能来自共享 $F_{\theta}$ 的表征变化，不能来自测试时额外调用反事实预测器。

### 4.2 四个实验臂都有同一个 endpoint 任务

$\rho_{z}$ 表示按隐藏特征维度平均的归一化均方误差，$\rho_{y}$ 表示按物理 patch 维度平均的归一化均方误差。$\rho_{y}$ 必须使用第 3.1 节的共同相位规则；若没有共同相位，则使用提前冻结的相位不变表示。所有归一化常数只来自 `source-encoder-train`，并与 route 和 dead-zone 使用的 norm 保持一致。

本文的 **endpoint** 永远只表示“同一个世界、正确配对、零动作”的预测任务。它不指下游定位损失，也不能在不同表里换成别的含义。

预测器从同一个状态同时输出隐藏目标和物理 patch：

$$
\begin{aligned}
\left(
\widehat z_{u\to u}(m,q),
\widehat y_{u\to u}(m,q)
\right)
&=
P_{\psi}\!\left(C_u(m),0,q\right),\\
\ell_B(u,m,q)
&=
\rho_z
\!\left(
\widehat z_{u\to u}(m,q),z_u(q)
\right)\\
&\quad
+\lambda_{By}\,
\rho_y
\!\left(
\widehat y_{u\to u}(m,q),y_u(q)
\right).
\end{aligned}
$$

$P_{\psi}$ 是参数为 $\psi$ 的共享预测器，$action=0$ 表示不编辑地图，带帽子的 $\widehat z$ 和 `ŷ` 是预测值。`ℓ_B` 是一个样本的 endpoint 损失，$\lambda_{By}$ 控制物理 patch 误差在这个损失中的权重。这项任务给模型当前世界的部分 CSI 和正确地图，让它在“不做编辑”的条件下还原当前世界的隐藏特征和物理 CSI patch。

整个数据集的 endpoint 损失采用 bank-macro 平均：

$$
\mathcal L_B
=
\mathbb E_{\substack{
b,\;u,\;x,\;(m,q)\\
\text{按预注册的 bank-macro 规则均匀采样}
}}
\left[
\ell_B(u,m,q)
\right].
$$

$\mathcal L_{B}$ 是数据集级 endpoint 目标。bank-macro 表示先让每个 bank 权重相同，再在 bank 内部取平均，因此位置多、节点多的 bank 不能因为样本条数大，就在总损失里自动占更高权重。相同 K-world 数据中的每个 $(H_{u},M_{u})$ 都作为一条独立的正确配对使用，但 endpoint 看不到边的 route、带方向动作或跨世界目标。自然原图以提前固定的混合权重进入四臂完全相同的 endpoint 数据层，不改变 bank-macro 原则。如果实现中还保留共享 CSI-MAE 重建项，它必须以相同权重出现在四臂中，并计入 $\mathcal L_{B}$。

### 4.3 Paired alignment：比较哪张地图更能解释当前 CSI

alignment 不额外训练一个互不相关的二分类分数头。它直接看“给定候选地图、动作设为 0 时，预测当前 CSI 会错多少”。错误越小，说明候选地图越能解释当前 CSI。

先冻结一个覆盖完整信道的小型 mask/query 集合 $B_{align}$。一组 mask/query 下的预测误差与最终分数为：

$$
\begin{aligned}
\ell_0(H,M;m,q)
&=
\rho_z
\!\left(
P_z\!\left(
F_{\theta}(V_m(H),G(M),c,m),0,q
\right),
T_{\mathrm{CSI}}(H)[q]
\right)\\
&\quad+
\lambda_{sy}\,
\rho_y
\!\left(
P_y\!\left(
F_{\theta}(V_m(H),G(M),c,m),0,q
\right),
\Psi_q(H)
\right),\\[3pt]
\operatorname{score}(H,M,c)
&=
-\frac{1}{|\mathcal B_{\mathrm{align}}|}
\sum_{(m,q)\in\mathcal B_{\mathrm{align}}}
\ell_0(H,M;m,q).
\end{aligned}
$$

$P_{z}$ 和 $P_{y}$ 是同一个预测器的隐藏特征输出与物理 patch 输出，$\lambda_{sy}$ 是物理误差权重。`score` 把所有固定查询上的平均预测误差取负数：一张地图如果能让预测器更准确地解释当前 CSI，误差会更小，取负后分数就会更高。

每次 alignment 更新时，四元组中的四种候选配对都要使用同一个 $B_{align}$ 和同一组数值增强。必须先算完整 mask/query bank 的平均分，再把平均分放进 hinge ranking。hinge ranking 可以理解成“正确地图至少要领先规定的分差”：已经领先够多就不再惩罚，没达到时只惩罚还差多少。这里考的是整套 mask/query 的总分，所以要先把一整套小题算成平均分，再判断是否领先；随机抽一道题先判输赢再平均，一般会变成另一个训练目标。native alignment 主结果沿用同一个分数定义；另用完全不参与训练的 `B_audit_hold` 检查结果是否依赖某一种 mask/query。alignment 与 response 因而共用同一个预测机制：alignment 用零动作，response 用非零动作。

对同一个 $(b,x,c,\xi_{\mathrm{phys}})$ 中的 active 无向边 `{u,v}`，双向 ranking 损失写成：

$$
\begin{aligned}
\ell_{\mathrm{rank}}(uv)
&=
\left[
m_{uv}
-\operatorname{score}(H_u,M_u)
+\operatorname{score}(H_u,M_v)
\right]_+\\
&\quad+
\left[
m_{uv}
-\operatorname{score}(H_v,M_v)
+\operatorname{score}(H_v,M_u)
\right]_+,\\
m_{uv}
&=
m_0\,\phi\!\left(\delta^{\mathrm A}_{uv}\right).
\end{aligned}
$$

第一行要求 $H_{u}$ 与 $M_{u}$ 的分数至少比 $H_{u}$ 与 $M_{v}$ 高 $m_{uv}$，第二行把方向反过来，要求 $H_{v}$ 更支持 $M_{v}$。$m_{0}$ 是基础间隔，$\phi$ 随物理差异单调增加，但有截断上界。在一条物理主路由判定为 active 的边上，两侧各自的 CSI 应相对更支持自己的生成地图。$\phi$ 的形状和上界只能在 `source-method-selection` 上冻结。主结果还要同时报告固定 margin 版本。$m_{uv}$ 只能叫 effect-aware margin，因为模型分数不是物理单位，不能称“物理标定 margin”。这个排序只在当前配对边内有效，不表示另一张地图在未知位置绝对不可能产生相似 CSI。

null 边不做 ranking。为了避免模型只要看到地图编辑就制造分数差，引入 dead zone：

$$
\operatorname{deadzone}(a,\kappa)
=
\left[\lvert a\rvert-\kappa\right]_+^2.
$$

$a$ 是两个分数的差，$\kappa$ 是允许的小差异半径。只要分数差没有超过 $\kappa$，损失就是 0；超过以后，才惩罚超出的部分。

alignment 使用的 $\kappa_{s}$ 不能根据任何最终实验臂反推，确定流程固定如下：

1. 用固定 endpoint 训练日程在 `source-encoder-train` 上训练一个四臂共享的 endpoint-pilot checkpoint，它只负责定标。
2. 把这个 checkpoint 切到评测模式，在 `source-method-selection` 的 canonical no-op pairs 上，用同一个 $B_{align}$ 和同一种分数归一化计算预测能量的绝对差。
3. no-op pair 的两侧使用同一个 $H$ 和语义相同的地图，只把一侧经过 empty-edit canonical rerender。噪声稳健性版本使用相互独立的观测重复。
4. $\kappa_{s}$ 取这些绝对差提前规定的 $1 - \alpha_{s}$ 分位数，$\alpha_{s}$ 必须在读取这块数据前固定。

四臂共用同一个 $\kappa_{s}$。最终 checkpoint 和目标城市数据都不能修改它。还要用提前 route 为 physical-null 的边检查这个容差是否覆盖 noise floor。若覆盖不了，只能修复相位基准或噪声模型，或者判定数据资格门失败，不能看完各臂结果后再放宽 $\kappa_{s}$。

一条 null 边的 alignment 损失为：

$$
\begin{aligned}
\ell_{\mathrm{nullA}}(uv)
&=
\operatorname{deadzone}
\!\left(
\operatorname{score}(H_u,M_u)
-\operatorname{score}(H_u,M_v),
\kappa_s
\right)\\
&\quad+
\operatorname{deadzone}
\!\left(
\operatorname{score}(H_v,M_v)
-\operatorname{score}(H_v,M_u),
\kappa_s
\right).
\end{aligned}
$$

两个 dead-zone 项分别检查边的两个方向。null 边允许存在噪声范围内的小分数差，但不能让模型对两侧产生明显不同的 compatibility 判断。

数据集级 alignment 目标按 route 分开做 bank-macro 平均：

$$
\mathcal L_A
=
\operatorname{MacroAvg}_{\mathrm{active}}
\!\left[
\ell_{\mathrm{rank}}
\right]
+
\lambda_{nA}\,
\operatorname{MacroAvg}_{\mathrm{null}}
\!\left[
\ell_{\mathrm{nullA}}
\right].
$$

$\lambda_{nA}$ 是 null alignment 项的权重。active 和 null 各自只在通过预注册场景级 coverage gate、至少包含该 route 的训练 bank 中计算。先让 eligible bank 权重相同，再在每个 bank 内均匀选符合 route 的无向边和位置。这样 bank 大小以及 active 或 null 所占比例，不会暗中改变 $\lambda_{nA}$ 的实际作用。gray 的 alignment 损失为 0，也不进入这两个条件均值。

### 4.4 Intervention-response：直接预测编辑后的 RT 目标

response 从同一个状态 $C_{u}(m)$ 和同一个预测器出发，只把零动作换成带方向动作：

$$
\left(
\widehat z_{u\to v}(m,q),
\widehat y_{u\to v}(m,q)
\right)
=
P_{\psi}
\!\left(
C_u(m),e_{uv},q
\right).
$$

$e_{uv}$ 告诉预测器从世界 $u$ 走到相邻世界 $v$ 做了什么，输出是目标世界第 $q$ 个隐藏特征和物理 patch 的预测。模型只看源世界，不看目标 CSI，再根据地图编辑预测目标世界会出现什么 CSI。

真实变化与预测变化分别为：

$$
\begin{aligned}
\Delta z_{uv}(q)
&=
z_v(q)-z_u(q),\\
\widehat{\Delta z}_{uv}(m,q)
&=
\widehat z_{u\to v}(m,q)
-\widehat z_{u\to u}(m,q),\\[3pt]
\Delta y_{uv}(q)
&=
y_v(q)-y_u(q),\\
\widehat{\Delta y}_{uv}(m,q)
&=
\widehat y_{u\to v}(m,q)
-\widehat y_{u\to u}(m,q).
\end{aligned}
$$

$\Delta z$ 和 $\Delta y$ 是 RT 配对产生的真实变化，带帽子的 $\Delta \widehat z$ 和 $\Delta \widehat y$ 是模型预测的“非零动作输出减零动作输出”。这里比较的不是两个彼此无关的绝对预测，而是同一个源状态下“做编辑”和“不做编辑”会相差多少。

所有有方向的边，不论 active、gray 还是 null，都回归 RT 重追踪得到的目标：

$$
\begin{aligned}
\ell_{\mathrm{target}}(uv,m,q)
&=
\rho_z
\!\left(
\widehat z_{u\to v}(m,q),z_v(q)
\right),\\
\ell_{\mathrm{phys}}(uv,m,q)
&=
\rho_y
\!\left(
\widehat y_{u\to v}(m,q),y_v(q)
\right).
\end{aligned}
$$

第一项让预测在 teacher 隐藏空间里靠近目标世界，第二项让预测在可解释的物理 CSI patch 上靠近目标世界。gray 和 null 仍然有一个真实模拟目标，不能因为它们不参加强 ranking 就丢掉这条信息。

active patch 还要额外学对变化方向和幅度：

$$
\begin{aligned}
\ell_{\Delta}(uv,m,q)
&=
\rho_z
\!\left(
\widehat{\Delta z}_{uv}(m,q),
\Delta z_{uv}(q)
\right)\\
&\quad+
\lambda_{\Delta y}\,
\rho_y
\!\left(
\widehat{\Delta y}_{uv}(m,q),
\Delta y_{uv}(q)
\right).
\end{aligned}
$$

$\lambda _\Delta y$ 控制物理变化误差的权重。直接预测对目标还不够，active 样本还要明确约束“相对源世界，变化方向和幅度是否正确”。

null patch 使用允许小变化的 dead-zone：

$$
\begin{aligned}
\ell_{\mathrm{nullR}}(uv,m,q)
&=
\left[
\operatorname{RMS}
\!\left(
\widehat{\Delta z}_{uv}(m,q)
\right)
-\kappa_z
\right]_+^2\\
&\quad+
\lambda_{0y}
\left[
\operatorname{physical\_norm}
\!\left(
\widehat{\Delta y}_{uv}(m,q)
\right)
-\kappa_y
\right]_+^2.
\end{aligned}
$$

$\kappa_{z}$ 和 $\kappa_{y}$ 是隐藏空间与物理空间允许的小变化半径，$\lambda_{0y}$ 是物理 null 项权重，`RMS` 是前文解释过的均方根大小。null 不要求模型输出严格为 0，只禁止模型凭空预测超过噪声范围的大变化。

$\kappa_{z}$ 和 $\kappa_{y}$ 分别由 identity 或 no-edit 条件下，隐藏特征和物理 patch 的源数据噪声范围确定并冻结。默认设置为：

$$
\kappa_z=\tau^{\mathrm R}_0,
\qquad
\kappa_y=\epsilon^{\mathrm R}_0.
$$

$\kappa_{z}$ 和 $\kappa_{y}$ 直接复用 response 路由的两个低阈值，让判为 null 的标准与 null 允许的变化范围使用同一把尺子。实现中必须做单元测试，确保所有 $r^{\mathrm R}(uv,q)=null$ 的真实 RT 变化 $\Delta z_{uv}$ 和 $\Delta y_{uv}$ 都落在对应 dead zone 内。如果 route 使用了复合物理表示 $\Phi$，就要把 $\kappa$ 提前调整为能够覆盖源数据 null RT 差值的预注册上界。否则精确 target 会要求模型学习一个小变化，null penalty 却同时惩罚它，损失定义互相打架。gray 只承担 target 和 physical patch 回归，不承担强 delta，也不承担零变化假设。

把每条无向边的两个方向各放一次，得到 directed edge set。数据集级 response 目标为：

$$
\begin{aligned}
\mathcal L_R
&=
\operatorname{MacroAvg}_{\mathrm{all\ directed\ edges}}
\!\left[
\ell_{\mathrm{target}}
+\lambda_{\mathrm{phys}}\ell_{\mathrm{phys}}
\right]\\
&\quad+
\lambda_{\Delta}\,
\operatorname{MacroAvg}_{\mathrm{active\ patches}}
\!\left[
\ell_{\Delta}
\right]\\
&\quad+
\lambda_{nR}\,
\operatorname{MacroAvg}_{\mathrm{null\ patches}}
\!\left[
\ell_{\mathrm{nullR}}
\right].
\end{aligned}
$$

$\lambda_{\mathrm{phys}}$、$\lambda _\Delta$ 和 $\lambda_{nR}$ 分别控制物理目标、active 变化量和 null 过度反应三部分的权重。第一项不按 route 筛选，所以 active、gray、null 都学习 RT 目标；后两项分别只在 active 和 null 内部归一化。每一项都先让符合条件的 bank 权重相同，再在 bank 内均匀采有向边、位置和 mask/query。采样器已经让 $u \to v$ 与 $v \to u$ 等频，公式里不能再重复加一项“对称损失”。

如果某个训练 split 没有足够的 active 或 null scene-bank coverage，数据资格门直接失败。程序不能把空的条件均值悄悄写成 0。论文必须随实验报告每个条件均值实际使用的 bank、edge、position 和 patch 数量。

### 4.5 四臂中每项监督的剂量必须一样

alignment 和 response 的原始损失大小可能相差很多，先用一个不进入最终结果的短 pilot 把它们归一化。做法是在共享初始化上，用 `source-method-selection` 中固定的短 batch 序列 $P$ 训练临时模型，对每个 batch 的 scene-bank-macro 目标做无偏分层估计。“分层”表示先按方案规定的 bank 和 route 层级抽样，不让样本多的层自然占满 batch；“无偏”表示把这种抽样重复很多次后，估计的平均值会回到完整目标本身，不会系统性偏向某类 bank 或 route：

$$
\begin{aligned}
c_A
&=
\max
\left(
\operatorname{Mean}_{p\in\mathcal P}
\widehat{\mathcal L}_{A,p},
\epsilon
\right),\\
c_R
&=
\max
\left(
\operatorname{Mean}_{p\in\mathcal P}
\widehat{\mathcal L}_{R,p},
\epsilon
\right),\\[3pt]
\widetilde{\mathcal L}_A
&=
\frac{\mathcal L_A}{c_A},
&
\widetilde{\mathcal L}_R
&=
\frac{\mathcal L_R}{c_R}.
\end{aligned}
$$

$c_{A}$ 和 $c_{R}$ 是两条辅助损失的固定尺度，$\epsilon$ 是防止除以 0 的数值下限，波浪号表示已经归一化。先把两条损失换到可比较的数量级，再设权重，避免某条损失只因为数值大就支配训练。

临时 pilot checkpoint 不能进入最终训练或评测。batch 序列 $P$、共享初始化和 $\epsilon$ 在四臂中完全相同。$c_{A}$ 和 $c_{R}$ 只计算一次，随后冻结，不能为不同实验臂分别重算。

固定 $\lambda_{A}$ 和 $\lambda_{R}$ 后，四臂目标为：

$$
\begin{aligned}
\mathcal L_{00}
&=
\mathcal L_B,
&&\text{Endpoint-only},\\
\mathcal L_{10}
&=
\mathcal L_B+\lambda_A\widetilde{\mathcal L}_A,
&&\text{Alignment-only},\\
\mathcal L_{01}
&=
\mathcal L_B+\lambda_R\widetilde{\mathcal L}_R,
&&\text{Response-only},\\
\mathcal L_{11}
&=
\mathcal L_B
+\lambda_A\widetilde{\mathcal L}_A
+\lambda_R\widetilde{\mathcal L}_R,
&&\text{CSI-PAIRS full}.
\end{aligned}
$$

`00` 表示两条辅助监督都关闭，`10` 只打开 alignment，`01` 只打开 response，`11` 两条都打开。同一项监督在单分支和 full 中必须保持同样权重。difference-in-differences 是“四项相减”的受控比较：用 full 的结果减去两个单分支的结果，再加回共同的 Endpoint，检查两项监督一起开启后是否还有额外收益。若 full 里的 alignment 或 response 权重另有变化，四臂就不再是标准的 `2 × 2` 交互检验，无法判断差值来自监督组合，还是来自某项监督在 full 中用得更多。

所有损失权重只能使用 `source-method-selection` 选择。实验还要报告两条分支传回 $F$ 的梯度范数、训练 FLOPs 和有效更新次数。FLOPs 是浮点运算量，用来衡量训练计算开销。如果 full 的 FLOPs 超过单分支预注册容差，就要增加相同 FLOPs 的单分支训练控制。把两条辅助损失强行塞进固定总梯度预算的 convex-mixture 版本，只能作为稳健性实验，不能替代主交互检验。

### 4.6 方法的新意是整套可审计协议

本文不把某一个损失公式包装成新算法。能够成立的新意由七件相互关联的设计组成：

1. 同一个场景 world bank 跨多个位置复用，避免地图变体变成位置 ID。
2. 同一个 $(x,c,\xi_{\mathrm{phys}})$ 下，干预边两侧都由冻结 RT 重新追踪，每个监督目标都能逐样本审计。
3. 只有物理主路由判定为 active 的边才进入相对 alignment。teacher 敏感与否只进入资格审计，不改这条边是否受 alignment 监督。
4. gray 和 null 不被强行标成负样本，null 还明确限制模型过度反应。
5. response 直接靠近由目标世界 $H_{v}$ 得到的 RT target，不再只把另一侧推远。
6. 两条监督共同更新同一个 state encoder，训练结束也只保留这个 encoder 做定位。
7. 最终主张由严格四臂、交互效应、机械拼接对照和跨城市下游结果共同决定。

## 5. 怎么判断模型真的理解了 CSI 和地图是否配套

这一节有两个容易混淆的概率。$q_{\mathrm{comp}}$ 回答“给定这两张候选地图，哪一张更像当前 CSI 的生成地图”；$p_{\mathrm{fail}}$ 回答“定位结果出错的概率有多大”。前者是配对比较，后者是风险预测，二者不能互相替代。

### 5.1 CGS：用固定难度的小测验检查表征

先解释几个词。$H$ 表示 CSI，也就是无线信号经过环境后留下的信道信息。$M$ 表示地图。$u$ 和 $v$ 是同一个场景世界库中的两个世界，它们只相差一次受控地图编辑。`active` 表示这次编辑确实改变了物理信道，而且冻结的 CSI 教师模型也能看出变化。

对每条留出的 active 边，我们固定场景、位置、无线配置和物理扰动，然后构造四个组合：

$$
\begin{aligned}
\mathcal Q_{\mathrm{matched}}
&=
\left\{
(H_u,M_u),(H_v,M_v)
\right\},\\
\mathcal Q_{\mathrm{alternative}}
&=
\left\{
(H_u,M_v),(H_v,M_u)
\right\}.
\end{aligned}
$$

第一行中，CSI 与生成它的地图来自同一个世界。第二行只交换地图。它们拥有完全相同的两份 CSI 和两张地图，所以只看 CSI 或只看地图都无法完成分类。

本文把 CGS 固定定义为“固定探针预算下的 compatibility 可解码性”。可以把探针理解为一场统一规格的小测验：冻结每个模型最终保留的编码器 $F_{\theta}$，不允许它继续学习，再给所有模型相同的训练数据、相同大小的线性探针和两层 MLP 探针、相同训练步数、相同参数量，以及相同的源域选模预算。MLP 是多层感知机，也就是一种小型全连接网络。

主指标是 active 样本上区分“正确配对”和“成对备选”的 AUROC。AUROC 越高，说明保留下来的表征越容易支持这项判断。还要按物理变化幅度 $\delta_{\mathrm{phys}}$ 分成四档分别报告，避免结果只由大变化样本撑起来。

CGS 衡量的是“在固定探针预算下能否读出来”，不能写成表征中存在多少“固有信息”。gray 样本是物理主路由未达到 active 也未落入 null 的模糊样本，它不进入 AUROC，只报告两种配对的分数差分布。null 样本是物理差异落在噪声范围内的编辑，也不能被当成负样本。null 单独报告三项：正确配对与交换地图后的绝对分数差；超过源域噪声容差的错误不兼容率；与 identity 或 no-edit 重复实验噪声底的等效检验。这里要证明“足够接近零”，不能用“统计上不显著”冒充“等于零”。

不看地图的 CSI-only 模型只作为协议负对照。它本来就没有地图输入，不能把它与看地图的模型并排后宣称“没有学到地图”。

如果能公平运行至少 3 个 map-conditioned 模型，则保留 V4.1 的跨模型散点图：横轴是统一协议下的 active CGS，纵轴是未见城市的定位中位误差。图只检查两者趋势是否一致，不能凭少量模型的相关性写成因果关系。CSI-only 负对照用不同图例单独标出。

### 5.2 模型自带分数和统一探针必须分表

四个实验臂都带有同一个 zero-action predictor，也就是“地图不变时预测当前世界”的预测器，所以都能算原生 compatibility energy $s(H,M)$。分数越高，表示模型认为这份 CSI 与这张地图越匹配。不过，只有 alignment-only 和 full 使用了 alignment 损失 $\mathcal L_{A}$，直接训练过 active 样本的分数排序。

四臂的原生 active AUROC、校准误差和可靠性曲线单独放一张表，不混进跨模型的统一探针主图。还要报告原生 energy 与统一探针排序之间的 Spearman 相关系数，以及二者 AUROC 的差值。Spearman 只看排序是否一致，这项检查能发现“模型自带读出头表现很好，但保留下来的表示并没有保存相同信息”。

response-only 和 full 的原生响应预测器也单独成表。原生 response 唯一的主指标是：在完整信道路由被判为 active，也就是 $r^{\mathrm A}(uv)=active$ 的转换上，不向模型泄露目标 CSI，靠多次遮挡查询拼出完整预测 CSI，再计算按场景宏平均的 NMSE。NMSE 是归一化均方误差，越低越好。null 样本上的凭空变化量是独立安全门。

SGCS 用来比较预测变化与真实变化的方向和相对结构是否一致。SGCS、变化方向余弦、变化幅度，以及路径损耗、时延扩展、角度扩展的变化都属于次要指标，不能在看完结果后替换主指标。

比较四臂保留下来的表示时，再训练一个完全统一的 action-conditioned response probe。它读取冻结的 $F_\theta (H_{u},M_{u})$、同一份带正负方向的编辑网格和同一查询位置，预测目标世界的物理 CSI 小块 $y_{v}^q$。统一 response probe 的唯一主指标是小块路由被判为 active，也就是 $r^{\mathrm R}(uv,q)=active$ 的样本上，按场景宏平均的物理小块 NMSE。不能看完实验后改成较好看的 latent 变化误差。这一设计把“信息留在最终编码器里”与“临时任务头会做题”分开了。

### 5.3 q_comp 只比较一对候选地图

把一条 active 边上的两张候选地图随机排列为 $M_{a}$ 和 $M_{b}$，先算相对分数：

$$
d_{\mathrm{comp}}
=
s(H,M_a)-s(H,M_b).
$$

$s(H,M)$ 是 CSI 与地图的原生匹配分数。$d_{\mathrm{comp}}$ 为正，表示模型更偏向第一张地图；为负，表示更偏向第二张。

再把分数差变成概率：

$$
q_{\mathrm{comp}}
=
\sigma
\!\left(
\frac{d_{\mathrm{comp}}}{T_A}
\right),
\qquad
T_A>0.
$$

`sigmoid` 是把任意实数压到 0 至 1 的函数。$T_{A}$ 是温度参数，只能在独立的 `source-calibration-fit` active 边上，用 NLL 选出，之后冻结。NLL 可以理解为“模型给真实答案的概率越低，惩罚越大”的校准损失。

这个公式不加截距，因此天然满足：

$$
q_{\mathrm{comp}}(d)
+
q_{\mathrm{comp}}(-d)
=
1.
$$

交换两张候选地图的顺序后，两个概率正好互补，不会出现同一对地图换个顺序就得到互相矛盾的结论。$q_{\mathrm{comp}}$ 的准确含义是：“在这条 active 配对编辑边的两个候选中，排在第一的地图是 CSI 生成侧地图的概率。”候选顺序必须随机。

null 边的理想输出接近 0.5，因为两张图在可检测信道效果上几乎没有区别，但不能用 null 数据重新拟合校准器。wrong-city、geometry-destroyed 和 empty map 也不自动拥有概率解释。只有源域校准预先包含同一种破坏类型时，才能对它们报告 $q_{\mathrm{comp}}$；否则只报原始分数和压力测试行为。

active 资格依赖配对反事实 CSI $H_{v}$ 和冻结教师 $T(H_{v})$。线上只有当前 CSI 和两张候选地图时，模型无法自行知道当前样本是否属于这个有效范围。因此，$q_{\mathrm{comp}}$ 是离线配对审计概率，不是一个可直接部署、还能自行判断适用范围的全局地图真伪概率。

$p_{\mathrm{fail}}$ 直接在预注册的 correct、active、gray、null 混合数据上学习真实失败标签，测试时不需要 route 标签。它仍然需要冻结的候选集合，而且只在源域特征覆盖范围内具有校准概率含义。

### 5.4 p_fail 预测定位会不会失败

定位头同时输出位置均值 $\mu$ 和原始尺度 $v$。均值是预测位置，尺度表示模型对各坐标方向有多不确定。尺度必须通过固定变换变成正数：

$$
\Sigma
=
\operatorname{diag}
\left(
\left(
\operatorname{softplus}(v)+\sigma_{\min}
\right)^2
\right).
$$

`softplus` 会把数值变成正数；$\sigma_{min}$ 是所有方法相同的最小不确定度；平方后得到方差；`diag` 表示这里只用每个坐标方向自己的方差，不额外预测方向之间的相关性。所有方法都用同一个 $\sigma_{min}$，在源域位置标签上共同采用 Gaussian NLL 和 Huber 点误差训练。Gaussian NLL 要求预测位置和不确定度互相匹配，Huber 损失则限制位置偏差，同时降低少数极端误差的影响。

把定位头的总体不确定度记为：

$$
u_g=\log\det(\Sigma).
$$

$\operatorname{det}(\Sigma)$ 是不确定范围的体积尺度，取对数后更容易校准。定位失败标签定义为：

$$
Y_{\mathrm{fail}}
=
\begin{cases}
1,
& \left\lVert\widehat x-x\right\rVert_2>\tau_{\mathrm{loc}},\\
0,
& \text{其他情况}.
\end{cases}
$$

$\widehat x$ 是预测位置，$x$ 是真实位置，$\lVert\cdot\rVert_2$ 是直线距离，$\tau_{\mathrm{loc}}$ 是在源域提前登记的失败距离阈值。

$p_{\mathrm{fail}}$ 还需要一个固定的配对候选集合。对实际送入定位头的地图 $M_{used}$，必须在运行 RT 和查看任何模型输出之前，用确定的纯地图规则生成相同大小的候选集合 $P(M_{used})$。

这条规则只能读取 $M_{used}$、公开无线配置，以及预注册的编辑库和随机种子。它不能读取目标世界 CSI、active/gray/null 标签、物理效果档位、哪一边是真实生成侧、bank/world ID、接收机位置标签或定位结果。

候选必须由标准编辑算子直接作用在 supplied map 上生成，不能等 RT 跑完后专挑效果最大的一条。

如果没有合法编辑，使用预注册的 fallback，并单独报告这种情况覆盖了多少样本，不能悄悄删掉样本。只有一张合法候选图时，集合大小为 1；有多张时，固定取最不利的相对分数：

$$
d_{\mathrm{used}}
=
\min_{M'\in\mathcal P(M_{\mathrm{used}})}
\left[
s(H,M_{\mathrm{used}})
-
s(H,M')
\right].
$$

这里始终以实际 supplied map 为正方向。$d_{\mathrm{used}}$ 越大，说明当前地图即使和最有竞争力的候选相比也更匹配；$d_{\mathrm{used}}$ 越小，风险通常越高。最终风险为：

$$
p_{\mathrm{fail}}
=
\operatorname{Cal}_{\mathrm{risk}}
\!\left(
d_{\mathrm{used}},
u_g(H,M_{\mathrm{used}})
\right).
$$

主风险校准器 `Cal_risk` 固定使用带 L2 正则的逻辑回归。L2 正则用来防止少量校准数据把系数推得过大。输入在拟合前按中位数和 MAD 做稳健标准化：

$$
\begin{aligned}
\widetilde d
&=
\frac{
d-\operatorname{median}_{\mathrm{fit}}(d)
}{
1.4826\,\operatorname{MAD}_{\mathrm{fit}}(d)+\epsilon_s
},\\
\widetilde u
&=
\frac{
u-\operatorname{median}_{\mathrm{fit}}(u)
}{
1.4826\,\operatorname{MAD}_{\mathrm{fit}}(u)+\epsilon_s
}.
\end{aligned}
$$

MAD 是每个值到中位数距离的中位数，对极端值不敏感。所有中位数和 MAD 只能来自 `source-calibration-fit`；防止分母为零的小常数 $\epsilon_{s}$ 在 `source-method-selection` 冻结。

校准器本身是：

$$
\operatorname{Cal}_{\mathrm{risk}}(d,u)
=
\sigma
\!\left(
\beta_0
+\beta_d\widetilde d
+\beta_u\widetilde u
\right).
$$

系数带有方向约束：

$$
\beta_d\le 0,
\qquad
\beta_u\ge 0.
$$

也就是说，当前地图相对越匹配，风险不能反而升高；定位头越不确定，风险不能反而降低。如果这个受约束模型不能胜过只看 $u_{g}$ 的模型，或者系数被压到几乎没有信息的边界，“compatibility 能转化为定位风险”这一主张就失败。

正则强度只能从预先声明的网格中选。每个候选都在 `source-calibration-fit` 拟合，只用 `source-calibration-selection` 的 NLL 决定最终参数，然后全部冻结。额外加入 $\widetilde d \times \widetilde u$ 交互项的模型只能作为预先声明的敏感性分析，不能与主校准器事后择优。

correct、active、gray、null 在校准数据中的比例，要先在 `source-method-selection` 冻结。因此，$p_{\mathrm{fail}}$ 的概率含义只对应这套预注册的配对审计混合比例。gray 和 null 可以提供真实定位失败标签，但仍然没有 ranking 标签。如果要把概率外推到自然部署时的样本比例，必须提前给出部署先验，并做标签分布偏移修正或重新加权；做不到时，只能报告风险排序与覆盖率，不能说部署概率已经校准。

$q_{\mathrm{comp}}$ 可以随机交换两张候选图的顺序，因为它预测哪一侧是生成地图。$p_{\mathrm{fail}}$ 不能使用这个随机符号。对每个定位输入，$M_{used}$ 必须就是实际送入定位头的图，候选数量、生成规则和取最小值的聚合方法在源城和目标城完全一致，$u_{g}$ 也必须来自 $(H,M_{used})$ 的定位输出。校准器到了目标城市完全冻结。

比较分成两层。公平比较时，四臂都使用同一个冻结 compatibility probe 的 margin 和同一个 $u_{g}$。原生诊断时，四臂都报告自身 $s(H,M)$ 产生的 $q_{\mathrm{comp}}$ 与 $p_{\mathrm{fail}}$，并明确只有 alignment-only 和 full 直接接受过 alignment 训练。

风险表要同时报告只看 $d_{\mathrm{used}}$、只看 $u_{g}$、两者联合这三种模型，还要报告定位 ECE、Brier、NLL、Spearman、风险覆盖曲线、AURC，以及保留 90%、75%、50% 样本时的中位误差和 P90。P90 是第 90 百分位误差，也就是 90% 样本的误差不超过这个数。

ECE 看预测概率和真实频率是否一致；Brier 是概率与 0/1 标签的均方误差；AURC 衡量按风险拒绝高风险样本后，剩余误差下降得是否足够快。

校准概率还必须限制在源域数据覆盖的特征范围内。先把标准化后的 $d_{\mathrm{used}}$ 和 $u_{g}$ 截断到 -5 至 5：

$$
v
=
\operatorname{clip}
\!\left(
(\widetilde d,\widetilde u),
-5,5
\right).
$$

然后用 `source-calibration-fit` 的 $v$ 计算中心 $\mu_{v}$ 和协方差 $\Sigma_v$，再算每个点到源域分布的稳健 Mahalanobis 距离：

$$
D_{\mathrm{support}}^2(v)
=
(v-\mu_v)^{\top}
\left(
\Sigma_v+\epsilon_I I
\right)^{-1}
(v-\mu_v).
$$

这个公式可以跳过，只需记住：$D_{\mathrm{support}}^2$ 越小，样本越像源域校准数据；越大，样本越可能超出校准器熟悉的范围。

公式中的 $^\top$ 表示转置，也就是把行和列交换后参与乘法；$^{-1}$ 表示逆矩阵，用来按照源域数据在不同方向上的正常波动调整距离。$\epsilon_{I}$ 防止协方差矩阵无法求逆，$I$ 是单位矩阵。

距离阈值取 `source-calibration-selection` 上预注册的 $1 - \alpha_{\mathrm{support}}$ 分位数；$\alpha_{\mathrm{support}}$ 和 $\epsilon_{I}$ 在 `source-method-selection` 后冻结。所有目标样本都要标记为范围内或范围外，并报告范围外比例。

范围外样本只报原始分数和真实误差，不能把 $p_{\mathrm{fail}}$ 当作已校准概率。

不同实验臂学到的源域有效范围可能不同，不能允许 full 把难样本都判成范围外，再获得看似更好的 ECE。四臂的主校准比较使用共同范围：

$$
S_{\mathrm{common}}
=
\left\{
i\;\middle|\;
D_{\mathrm{support},a}^2(v_i)
\le
t_{\mathrm{support},a},
\ \forall a\in\{00,10,01,11\}
\right\}.
$$

也就是只比较四臂都认为落在各自源域有效范围内的同一批样本。共同范围覆盖率必须超过预注册下限，而且 full 的范围外比例相对两个单分支要达到预注册非劣界。通过这两道门后，才比较共同范围上的 ECE、Brier、NLL 和可靠性置信区间。每个臂自己的范围内校准只作补充；全体查询样本仍要报告原始风险排序、AURC 和误差，但不能给范围外样本附上概率含义。

风险主结果只使用完全没有目标更新的 $k=0$ checkpoint。若要报告 $k=8、32、128$，必须先在源城市复现完全相同的少样本适配流程，并为每个 $k$ 和每套更新规则分别拟合、选择、冻结校准器。目标城市的少量位置标签不能参与风险校准。否则，少样本实验只报告定位误差，不报告 ECE、AURC 或概率结论。

当前 $p_{\mathrm{fail}}$ 的概率主张明确需要配对候选上下文。只有一张待检查地图、又没有预注册候选库的部署场景，不在本论文的校准风险结论内。固定编辑候选库可以放在附录探索，但不能写成论文已经解决了单地图风险检测。

## 6. 理论能说明什么，不能说明什么

本节的作用是划清结论边界。几个命题说明“现有监督缺了什么”和“新监督约束了什么”，不直接保证有限大小的神经网络一定训练成功，也不预先证明两个分支联合后一定更好。

### 6.1 命题 1：只看原世界，学不到编辑后的唯一答案

常规正确配对训练给预测器的输入是 $(H,M,a)$，其中 $a$ 表示地图动作。训练数据只出现 $a=0$，也就是地图没有变化。即使两个预测器在这些已见样本上几乎处处输出相同、训练风险也相同，它们在 $a\ne 0$ 的编辑样本上仍然可以给出任意不同的结果。

因此，正确 CSI 与正确地图上的 endpoint 或 masked prediction，不能单独规定地图编辑后应当得到哪个目标 CSI。这个命题只指出监督覆盖范围的缺口，不表示所有 $(H_{i},M_{j})$ 都不可能配套，也不把 wrong-map 现象当成全局不相容的证明。

### 6.2 命题 2：对称四元组挡住单模态捷径

固定场景库 $b$、位置 $x$、无线配置 $c$ 和共享物理扰动 $\xi_{\mathrm{phys}}$，取一条 active 无向边 `{u,v}`。两组样本是：

$$
\begin{aligned}
\mathcal Q_{\mathrm{matched}}
&=
\left\{
(H_u,M_u),(H_v,M_v)
\right\},\\
\mathcal Q_{\mathrm{alternative}}
&=
\left\{
(H_u,M_v),(H_v,M_u)
\right\},\\
\operatorname{Marginal}_{H}
(\mathcal Q_{\mathrm{matched}})
&=
\operatorname{Marginal}_{H}
(\mathcal Q_{\mathrm{alternative}})
=
\{H_u,H_v\},\\
\operatorname{Marginal}_{M}
(\mathcal Q_{\mathrm{matched}})
&=
\operatorname{Marginal}_{M}
(\mathcal Q_{\mathrm{alternative}})
=
\{M_u,M_v\}.
\end{aligned}
$$

“边际相同”是说，把配对关系遮住后，两组里出现的 CSI 完全相同，出现的地图也完全相同。只要 active 门对两个方向采用同一规则，而且两侧等权进入数据，物理门和教师门都不会破坏这个等式。因此，任何只看 CSI 的 $\phi (H)$，或只看地图的 $\psi (M)$，都无法区分正确配对和交换配对。

这个命题没有排除“同时读取 CSI 和地图后，靠 world ID 对暗号”的捷径。scene-level world bank 的跨位置复用、完整留出 unseen bank 和 unseen city，以及显式 variant-ID baseline，负责从实验上排除这种可能。

### 6.3 命题 3：active 排序损失为零，只说明局部相对关系

如果某条 active 边的间隔 $m_{uv}$ 大于 0，而且两个方向的 ranking loss 都为零，那么必须满足：

$$
\begin{aligned}
s(H_u,M_u)
&\ge
s(H_u,M_v)+m_{uv},\\
s(H_v,M_v)
&\ge
s(H_v,M_u)+m_{uv}.
\end{aligned}
$$

也就是，对边的两个方向，正确地图得分都至少比交换地图高出预设间隔。常数打分、CSI-only 打分、map-only 打分，以及只看同一 tile 场景 ID 的打分，都不能同时满足这两个不等式。

这一结果只证明分数必须利用 CSI 和地图的联合配对关系，而且只在 active 配对范围内成立。它不能推出任意城市、任意位置、任意错误地图之间都存在绝对 compatibility 排序。

### 6.4 引理：target regression 比排序多了一条“靠近谁”的约束

假设 active 配对上，原世界 identity 损失 $L_{B}=0$，目标世界回归损失 $L_{target}=0$，而且所用距离只有在两个输入完全相同时才等于零，那么：

$$
\widehat z_{u\to u}=z_u,
\qquad
\widehat z_{u\to v}=z_v,
\qquad
\widehat{\Delta z}_{uv}=\Delta z_{uv}.
$$

$z$ 是 CSI 的内部表示。第一行表示零动作时还原源世界，第二行表示执行编辑后落到目标世界，第三行表示方向和幅度都正确。$L_\Delta$ 因而不是新的识别来源，只是在有限训练条件下，直接加强变化方向和幅度。ranking 只要求错误分支分数足够低，并没有规定预测向量要靠近 RT 重追踪得到的真实目标。这是损失语义上的区别，不包装成新的深层理论贡献。

### 6.5 动作本身能提供多少额外信息

令 $S$ 表示不含具体动作的全部合法输入，$A$ 是进入模型前的原始带符号编辑网格，$Z$ 是目标世界查询小块的内部表示：

$$
\begin{aligned}
S
&=
\left(
V_m(H_u),G(M_u),c,m,q
\right),\\
A
&=
\Delta_{uv},\\
Z
&=
z_v^q.
\end{aligned}
$$

branch bundle 保证同一个 $S$ 对应多个有非零概率出现的动作。

在平方误差下，知道动作与不知道动作时的理论最优风险之差为：

$$
R_{\mathrm{noA}}^\star
-
R_{\mathrm A}^\star
=
\mathbb E
\left[
\operatorname{tr}
\operatorname{Cov}
\left(
\mathbb E[Z\mid S,A]
\mid S
\right)
\right]
\ge 0.
$$

`Risk_best(no action)` 是不知道动作时能达到的最小平均误差，`Risk_best(with action)` 是知道动作时的最小平均误差。公式中的竖线 `|` 表示“在右侧条件已经给定时”，不是除法。比如 `E[Z | S,A]` 表示已知 $S$ 和动作 $A$ 时，目标 $Z$ 的平均值；外层的 `| S` 表示固定同一个 $S$，再看这个平均值会怎样随动作变化。

$E$ 表示对数据取平均，`Cov` 表示波动范围，`trace` 把各表示维度的波动相加。直观上，只要同样的源输入配上不同动作时，目标平均结果确实会改变，动作就可能降低最佳预测误差；如果改变发生在非零比例的样本上，这个差值才严格大于 0。

实际模型读取的是动作编码器给出的 $e_{uv} = E\phi (A)$。动作编码器可能塌缩成几乎同一个向量，把原始动作信息丢掉，所以理论式不能证明模型真的使用了动作。必须通过 full 对 no-action、动作 embedding 的有效秩，以及固定 checkpoint 的 action swap 实验验证。仅有物理 CSI 差异也不够，冻结教师还必须保留这种差异。这个结果只讨论“理论上能达到的最低平均误差”，不保证有限网络和实际优化一定达到这个最低值。

### 6.6 理论不预先承诺协同

alignment 的相对排序无法唯一确定 response 向量；response 任务头命中目标，也不保证 compatibility 信息留在最终编码器 $F_{\theta}$ 中。这只能说明两项监督互不包含，不能证明联合训练一定优于单独训练。互补、简单相加还是相互干扰，必须交给第 7 节的 $2\times 2$ 四臂实验、交互效应和参数量、FLOPs 匹配的机械拼接对照判断。

### 6.7 源 CSI 的信息价值，以及不能输入真实位置的边界

令 $B$ 包含只看地图和编辑的方法可以合法读取的全部信息，$C$ 是遮挡后的源 CSI，$Y$ 是目标 CSI 表示的真实变化：

$$
\begin{aligned}
B
&=
\left(
G(M_u),\Delta_{uv},c,m,q
\right),\\
C
&=
V_m(H_u),\\
Y
&=
\Delta z_{uv}^q.
\end{aligned}
$$

在平方误差下，加入源 CSI 前后的理论最优风险差为：

$$
R_B^\star
-
R_{B,C}^\star
=
\mathbb E
\left[
\operatorname{tr}
\operatorname{Cov}
\left(
\mathbb E[Y\mid B,C]
\mid B
\right)
\right]
\ge 0.
$$

左边第一项是只看地图和编辑能达到的最低误差，第二项是再加入源 CSI 后的最低误差。竖线 `|` 仍表示“在右侧条件已经给定时”：`E[Y | B,C]` 是已知地图、编辑和源 CSI 后，目标变化 $Y$ 的平均值；外层的 `| B` 是固定地图与编辑，再看加入不同源 CSI 后这个平均值怎样变化。

只要同一世界和动作跨许多位置复用，而且源 CSI 会改变目标变化的条件平均值，源 CSI 就有额外信息。实验用 full 与 edit/map-only 对照检验这点。

这个公式只讨论理论上能达到的最低平均误差，不保证有限网络真的达到这个最低值，也不保证源 CSI 能彻底消除位置歧义。真实接收机位置 $x$ 只能进入 RT 生成、配对索引、效果标注、下游定位标签和 oracle 诊断，不能成为主预训练模型的输入。

## 7. 四臂实验决定 CSI-PAIRS 是否成立

### 7.1 四臂只改变两项监督的开关

四臂是一个严格的 $2\times 2$ 因子实验。第一个开关决定是否加入 alignment，第二个开关决定是否加入 response。其余工程配置保持一致。

| 实验臂 | 共同 endpoint $\mathcal L_{B}$ | alignment $\mathcal L_{A}$ | response $\mathcal L_{R}$ | 下游保留什么 |
|---|---:|---:|---:|---|
| **Endpoint** | 有 | 无 | 无 | 同一个 $F_{\theta}$ |
| **Alignment-only** | 有 | 有 | 无 | 同一个 $F_{\theta}$ |
| **Response-only** | 有 | 无 | 有 | 同一个 $F_{\theta}$ |
| **CSI-PAIRS full** | 有 | 有 | 有 | 同一个 $F_{\theta}$ |

全文固定使用这四个名称。`Endpoint` 只指第 4.2 节的同世界 identity 或 matched prediction，不再一会儿表示 SigMap 定位损失，一会儿又表示 MAE。

### 7.2 除监督开关外，所有条件都要匹配

四臂必须共享同一个 Stage-0 教师、物理目标定义和 CSI 初始化；同一批场景级世界库、自然原图、原始 `NK` 条信道和共同位置集；同一批边、方向、遮挡、查询、数值增强和 batch 顺序；同样大小的地图 tokenizer、状态编码器、融合模块和预测器；相同的训练步数、相同的去重后源物理单元数量、源域选模集和调参次数；相同源域定位标签、位置头、优化器与 $k>0$ 时允许更新的参数集合；相同的目标城市信息预算。

所有臂都实例化同规格模块，只把对应 loss factor 设为零或非零。$\lambda_{A}$ 在 alignment-only 和 full 中必须相同，$\lambda_{R}$ 在 response-only 和 full 中也必须相同。没有使用 action 的实验臂不能靠加宽 backbone 获得补偿，full 也不能因为多两个训练目标就得到更多目标城市调参机会。

alignment 需要额外运行交换地图的前向计算，response 需要额外运行动作分支，因此四臂的原始 FLOPs 不可能字面相同。主结果必须如实报告训练 FLOPs、源域 forward 次数和实际耗时。

差异超过预注册容差时，要给单分支补充 equal-FLOP 控制，也就是让它们使用相同总计算量。

不能在主四臂中临时调 $\lambda_{A}$ 或 $\lambda_{R}$ 来凑总梯度，因为那会改变两个因子的实际剂量。等步数、等 FLOPs 和梯度范数匹配只作为稳健性实验，用来检查 full 的优势是否只是来自更多更新。

### 7.3 主结果、交互量和七道协同门

定位误差越小越好，为了计算交互，先把它转换成越大越好的无量纲效用。对实验臂 $a$、目标城市 $c$、标签预算 $k$、独立 world bank $b$、配对训练随机种子 $s$ 和配对标签抽样 $d$，只在冻结的目标查询位置上，用预注册的正确或自然 supplied map，计算 bank 内定位误差的中位数。少样本适配用过的位置及其所有 sibling worlds 必须先排除。

每个 bank 的平均对数误差定义为：

$$
\bar{\ell}(a,c,k,b)
=
\mathbb E_{s,d}
\left[
\log
\left(
\frac{
\operatorname{MedianError}
(\mathrm{query},\mathrm{correct},a,c,k,b,s,d)
}{
1\,\mathrm m
}
\right)
\right].
$$

除以 1 米只是去掉单位，方便取对数。$k=0$ 没有标签抽样。总体主效用定义为：

$$
J_a
=
-\frac{1}{2|\mathcal C_{\mathrm{target}}|}
\sum_{c\in\mathcal C_{\mathrm{target}}}
\sum_{k\in\{0,8\}}
\frac{1}{|\mathcal B_c|}
\sum_{b\in\mathcal B_c}
\bar{\ell}(a,c,k,b).
$$

负号把“误差越小”改成“效用越大”。它先在每个 bank 内算中位误差，再对随机种子、标签抽样、bank 和城市做规定好的平均，避免样本多的城市或 bank 支配结果。主效用只使用 $k=0$ 和 $k=8$。

$k=8、32、128$ 使用多个在查看目标结果前冻结的、按唯一位置抽取的标签集合。每个标签抽样中，8 个接收机位置必须是同一抽样下 32 个位置的子集，32 个又是 128 个的子集。每个位置只提供一条正确或自然观测，四臂共享完全相同的抽样。抽样次数在 `source-method-selection` 后冻结。普通的整城市 pooled median 仍放在主表，但不参与交互量，不能在整城市中位数和 bank 宏平均之间事后选好看的口径。

四臂效用按两个二进制开关记为：$J_{00}$ 是 Endpoint，$J_{10}$ 是 Alignment-only，$J_{01}$ 是 Response-only，$J_{11}$ 是 full。交互量为：

$$
\Delta_{\mathrm{int}}
=
J_{11}-J_{10}-J_{01}+J_{00}.
$$

它衡量 full 超出两个单分支简单相加预期的部分。$\Delta_{\mathrm{int}} > 0$ 本身还不够，必须看置信区间和实际效应下限。

主置信区间采用配对多层 bootstrap。bootstrap 可以理解为反复重采数据，观察结论在抽样变化下是否稳定。目标城市固定为报告层，每个城市内重采独立 bank；训练 seed 在所有 bank 上成对重采；$k=8$ 的标签抽样在对应城市内成对重采。

每一次抽样时，bank、seed 和标签抽样索引在四臂之间完全相同。bank 始终是最高层的数据推断单位，seed 和标签抽样只传播算法与少样本选择的不确定性，不能假装增加了场景样本量。

还要补充只重采 bank、固定 seed 和标签抽样的聚类置信区间，以及每次删去一个 seed 或标签抽样的敏感性分析。主判定使用预注册的多层置信区间。

论文只有同时通过以下七道门，才能写“在预注册的对数 bank 中位误差效用上存在正交互”：

1. full 的 $J_{a}$ 分别胜过 alignment-only 和 response-only。
2. full 的 active CGS 相对 alignment-only 达到预注册非劣界，也就是没有差到超过允许范围。
3. full 的 active 完整信道、target-free CSI NMSE 相对 response-only 达到预注册非劣界。
4. $\Delta_{\mathrm{int}}$ 的配对多层 scene-bank bootstrap 置信区间下界高于 0，而且超过预注册的最小实际效应。
5. 两个目标城市的 $k=0$ 和 $k=8$ 都不能出现超过允许范围的反向退化。
6. full 的 $J_{a}$ 分别达到对 equal-FLOP alignment-only 和 equal-FLOP response-only 的预注册实际优越界，排除优势只是来自更多 forward 或更新。
7. full 必须同时胜过第 7.4 节的 parameter-matched 和 FLOP-matched 两个机械拼接对照。若同一个配置确实能同时匹配参数量与 FLOPs，则它必须在两项资源核算上都通过。

full 同时胜过两个单分支只是必要条件，不能单独证明 synergy。$\Delta_{\mathrm{int}}$ 也只是在预注册的“负对数 bank 中位误差”尺度上的交互，不能扩写成所有指标都存在普遍协同。如果前 3 条成立，但 $\Delta_{\mathrm{int}} \le 0$，论文只能写“两类监督互补并且可以统一训练”。如果 full 没有胜过两个单分支，CSI-PAIRS 就不能作为论文 headline 方法。

### 7.4 机械拼接对照排除“把两个模型接起来就行”

机械拼接对照训练两个完全独立、互不共享梯度的编码器，一个只学 alignment，另一个只学 response。下游把两种表示拼接起来，再经过一个事先声明、可以训练的线性 bottleneck，压到与 full 相同的表示维度。bottleneck 和位置头一起训练，只能使用四臂共同的源域定位标签；它的参数量、训练数据和 FLOPs 全部计入资源核算。这里“事先声明”是指在看结果前固定架构，不是冻结一个随机投影。随机投影可能主动丢掉单分支信息，使对照被人为削弱。

不能直接把两个编码器都设成“半宽”。Transformer 参数量大致随宽度平方增长，半宽不等于一半资源。要先对实际实现做性能统计，再反解合适的宽度和深度。如果一个配置无法同时匹配参数量和 FLOPs，就分别报告 parameter-matched 和 FLOP-matched 两个拼接对照，并公开各自资源误差比例。

资源匹配的拼接对照只能回答“相同总资源下，共享编码是否更高效”，仍不能排除 full 中每项任务拥有更大有效容量。为此还要报告 `2× compute generous upper control`，也就是一个宽松上界：使用两个全宽单分支编码器，不降维直接拼接，计算量约为 full 的 2 倍。它要使用匹配输入维度的位置头，并如实计算额外参数和 FLOPs。full 若以约一半资源追平它，支持共享编码效率；full 若还能胜过它，才构成更强的非机械证据。

P1 再加两项停止梯度诊断：一项让 response loss 不能更新共享 $F_{\theta}$，另一项让 alignment loss 不能更新共享 $F_{\theta}$。如果停止某分支更新 $F_{\theta}$ 后，对应的跨能力和下游增益消失，才能支持两类监督确实通过共享表示发生作用。

### 7.5 每个分支先做好本职，再看跨能力

| 目标 | 模型原生指标 | 四臂统一冻结探针 |
|---|---|---|
| Alignment | 四臂的预测 energy active AUROC 和 $q_{\mathrm{comp}}$ 校准；alignment-only 与 full 直接受 $\mathcal L_{A}$ 监督 | 相同探针架构、训练步数下的 CGS 和 null 过度区分 |
| Response | response-only 与 full 的 target-free NMSE、SGCS、变化方向和幅度 | 相同探针架构、训练步数下的动作条件响应可解码性 |
| Localization | 不加额外任务头，只使用 $F_{\theta}$ | 相同位置头和相同标签预算 |
| Risk | 四臂都报告原生 energy；alignment-only 与 full 直接受 $\mathcal L_{A}$ 监督 | 四臂统一探针 margin 和同一个 $u_{g}$ |

full 不需要在每个原生指标上都显著超过专门的单分支。它必须在两个分支各自的本职指标上达到非劣，同时在另一分支对应的跨能力和最终跨城定位上获得增益。

## 8. 五个必须优先回答的实验问题

### RQ1：现有模型是否反事实一致地使用地图

按第 1.1 节的六种输入条件，至少评测两个 map-conditioned 模型。外部代码或数据拿不到时，可以按照论文规格做 controlled implementation，但必须准确写成“按论文规格控制实现”，不能声称是 faithful reproduction。

主读数是定位误差中位数和 P90。只有原本就有同构 masked predictor 的模型才额外报告预测误差，不能为了表面统一，临时给纯定位模型加一个新任务头。

active 和 null 必须分开解释。如果 active alternative 与 correct 几乎没有差别，而 empty map 明显变差，支持“模型会用粗略地图条件，却不会根据局部几何变化响应”。null alternative 本来不该导致明显退化；如果模型反应很强，说明它可能只是在识别编辑痕迹。wrong-city 只是一项压力测试，不能替代 active 配对证据。

训练后再用同一套六条件做 before/after 行为恢复图，检查四件事：active margin 是否恢复；null 是否没有过度判错；配对上下文中的 $p_{\mathrm{fail}}$ 是否会随真实定位失败上升；使用正确地图时的跨城定位是否改善。wrong-city、geometry-destroyed 和 empty 若没有预先进入源域风险校准类型，只报原始风险分数，不能画成校准概率。错图后误差上升本身不是优点；模型若没有可靠的 compatibility 或风险读数，不能称为 graceful degradation，也就是“知道自己可能错了并稳妥退化”。

### RQ2：alignment 是否学到范围有限但可靠的相对 compatibility

四臂统一 CGS 主表必须报告 active AUROC，并按 $\delta_{\mathrm{phys}}$ 分四档；未见位置、未见编辑、未见 world bank 和未见城市；null 分数差与错误不兼容率；gray 分数差分布，不给 gray 强行贴二分类标签；constant、CSI-only、map-only、scene-ID-only、edit-status XOR 和 variant-ID matcher 这些捷径基线；以及四臂原生预测 energy 与统一探针是否一致。

alignment-only 必须在留出 bank 和城市的 active CGS 上胜过 Endpoint，同时 null 过判不能超过预注册容差。如果只在训练 bank 上很高，换到留出 bank 就下降，说明模型记住了 world fingerprint，不算学会 compatibility。

P1 加入 V4.1 旧方案“所有非对角组合都做 ranking”的对照，直接检查它是否一边提高 active AUROC，一边制造 null 和 gray 误报。这项对照用来证明 v6 的 effect routing 确实修复了可测的错误监督，不只是把措辞写得更保守。

### RQ3：response 能否预测 RT 重追踪后的方向和幅度

先冻结一组 target-free 的遮挡与查询组合，使完整 CSI 中每个目标小块至少被查询一次。每次模型只能看到遮挡后的源 CSI $V_{m}(H_{u})$、源地图和带方向的编辑，预测一个目标小块。重叠小块按预注册规则取平均，再拼成完整预测 CSI $\widehat H_{v}$。真实目标 $H_{v}$ 只能在整张 CSI 组装完成后用于评分，不能在组装过程中提供任何小块或 token。

原生 response 唯一主指标是：在完整信道路由为 active，也就是 $r^{\mathrm A}(uv)=active$ 的转换上，对不看目标的完整 $\widehat H_{v}$ 计算 scene-macro CSI NMSE。scene-macro 表示先在每个独立场景或 bank 内计算，再让每个场景等权平均，避免大场景支配结果。null 安全门另行检查模型凭空预测的变化是否落在重复仿真的噪声底等效界内。

次要指标包括 SGCS；路径损耗、时延扩展和角度扩展的变化误差与方向正确率；active 样本的变化方向余弦和幅度比；gray 样本的目标绝对误差。gray 不做强变化结论。

辅助的内部表示指标 TransitionSkill 定义为：

$$
\operatorname{TransitionSkill}
=
1-
\frac{
\displaystyle
\sum_{(u,v)\in S_+}
d\!\left(
\widehat z_{u\to v},z_v
\right)
}{
\displaystyle
\sum_{(u,v)\in S_+}
d(z_u,z_v)
}.
$$

$d(a,b)$ 使用第 4.2 节已经冻结的内部表示归一化距离。`S₊` 只包含 $r^{\mathrm A}(uv)=active$ 且教师也敏感的完整转换单元。分子是模型执行动作后的目标预测误差，分母是“直接复制源表示、不预测任何变化”的误差。分数为 1 表示预测完全正确，0 表示只和复制源表示一样好，小于 0 表示还不如复制。必须先在每个独立 bank 内分别计算分子、复制分母和比值，再做场景宏平均。分母或覆盖不足时记 `N/A`，不能事后加小常数把它变成可计算。

response 必须包含以下对照：

| 对照 | 它在排查什么 |
|---|---|
| CSI persistence 或 copy | 完全不预测变化是否已经很强 |
| no-action | 同一源输入存在多个未来时，动作是否提供信息 |
| 固定 checkpoint、几何幅度匹配的 action swap | 同一个训练好模型是否读取了正确动作 |
| w/o source map | 预测响应是否需要源几何 |
| edit/map-only | 模型是否只记住某类编辑的平均响应 |
| CSI-only | 模型是否只靠源 CSI 外推 |
| oracle-x | 显式知道接收机位置时的可预测上界，必须单列为额外信息预算 |

测试时的 action swap 固定同一个源物理单元、遮挡、查询和 checkpoint，只把正确动作 $e_{uv}$ 换成错误动作 $e_{u}w$。错误动作必须来自同一 source、同一编辑类别和相近编辑幅度。报告 exact、fallback 和 failed candidate 各自的覆盖率，所有方法只能在完全相同的合格样本上比较，不能各用各的分母。

### RQ4：联合监督是否改善严格的未见城市定位

至少完整留出两个城市，目标位置标签数固定为：

$$
k\in\{0,8,32,128\}.
$$

每个城市分别报告误差中位数、P90、至少 3 个训练随机种子的算法方差和 scene-bank bootstrap 区间，再给城市宏平均。$k=0$ 和 $k=8$ 是主检验，$k=32$ 和 $k=128$ 只展示标签效率曲线。同时报告域内正确自然图精度，检查配对训练有没有牺牲常规性能。

$k$ 表示拥有标签的不同接收机位置数量，不是 CSI 与地图样本条数。few-shot support 中，每个位置只提供一条预注册的正确或自然观测。support 是用来适配模型的少量样本，query 是最终评测样本。二者必须按第 2.7 节以整组隔离：任何 support 位置及其所有 sibling worlds，都不能进入定位中位数、P90、风险或交互效应的计算分母。

所有城市使用已知基站位姿建立同一种右手米制局部坐标。原点是参考基站，x 轴沿阵列 boresight 的水平投影，z 轴沿重力方向。地图、基站 token 和位置输出必须使用同一个刚体坐标。目标城市不能用位置标签、城市包围盒、目标数据均值方差或测试结果拟合归一化。

严格 $k=0$ 的 single-map localization 只允许读取 supplied target map、基站位姿和单条待定位 CSI。禁止用目标城市无标签 CSI 更新模型，也禁止 target fingerprint 或 reference 库、目标统计量、目标校准、梯度更新和目标超参数选择。需要目标参考库的方法必须单列为 transductive 或 retrieval 设置。第 5.4 节的 paired-proposal risk audit 多用了候选地图，是额外信息设置，不能和严格单地图定位混写。

主结果表先放严格四臂，再放外部强基线。最终判断看 full 是否胜过两个单分支，以及第 7.3 节的交互量。full 胜过一个较弱外部基线，不能替代严格内部比较。

### RQ5：配对上下文中的 compatibility 能否转成可靠定位风险

RQ5 是 paired-proposal risk audit。除了 single-map localization 可用的信息，它还额外允许第 5.4 节预先冻结、只由地图侧规则生成的 $P(M_{used})$。这部分额外候选地图预算必须单列，不能暗示严格单图部署也能直接计算 $p_{\mathrm{fail}}$。

四臂先使用统一探针 margin 做公平比较，再全部报告原生预测 energy，并标明只有 alignment-only 和 full 直接优化过 alignment。所有校准只使用第 2.7 节的 `source-calibration-fit` 和 `source-calibration-selection`，到了目标城市完全冻结。主表中的 ECE、Brier、NLL 和 AURC 只对应 $k=0$ checkpoint。

风险结论以第 14 节 C9 的四道门为准，不再另设一套八项硬门：

1. 比较候选集合提前冻结，主结果只看 k=0 的配对审计样本，四臂用共同支持范围作分母。
2. ECE、Brier、NLL 和可靠性置信区间通过提前登记的校准门。
3. 共同支持范围的覆盖率足够，而且 Full 的支持范围外比例不劣于两个单分支。
4. 所有提前登记评测点的 AURC 都胜过随机拒绝和只用 $u_{g}$ 的对照。随着覆盖率降低，保留下来的样本定位误差还要单调变小。

$q_{\mathrm{comp}}$ 与 $p_{\mathrm{fail}}$ 仍必须分表，不能拿一组代替另一组；分表是报告要求，不是第五道独立硬门。active 是否抬高风险、correct/null 是否虚报，写入诊断表，不单独否决 C9。

## 9. 路径机制：收益是否出现在地图真正影响无线传播的地方

### 9.1 A_path 用来分层，不单独证明因果

RT 是射线追踪。它会记录信号从基站到接收机经过的传播路径，以及每条路径的接收功率。对固定位置 $x$ 和两个世界 `u、v`，分别把保留路径集合写成 `Paths(u)` 和 `Paths(v)`。`Power(u,p)` 表示路径 $p$ 在世界 $u$ 中的接收功率，`Power(v,p)` 表示它在世界 $v$ 中的接收功率。

优先使用 RT 提供的 persistent path ID 和按顺序记录的表面交互 ID，把两个世界的路径一一对应。如果引擎没有稳定路径 ID，就在 `source-method-selection` 上提前固定对时延、到达角 AoA、出发角 AoD 和交互顺序的容差，再用确定性的二分图匹配建立一一对应。没有匹配上的源路径记为消失路径，没有匹配上的目标路径记为新增路径。匹配容差、相同分数时的选择规则和 path-ID 规则，都必须在查看目标结果和模型结果前冻结。

对源世界中的路径定义指示量 $I^u_{uv}(p)$：如果路径碰到被编辑表面，或者在另一个世界找不到对应路径，就取 1，否则取 0。目标世界的 $I^v_{uv}(p)$ 同理。然后计算：

$$
A_{\mathrm{path}}(x,u,v)
=
\frac{
\displaystyle
\sum_{p\in\operatorname{Paths}(u)}
\operatorname{Power}(u,p)\,I^{u}_{uv}(p)
+
\sum_{p\in\operatorname{Paths}(v)}
\operatorname{Power}(v,p)\,I^{v}_{uv}(p)
}{
\displaystyle
\sum_{p\in\operatorname{Paths}(u)}
\operatorname{Power}(u,p)
+
\sum_{p\in\operatorname{Paths}(v)}
\operatorname{Power}(v,p)
},
\qquad
0\le A_{\mathrm{path}}\le 1.
$$

分子是“与这次编辑有关，或因编辑新增、消失的路径功率”，分母是两个世界全部保留路径的总功率。因此，$A_{\mathrm{path}}$ 越大，表示这次地图编辑影响了越大比例的主要无线传播能量。每条路径的功率在各自世界中只算一次。若一条匹配路径碰到编辑表面，源、目标两侧的功率各算一次，正好对应分母中的两个世界，不是重复计数。

RT 只保留累计功率达到预注册 `Q%` 的主要路径。$Q$ 根据 `source-method-selection` 上的路径截断收敛曲线冻结，要求继续提高累计功率阈值时，$A_{\mathrm{path}}$ 的变化已经小于预注册容差。

零影响组定义为 $A_{\mathrm{path}} \le \epsilon_{\mathrm{path}}$。$\epsilon_{\mathrm{path}}$ 来自 no-edit 或 identity 数值重复实验中 $A_{\mathrm{path}}$ 的 $1 - \alpha_{\mathrm{path}}$ 分位数，$\alpha_{\mathrm{path}}$ 必须在读取这块数据前固定。$A_{\mathrm{path}}$ 只用于评测分层，不能输入模型，不能参与动作表示、active/gray/null route，也不能改变采样优先级。

地图侧编辑幅度只由编辑网格计算：

$$
\delta_{\mathrm{map}}
=
w_o
\frac{\text{占用变化面积}}{\text{tile 面积}}
+
w_h
\frac{\text{高度变化的均方根}}{h_{\mathrm{ref}}}
+
w_m
\frac{\text{材质变化面积}}{\text{tile 面积}}.
$$

`w_o、w_h、w_m` 是三种变化的权重，$h_{ref}$ 是参考高度。它们只能用 `source-method-selection` 的地图尺度确定，不能读取 CSI、route、传播路径或模型结果。

比较 active 与 zero，或比较低 $A_{\mathrm{path}}$ 与高 $A_{\mathrm{path}}$ 时，要控制编辑类别、面积或高度或材质变化规模、编辑位置到基站和接收机的距离、场景、LoS/NLoS，以及地图侧幅度 $\delta_{\mathrm{map}}$。同时报告 SMD、两组特征重叠程度和有效样本量。SMD 是标准化均值差，用来检查两组样本在这些条件上是否真的平衡。

$\delta_{\mathrm{phys}}$ 是编辑影响信道后的中间结果，不能先按它匹配两组，再声称测出了 path-incidence 的作用。

预注册要检查四个现象：

- alignment 分数差随 $A_{\mathrm{path}}$ 增长；
- full 或 response 相对 no-action 和 action-swap 的响应误差优势随 $A_{\mathrm{path}}$ 增长；
- zero 组的预测变化不超过噪声底容差；
- full 相对两个单分支的收益主要出现在几何确实相关的分层。

这类结果称为“与机制一致的效应异质性”，论文术语是 `mechanism-consistent effect heterogeneity`，不能写成“path-incidence 单独识别了因果效应”。

### 9.2 定位样本只能聚合与自身世界相连的编辑

$A_{\mathrm{path}}(x,u,v)$ 描述的是从世界 $u$ 到世界 $v$ 的一次转换，不能把整张超立方世界库中与当前源世界无关的所有边都平均给同一个定位样本。

如果定位样本使用 world bank 中的状态 $u$，只聚合与 $u$ 直接相连的边：

$$
\begin{aligned}
A_{\mathrm{path,loc}}(x,u)
&=
\frac{
\displaystyle
\sum_{v\in\mathcal N_b(u)}
A_{\mathrm{path}}(x,u,v)
}{
|\mathcal N_b(u)|
},\\
\mathcal N_b(u)
&=
\left\{
v\;\middle|\;\{u,v\}\in\mathcal E_b
\right\}.
\end{aligned}
$$

$N_{b}(u)$ 就是状态 $u$ 的直接邻居集合，`|N_b(u)|` 是邻居数量。这个平均值表示“对从当前世界直接可做的预注册编辑而言，传播路径有多大比例会受影响”。

如果主定位输入是独立自然图 `M_b_nat`，则必须在生成 RT 前另行冻结一个自然状态的 direct-edit proposal set `N_b_nat`，只平均从这张自然图直接出发的编辑。所有方法共用同一候选集合，不能根据模型结果或目标物理效果选择候选。

禁止事后挑最大的 $A_{\mathrm{path}}$，也不能只保留最有利于 full 的编辑。按 $A_{\mathrm{path,loc}}$ 的 zero、low、medium、high 四档，报告 full 相对 Endpoint、Alignment-only 和 Response-only 的定位差值。这样才能检验 V4.1 的主线，也就是收益是否集中在路径相关区域，同时避免把无关转换的属性贴到当前定位样本上。

### 9.3 LoS、NLoS 和几何可辨识性只作辅助分析

LoS 表示基站与接收机之间有直达路径，NLoS 表示没有直达路径。NLoS 通常更依赖反射和绕射几何，但 LoS 不代表地图没用，NLoS 也不代表地图一定有用。它们是 P0 的辅助分析轴，必须在相同 $A_{\mathrm{path}}$ 档位内比较，不能用 LoS/NLoS 代替 path-incidence。

几何可辨识性 $I(x)$ 保留为 P1。它根据三维网格和基站位姿计算量化路径签名是否发生碰撞，用来检查高 $A_{\mathrm{path}}$ 样本的收益是否集中在物理上能够区分的位置。如果 $I(x)$ 没有通过预注册的相关性和碰撞审计，就删掉论文中“收益集中在可辨识区域”这句，不影响 P0 的路径机制结果。

## 10. 捷径、泄漏和负对照

### 10.1 每一类捷径都要有对应测试

| 风险 | 设计防线 | 必做测试 |
|---|---|---|
| 每个位置单独造地图，导致地图泄露位置 ID | 场景级 world bank 在全部位置间复用 | 比较旧 per-location bank 与新 bank 的 map-only 和 position probe |
| 以用户为中心裁图，裁剪方式泄露位置 | 使用固定的基站中心网格 | 单元测试 token 数量、顺序和 padding 是否与位置 $x$ 独立 |
| 某一节点永远是无编辑原图，模型只识别编辑状态 | 随机 anchor、统一 canonical render，并隔离自然图分支 | edit-status XOR baseline |
| 记住 world ID 或 edit ID | 整体留出未见 bank、edit 和 city | variant-ID matcher 可在训练侧高，但留出侧必须回落 |
| 只靠单一模态完成任务 | active 边四元组保证两类样本的单模态边际相同 | constant、CSI-only、map-only、scene-ID-only 基线 |
| action 字段直接携带结果或身份 | 不含 ID 的 signed grid | 扫描序列化字段，并运行 edit-only baseline |
| 目标地图或真实位置泄漏 | teacher 只读 CSI，主模型不读 $x$ | target bitwise test 和输入 allowlist 审计 |
| 目标 CSI $H_{v}$ 污染预测器输入 | 目标只作监督 | 替换 batch 中其他元素后，源分支输出必须不变；禁止跨支路共享 BN（批归一化）统计量或 attention |
| active/gray/null route 标签泄漏给模型 | route 只决定损失，不进入输入 | 审计模型输入 schema 和保存下来的 tensor |
| 目标城市信息泄漏 | 外层目标城市从 Stage 0 起完整留出 | 审计归一化、阈值、checkpoint、校准和梯度 |
| 人工编辑数据造成域偏移 | 四臂等量看到自然原图 | 报告自然正确图的域内误差和表征分布变化 |
| 把仿真噪声误当成地图编辑效果 | 配对世界共享物理扰动 $\xi_{\mathrm{phys}}$，两侧观测噪声 $\eta$ 独立，并另做重复仿真 | no-edit 或 identity 的噪声底审计 |

### 10.2 打乱真实配对，检查增益是否来自物理关系

这项负对照保留 V4.1 的证据链 A，并覆盖两个监督分支。在编辑类别、效果档位、场景和数据量都相同的条件下，分别打乱 alignment 中 CSI 与地图边的对应关系，以及 response 中动作与目标 CSI 的对应关系。单模态数据分布、损失形式、训练步数和效果分布保持不变。

如果 shuffled-pair 模型在真实留出编辑上仍获得同等 CGS、response 或跨城定位增益，说明性能不依赖真实物理配对，论文关于 paired supervision 的主张失败。

测试时 action swap 和训练时 shuffled pairing 不能互相替代。action swap 检查一个已经训练好的模型是否读取了正确动作；shuffled training 检查训练增益是否依赖真实配对关系。

### 10.3 最终编码器必须保留两种能力

训练完成后，丢掉临时任务头，只对冻结的最终编码器 $F_{\theta}$ 做五项检查：移除源地图；把正确源地图换成同一 active 边上的备选地图；运行固定预算 compatibility probe；运行固定预算 action-conditioned response probe；查看源地图的梯度或 attention，但它们只能作辅助可视化，不能作为决定性证据。

如果模型原生任务头表现好，而冻结 $F_{\theta}$ 后的统一探针表现差，结论必须降级为“即将被丢弃的任务头学会了任务”，不能声称最终表征获得了这项能力。

## 11. 要和哪些方法比较

### 11.1 最重要的比较是四臂实验

baseline 就是拿来比较的参照方法。CSI-PAIRS 最重要的参照，不是某篇外部论文，而是同一套实验里的四个版本：只做基础预测的 Endpoint、只加对齐监督的 Alignment-only、只加响应监督的 Response-only，以及两种监督都加的 CSI-PAIRS full。

这四个版本要用同一份数据、同一个骨干网络、同一个 teacher、相同训练步数、相同输入遮挡和输出查询、相同的源城市选模型预算，以及同一个下游定位头。每个版本实际用了多少 FLOPs 都要如实报告。Full 比任意一个单分支多一项监督，比 Endpoint 多两项监督，因此还要补一组计算量相同的单分支对照。外部方法做得再多，也不能代替这组四臂实验。

### 11.2 外部基线各自回答什么问题

| 对比方法 | 它在本文中负责回答的问题 | 论文里应怎样命名 |
|---|---|---|
| CSI-MAE | 只看 CSI 的遮挡重建方法能做到什么程度，也是 Stage 0 的预训练起点 | 写成 official-code adaptation，表示基于官方代码改到本文数据和划分上重新训练，不能把本文结果说成复现了原论文数字 |
| CSI-CLIP、CSI-CLIP++ | CSI 与 CIR 一致性学习的强对照。CIR 是 CSI 的另一种确定性表示，两者描述同一条信道 | 有官方实现时才直接用官方名称。自行实现的 ViT 版本写成 CSI-CLIP++-style controlled implementation |
| ContraWiMAE | 同时做重建和对比学习的预训练对照 | 用公开训练流程并按本文规则重新划分数据。没有官方权重时，不能写 official checkpoint |
| WWM | Wireless World Model，无线世界模型。一种在同一物理世界里，用已经配对的无线数据学习预测的方法，也是最接近本文出发点的思想基线 | 资源不足、没有完整复现时，写 WWM-inspired same-world matched prediction，不能写 faithful WWM |
| SigMap | 已经使用地图做定位，可用于错图现象和地图条件定位对照 | 写 paper-spec controlled implementation，并把目标城市微调预算统一到本文协议 |
| CSI-only map-free | 完全不看地图的负对照，用来检查提升是否真的需要地图 | 它本来就没有地图，因此 CGS 低不能被解释成它没有学会地图 grounding |
| Wi-GATr-inspired oracle-x | 把接收机真实位置告诉模型后的信息上界 | 单独标为 privileged-x，不能伪装成严格的 no-x 基线 |
| RFIR official-code adaptation | 场景修改与无线前向建模的邻近对照 | 只有共同物理读出方式和目标侧信息预算都说清楚后，才能比较 |

CSI-MAE、CSI-CLIP、CSI-CLIP++ 和 WWM 都能提供较强的 CFM（Channel Foundation Model，信道基础模型）起点，但它们主要在同一个物理世界，或同一信道的不同表示之间学习。CSI 与 CIR 可以互相确定转换，不等于地图真的改变后得到的两个物理世界。CSI-PAIRS 使用 RT 重新追踪地图编辑后的信道，这正是这些基线没有直接监督的部分。

related work 只保留三条清楚的差异。WWM、CSI-MAE 和 CSI-CLIP 系列主要学习同一世界或同一信道的不同表示，CSI-PAIRS 则比较 RT 重追踪前后的两个场景状态。

SigMap 用地图做定位，却没有把编辑明确分成 active 和 null 的成对干预。Wi-GATr、WiSER 与 RFIR 会输入位置或场景信息做无线前向建模，本文主实验不输入接收机位置，检查的是训练结束后保留下来的地图条件表征。排序损失已经是成熟工具，不能把 ranking 本身写成创新。

投稿前还要重新检索一次最新邻近工作。除非有单独而充分的证据，不写 first，默认只写 we study 或 we introduce。

---

## 12. 统计结果怎样才可信

这一节规定数据怎样分、数字怎样算，以及检验按什么顺序进行。规则必须在查看目标城市结果前定好。

### 12.1 数据不能混着用

源城市用于训练和确定方法，目标城市用于最终跨城测试。所有数据都服从第 2.7 节的唯一权限表，不能临时再造一份叫 pilot、validation 或 calibration 的重叠数据。训练探针的数据不能被校准器使用，校准器也不能读取选择探针的数据。最终未见世界库和未见城市只能评测，不能参与训练、选模型或调阈值。

四臂要使用相同的探针结构、样本量、训练步数、选择次数和校准器类型。风险校准中 correct、active、gray、null 四类样本的比例，在 source-method-selection 完成后就冻结。这个比例定义了本文人工审计混合中的风险概率。没有真实部署比例或重加权证据时，不能把它说成自然部署中的失败概率。

### 12.2 不能把同一场景的许多样本当成许多次独立证据

主方法至少训练 3 个随机种子。四臂使用成对的初始化、数据顺序和 mask bank，让比较尽量只反映监督方式的差别。随机种子只表示训练算法本身的波动。

统计上真正独立的单位是最高层的 scene tile 或 world bank，不是同一场景里的位置、边、方向、遮挡块、patch 或样本对。同一个世界库里的所有兄弟世界、位置、边、正反方向和数据增强都必须留在同一个统计组里。

AUROC、NMSE 的分子和分母、AURC 以及定位结果，都先在每个世界库内部计算，再对世界库做等权平均。两个目标城市分别报告，不能用两个城市的数据声称方法对所有城市都显著有效。

主交互效应采用第 7.3 节的配对多层 bootstrap。每次都以完整世界库为单位重复抽样，不能打散库内样本；同时成对重抽训练 seed 和 $k=8$ 的标签抽样，而且四臂使用完全相同的抽样索引。论文同时报告效应大小、置信区间，以及提前登记的最小实际改善幅度。

null 样本和零路径组要做等效检验，判断误差是否小到与重复测量噪声没有实际区别。p 值大于 0.05 只能说明没有检测到差异，不能写成“已经证明等于零”。

### 12.3 所有门槛提前冻结

下面这些内容只能用 source-method-selection 确定，并在目标城市评测前存档：active 阈值、输入遮挡和输出查询集合、主指标、非劣界限、action-swap 覆盖率、teacher 一致率、oracle 差距、null 幻觉上限以及全部排除规则。

source-calibration-fit、source-calibration-selection 和 source-method-selection 必须互相分开。到了目标城市，不能重新拟合温度、logistic 校准器、支持范围阈值或风险阈值。还要报告目标样本有多少落在源城市校准范围之外，避免平均 ECE 看着正常，却掩盖了大量模型从未见过的情况。

每次跨城实验都要保存一份信息预算清单，逐项记录是否使用了地图、基站位姿、无标签 CSI、参考库、位置标签、归一化统计，以及是否允许更新参数。

### 12.4 主结论按固定顺序验收

检验顺序如下，前一层没有通过，后一层只能作为探索结果，不能继续当作主结论。

1. Alignment-only 先在对齐任务上胜过 Endpoint，Response-only 先在响应任务上胜过 Endpoint 和 copy。
2. Full 在对齐和响应各自的原生主指标上，都要通过单侧非劣检验。这里的非劣是指没有差到超过提前设定的容许范围。
3. Full 的综合效用 $J_{a}$ 要分别胜过 Alignment-only 和 Response-only。
4. 检验提前登记的对数误差效用交互。$\Delta_{\mathrm{int}}$ 的单个估计值大于 0 还不够。第 7.3 节的七道协同门必须全部通过，配对多层 bootstrap 置信区间的下界也必须高于 0，并且超过提前登记的最小实际效应，才能说联合训练超过简单相加。
5. 检查两个目标城市的 k=0 和 k=8 结果，不能出现超过容许范围的反向退化。

同一层有多个优越性或非劣性比较时，使用 Holm 校正，或提前登记的同步世界库聚类 bootstrap 区间，控制整组比较的误报率。

---

## 13. 主文准备放哪些图表

主文优先保留 4 张图和 2 张表。

1. **图 1：错图现象与 CGS。** 同时展示正确地图、active 配对替代图、null 编辑图、其他城市地图、结构被破坏的地图和空地图。至少比较两个模型。active CGS 与 null 过度区分要并排出现，避免只展示好看的 active 结果。
2. **图 2：CSI-PAIRS 方法。** 画出一个场景级超立方世界库怎样跨位置复用，以及同一条编辑边对应的 alignment quartet、完全一致的分支数据包、带方向编辑、只读 CSI 的目标和共享 F/P。
3. **表 1：四臂的对齐和响应能力。** 四个版本都用同一 CGS 探针和同一响应探针，另报模型自身的响应结果、null 安全性，以及计算交互效应前必须通过的指标。
4. **表 2：两个未见城市的定位与风险。** 单地图定位在 k=0、8、32、128 时报告误差中位数和 P90。成对候选风险在主文中只报告 k=0，并给出 AURC、ECE、Brier、NLL、可靠性置信区间、共同支持范围的覆盖率和支持范围外比例。只有源城市的 episodic 校准器已经提前完成，才能把有少量标签时的风险结果放到附表。
5. **图 3：2×2 交互与机械拼接。** 展示 $\Delta_{\mathrm{int}}$、参数量匹配和 FLOPs 匹配的独立拼接，以及不压缩表示但计算量约为两倍的上界对照。
6. **图 4：传播路径机制。** 按 $A_{\mathrm{path}}$ 分组展示匹配分数差、响应优势、null 幻觉，以及提前登记的定位版 $A_{\mathrm{path,loc}}$ 对应的定位差值。

完整捷径测试、旧版每个位置一个世界库的泄漏对照、把所有非对角样本都做排序的旧方案、全部 active、gray、null 分桶、不同 tokenizer、停止梯度实验、I(x) 与 LoS/NLoS 联合分析、更多基线、自然地图域偏移和外部干预放在附录。

---

## 14. 每个结论都要先赚到证据

下面的 C1 至 C13 是论文结论与证据的对应表。实验排进时间表，不代表结论自动成立。证据没做完或结果不支持时，摘要、引言和贡献列表都要同步删掉或降低语气，不能指望 rebuttal 再补。

| 编号 | 想写进论文的结论 | 至少需要哪些证据 | 没达到时怎么写 |
|---|---|---|---|
| C1 | 现有地图条件模型缺少局部反事实几何响应 | 至少两个模型都完成 active 和 null 错图揭示实验 | 只报告具体模型的现象，不扩大成整个领域的问题 |
| C2 | 模型可能把地图压缩成 scene-ID | 源城市留出位置上，scene-ID 输入能追平地图输入，而且替换地图与替换 ID 时模型响应一致 | 删除 scene-ID 解释，只写模型对局部几何不敏感 |
| C3 | CGS 测到了 active 配对条件下的相容性 | 四臂使用同一探针，逐条编辑边检查边际是否匹配，null 不被过度区分，并在未见世界库和未见城市上复验 | 只说它是本数据集上的探针任务，不称为通用 grounding 指标 |
| C4 | Alignment 的收益依赖真实物理配对 | Alignment-only 胜 Endpoint；只看单一模态或 ID 的捷径失败；打乱配对关系后收益消失 | 删除 Alignment 方法结论 |
| C5 | Response 能预测 RT 重追踪后变化的方向和幅度 | 不给目标 CSI 的 mask-cover 测试、物理和 latent 指标、copy、no-action、action-swap 对照，以及 teacher 和 readout 的资格检查 | 证据只支持 latent 时就只写 latent prediction，否则删除响应结论 |
| C6 | 几何和响应信息留在最终保留的 F 中 | 去掉源地图、交换地图、四臂统一 frozen probe，并确保下游只读取 F | 只能说训练时临时使用的 head 学会了任务，不能说最终表征学会了 |
| C7 | 两种监督的联合价值不是机械拼接或单纯增加计算 | 第 7.3 节的七道协同门全部通过，细则见表后 | 只有交互门没通过时，只能写两项任务互补；Full 没胜过两个单分支或公平控制没通过时，取消 CSI-PAIRS 主标题 |
| C8 | 方法改善无标签或少标签的跨城定位 | 在两个目标城市严格限制可用信息，k=0 和 k=8 都达到提前登记的实际改善幅度 | 删除定位提升和 label-efficient 结论 |
| C9 | 在比较候选集合冻结后，k=0 的配对审计样本中可以预测定位何时失败 | 风险门的全部条件通过，细则见表后 | 删除“经过校准的风险”和“知道何时失败”，也不能拿 $q_{\mathrm{comp}}$ 代替定位风险 |
| C10 | 方法收益带有传播路径机制的一致信号 | 按 $A_{\mathrm{path}}$ 分组、完成组间匹配检查、零路径组通过等效检验，并提前登记定位样本怎样聚合多条编辑边 | 只报平均性能，不解释机制 |
| C11 | 数据来自经过校准的 RT | 独立校准和验证中的四项统计都通过 RT 资格门 | 只写由模拟器定义，不能写 calibrated RT |
| C12 | 结论不只适用于一个模拟器 | 增加小规模真实受控干预，或使用独立 RT 引擎得到一致结果 | 全文限定为 simulator-consistent |
| C13 | 方法有首创性 | 投稿日前更新邻近工作检索，保存可核查证据，并由 LLM-as-judge（codex / claude-code / cursor）给出绑定裁决 | 删除 first，只陈述已经核实的差异 |

**C7 的七道协同门**

1. Full 的 $J_{a}$ 分别胜过 Alignment-only 和 Response-only。
2. Full 的 active CGS 相比 Alignment-only 达到提前登记的非劣界，也就是没有差到超过允许范围。
3. Full 的 active 完整信道、target-free CSI NMSE 相比 Response-only 达到提前登记的非劣界。
4. $\Delta_{\mathrm{int}}$ 的配对多层 scene-bank bootstrap 置信区间下界高于 0，而且超过提前登记的最小实际效应。只看 $\Delta_{\mathrm{int}}$ 的单个估计值大于 0 不算通过。
5. 两个目标城市的 k=0 和 k=8 都没有出现超过允许范围的反向退化。
6. Full 的 $J_{a}$ 分别达到对 equal-FLOP Alignment-only 和 equal-FLOP Response-only 的提前登记实际优越界，排除优势只是来自更多 forward 或更新。
7. Full 同时胜过第 7.4 节中参数量匹配和 FLOPs 匹配的两个机械拼接对照。若同一个配置能同时匹配两项资源，也必须在参数量与 FLOPs 两项核算上都通过。

**C9 的风险门**

1. 比较候选集合提前冻结，主结果只看 k=0 的配对审计样本，四臂用共同支持范围作分母。
2. ECE、Brier、NLL 和可靠性置信区间通过提前登记的校准门。
3. 共同支持范围的覆盖率足够，而且 Full 的支持范围外比例不劣于两个单分支。
4. 所有提前登记评测点的 AURC 都胜过随机拒绝和只用 $u_{g}$ 的对照。随着覆盖率降低，保留下来的样本定位误差还要单调变小。

---

## 15. 什么时候继续，什么时候收缩结论

### 15.1 G0 至 G8 阶段闸门

Go 表示可以继续扩大实验，No-Go 表示当前证据不足，要先修问题或缩小论文结论。

| 闸门 | 达到什么条件才能继续 | 没达到时怎样处理 |
|---|---|---|
| G0 文献与资源 | 没有直接重合工作，RT、地图、真实实验或第二引擎的实施路径明确 | 缩小新颖性表述，严重重合时暂停项目 |
| G1 世界库资格 | 规范渲染通过，所有世界共用位置集 $X_{b}$，没有每位置地图或编辑状态泄漏，RT 资格明确 | 先修数据，不启动主训练 |
| G2 teacher 与 no-x | teacher 与物理变化的一致率和物理 readout 过门；不输入位置的遮挡预测胜过 copy、no-action 和 action-swap | 改用原始物理目标，或把研究转成相容性诊断 |
| G3 单分支成立 | Alignment-only 在 active CGS 上胜 Endpoint 且不误判 null；Response-only 在响应上胜 Endpoint 和 copy | 没成立的分支不能写进 Full 的主结论 |
| G4 Full 有联合价值 | 第 7.3 节七道协同门全部通过，包括原生任务非劣、胜原始与等 FLOPs 单分支、交互置信区间过门，以及胜机械拼接 | 降级为互补多任务，或只保留更强的单分支 |
| G5 跨城定位 | 两个目标城市的 k=0 和 k=8 都达到最小实际改善 | 删除少标签跨城定位结论 |
| G6 配对情境风险 | 在 k=0、四臂共同支持范围和冻结候选集合下，校准、AURC 与覆盖率三项闭环 | 删除“模型知道何时会失败” |
| G7 路径机制 | 匹配分数和响应优势随 $A_{\mathrm{path}}$ 增大，零路径组保持等效安全 | 删除机制解释 |
| G8 外部有效性 | 小规模真实干预或第二 RT 引擎在变化方向和 null 上一致 | 结论限定在模拟器一致范围内 |

### 15.2 已知风险和备用方案

| 可能出的问题 | 怎样看出问题已经发生 | 备用处理 |
|---|---|---|
| 错图现象不普遍 | 只在一个模型成立，或者所有模型都不成立 | 降低现象的重要性，把重点移到配对指标和方法 |
| teacher 感受不到物理变化 | teacher 空间和物理空间的一致率低 | 换成固定的原始物理目标，不能用 latent 结果掩盖问题 |
| 不输入位置时无法预测响应 | oracle-x 很强，而 no-x 接近 copy | 不能偷偷加入位置 x，改做概率响应，或回到相容性主线 |
| Alignment 只记住世界 ID | 训练集很高，未见世界库或城市明显下降 | 增加独立场景，随机选择 anchor，重做世界库和数据划分 |
| Response 只靠 action 和 CSI | 去掉地图后性能不降，交换地图也没有影响 | 缩小几何 grounding 结论，调整源信息融合，严重时停止 Full 路线 |
| null 样本出现明显幻觉 | null 误差超过噪声容许范围 | 增加 null 数据比例和 dead-zone 权重，检查人工编辑痕迹 |
| Full 干扰了单分支 | Full 在某个原生指标上明显退化 | 尝试梯度归一化和分阶段训练，仍失败就选一个主分支 |
| Full 只是两个效果相加 | Full 胜过单分支，但 $\Delta_{\mathrm{int}}$ 小于或等于 0，或者配对多层 bootstrap 置信区间下界没有同时高于 0 和提前登记的最小实际效应 | 只写 complementary multi-task，不写 synergy |
| Full 不胜机械拼接 | 资源匹配的特征拼接与 Full 持平或更好 | 承认共享训练没有额外价值，不能强推 PAIRS |
| CGS 提高但定位没有提高 | G5 没通过 | 不能用代理指标代替定位结果，检查信息是否留在 F 和定位头是否合理，再决定是否改成诊断论文 |
| 风险校准跨城漂移 | 大量目标样本落在源支持范围外，AURC 也没有改善 | 删除风险结论，另行研究跨域稳健校准 |
| 自然正确图上的性能下降 | 源城市或目标城市的正确地图误差明显变差 | 调整所有四臂共同使用的自然图分支比例，再完整重训四臂 |
| 工期超出预算 | 到 W5 时，优先级 P0 的证据仍没有补齐 | 改投更晚的会期，不能带着缺失主证据投稿 |

---

## 16. 八周执行安排

| 周次 | 本周工作 | 必须留下的结果 |
|---|---|---|
| W1 | 更新文献；在至少两个模型上跑错图揭示实验；确定地图网格、编辑类型、坐标协议和目标城市信息预算；完成 RT 资格检查 | 现象结果、数据规范和结论边界 |
| W2 | 在两到三个独立场景上做小型世界库；检查规范渲染、共同位置和泄漏；训练轻量 teacher 与 readout；做 no-x 和 oracle-x 的提前终止测试 | G1、G2 的初步决定。没通过就不扩大数据 |
| W3 | 批量生成场景级世界库 RT 数据；训练并冻结正式 teacher；重新检查物理变化与 teacher 变化是否一致 | 可审计的数据清单、正式分流规则和阈值 |
| W4 | 在源城市和未见世界库上训练严格四臂；完成单分支、捷径、action-swap 和打乱配对实验 | G3 决定、响应与 CGS 主表初版 |
| W5 | 在两个目标城市做 k=0、8、32、128 定位；完成交互和机械拼接对照 | G4、G5 的路线决定 |
| W6 | 训练统一探针；完成 $q_{\mathrm{comp}}$、$p_{\mathrm{fail}}$、风险覆盖率、路径参与度和信息保留检查 | 风险表、机制图以及 G6、G7 结果 |
| W7 | 补强外部基线；做小规模真实干预或第二 RT 引擎；完成优先级最高的 P1 项目 | 外部有效性证据和相关工作对照 |
| W8 | 整理主文图表、复现实验配置，逐条核对结论证据门并做内部红队审查 | 可以提交的论文版本 |

工作至少并行拆成三组：RT 与世界库、四臂与基线、统一探针与统计审计。W2 的初步闸门只决定要不要扩大数据。正式 G2 至 G7 必须用最终 teacher、正式世界库和最终模型检查点重跑。

---

## 17. 标题、摘要和贡献怎样写

### 17.1 工作标题

方法名已经冻结，论文标题还可以根据实验结果调整。当前工作标题是：

> **Do Map-Conditioned Wireless Models Use Geometry Counterfactually? CSI-PAIRS: Paired Alignment and Intervention-Response Supervision**

中文意思是：地图条件无线模型真的会按反事实方式使用几何信息吗？CSI-PAIRS 用成对对齐和干预响应监督来检验这个问题。

如果错图现象没有在至少两个模型上出现，就删除前面的领域级问题，改成直接介绍方法的标题。

### 17.2 中文摘要骨架

地图条件无线模型通常用正确配对的信号和场景训练，但正确地图上的性能好，并不能证明模型会对局部几何变化作出相应反应。我们用明确生效的 active 地图编辑和应当近似不生效的 null 编辑来检查这个缺口，并在跨位置复用的场景级世界库上提出 CSI-PAIRS。

对同一个接收状态和同一套无线配置，共享预测器一方面用零动作误差判断当前 CSI 更符合编辑边的哪一侧，另一方面根据不含身份编号的带方向编辑，预测 RT 重新追踪得到的目标信道。只有物理主路由判定为 active 的边才接受相对对齐监督，gray 和 null 不被当成强负样本。

实验使用严格的 2×2 四臂设计，在同数据、同骨干、同 teacher 和同下游头下比较 Endpoint、Alignment-only、Response-only 与 Full。评测包括固定预算的相容性识别、不读取目标 CSI 的信道响应、成对情境中的定位风险，以及两个未见城市上的无标签和少标签定位。

只有第 7.3 节的七道协同门全部通过，而且 $\Delta_{\mathrm{int}}$ 的配对多层 bootstrap 置信区间下界既高于 0，又超过提前登记的最小实际效应，摘要才可以写“联合监督产生了额外价值”。$\Delta_{\mathrm{int}}$ 的单个估计值大于 0 不够。没有真实干预或第二引擎审计时，结论必须限定在与模拟器一致的成对干预内。

### 17.3 预期贡献

实验完成前只能写“提出、建立、检验”，不能把还没有得到的结果写成事实。

1. **诊断和度量，**建立区分 active、gray、null 的错图与 CGS 协议，检查正确配对训练是否真的让模型对局部几何变化敏感。
2. **数据和方法，**建立跨位置复用、节点采样均衡、边和方向对称的场景级成对干预世界库，并让 Alignment 与目标响应监督共同训练同一个状态编码器和预测器。
3. **受控证据，**用同数据、同骨干、同 teacher、同训练步数和同源城市选择预算的四臂，再补等 FLOPs 单分支、交互、机械拼接、捷径、不读取目标的响应测试和信息保留审计，检查两种监督是否真的改善共享表征。
4. **下游和校准，**在不泄漏目标信息、以基站为坐标中心的协议下，检验两个未见城市的无标签和少标签定位、相对相容性，以及候选集合冻结后、只用源城市校准的定位失败风险。

---

## 18. 这项研究能说明什么，不能说明什么

1. 成对世界主要来自 RT。没有真实干预或第二引擎证据时，结论只能说与当前模拟器一致。
2. 主模型看不到接收机位置。只靠源 CSI 无法区分的信道碰撞会形成无法消除的条件方差。
3. 2.5D 的占用、高度和材质网格不能表示所有细小三维结构，也不能覆盖所有动态物体。
4. Frozen teacher 可能漏掉模拟器里真实存在的物理变化，因此必须同时公开 teacher 空间与物理空间的一致率，以及原始物理指标。
5. 场景级编辑只覆盖提前登记的变化类型，不能代表城市里所有地图过期和传播变化。
6. $q_{\mathrm{comp}}$ 只是 active 成对候选内部的相对概率。$p_{\mathrm{fail}}$ 的概率含义也只在源城市校准支持范围和指定的成对情境内成立。
7. 两个目标城市可以做严格留出验证，却不足以推出方法在所有城市都有效。
8. Full 的训练开销高于普通的同世界 Endpoint，不能写 training-cheap。它想减少的是定位标签需求，不是总计算量。
9. 定位是这篇论文唯一的主下游任务。结果不能直接外推到波束选择、无线感知、资源调度或整个 RAN（Radio Access Network，无线接入网）控制闭环。

---

## 附录 A：v6 为什么这样合并

这张表是内部决策记录，不进入论文正文。它保留每个版本取舍的理由，避免后面又走回已经确认有问题的方案。

| 原方案的问题 | v6 的处理 | 不处理会发生什么 |
|---|---|---|
| V4.1 给每个位置各做 K 张独立地图 | 一个场景级世界库跨全部位置复用 | 独特地图纹理可能直接暴露位置 ID，定位提升不能归因于几何理解 |
| V5 把超立方世界库称为 exchangeable | 改成节点采样均衡、双向边采样和边对称四元组 | 文字中的理论假设与实际采样不一致，边际匹配的论证不成立 |
| 超立方中存在唯一的 pristine 节点 | 随机选 anchor、统一规范渲染，并把自然图放入独立公共分支 | 模型可能用“是否被编辑”的 XOR 捷径直接猜标签 |
| 把所有 $H_{i}$ 与 $M_{j}$ 的交叉组合都当绝对负样本 | 只在固定物理条件、双重确认生效的 active 编辑边上做相对 Alignment | 信道碰撞以及 gray、null 样本会收到错误监督 |
| V4.1 的排序只把错误分支推远 | Response 直接回归 RT 目标 $z_{v}$、$y_{v}$ 及其变化量 | 模型可以向任意方向推开错误分支，无法声称预测了反事实变化 |
| V5 对 null 同时要求目标变化很小和变化量严格为零 | 在 latent 和物理空间都设置 dead-zone | 真实但小于阈值的变化会收到方向互相冲突的梯度 |
| Alignment 与 Response 可能只连接到两个临时 head | 对齐分数直接使用同一个预测器的零动作预测误差 | 共享骨干只停留在结构图上，无法回答机械拼接的质疑 |
| latent 物理损失会被 frozen readout 的误差带偏 | 预测器独立输出物理 patch，readout 只负责审计 | readout 从 $z_{v}$ 还原出的结果不等于 $y_{v}$ 时，latent 和物理目标会互相冲突 |
| Full 只与 Endpoint 比较 | 使用严格四臂、Pareto 非劣和交互效应 | 无法判断提升来自哪个分支，也无法证明联合训练有价值 |
| Full 胜过单分支就直接称为协同 | 增加 $\Delta_{\mathrm{int}}$、完整的七道协同门和资源匹配的独立拼接对照 | 简单加法会被误写成 synergy，很容易被审稿人否定 |
| 把 CGS 解释成任意地图的全局真假判断 | CGS 只在 active 成对候选范围内解释，null 单独报告 | AUROC 会被过度解释成任意位置上的物理可能性 |
| 把 $q_{\mathrm{comp}}$ 和定位不确定性混为一谈 | $q_{\mathrm{comp}}$ 与 $p_{\mathrm{fail}}$ 分别校准 | 会不会判断配对，被偷换成知不知道定位会失败 |
| 把 path-incidence 直接贴到定位样本 | 只聚合当前状态相连的编辑边；自然图使用提前冻结的直接编辑集合 | 平均无关编辑边，或事后挑最大变化，会人为制造机制曲线 |
| 用每个方法自己的原生 head 考自己的任务 | 四臂使用统一的相容性和响应探针 | 无法确认信息是否真的留在最终下游表征里 |
| active 和 null 损失直接在全局样本上求平均 | 先在各自 route 内做等权场景世界库平均 | 各城市 active、null 比例不同，会悄悄改变损失权重，结果不可比 |
| 成对世界复制同一份观测噪声 | 只共享物理外生条件 $\xi_{\mathrm{phys}}$，两侧观测噪声 η 独立，主结果用干净 RT | 模型可能预测人为共享的噪声，响应指标会虚高 |
| source pilot、validation、calibration 边界不清 | 设立七块互不重叠的源城市数据权限 | teacher、探针、校准器和阈值可能互相读取，造成循环泄漏 |
| k 没说明是位置数还是样本数 | k 表示不同位置的数量，每个位置只给一条观测，support 与 query 整组隔离 | k=8 可能被偷偷扩成 8K 条样本，支持集还可能进入测试集 |
| 风险校准器和候选 proposal 没有锁定 | 只根据地图侧信息生成候选，使用单调 logistic 校准、支持范围检测，并以 k=0 为主风险设置 | 用 RT 结果或 route 选候选，或微调后继续用旧校准器，都会得到虚假的好校准 |
| 独立拼接对照使用冻结的随机投影 | 提前声明可训练 bottleneck，并把它计入参数量和 FLOPs；另报不降维、计算量约两倍的对照 | 随机投影可能主动丢失信息，让机械拼接基线吃亏 |

最终固定的中心表述是：

> **Alignment 在确认生效的一条编辑边上，判断哪一侧更能解释当前 CSI；Response 说明沿这条边变化时，信道应往哪个方向移动、移动多少。CSI-PAIRS 让两种监督共用同一个零动作与干预预测器，训练结束后只保留二者共同更新的状态编码器做下游定位。**

论文需要保留的英文原句是：

> **Alignment identifies the better-explaining side of a verified intervention edge; response supervision identifies its displacement. CSI-PAIRS trains both through the same zero-action/intervention predictor and retains only their shared state encoder for downstream localization.**

---

## 全文术语表

这张表按第一次读研究方案时最容易卡住的地方整理。英文名要用于论文或代码时保留，阅读时先看右栏的中文意思。

### 无线通信与数据

| 术语 | 通俗解释 |
|---|---|
| CSI | Channel State Information，信道状态信息。可以把它理解成无线信号从发射端走到接收端后留下的一份详细记录 |
| CIR | Channel Impulse Response，信道冲激响应。它和 CSI 描述的是同一信道，只是表示方式不同，可以互相确定转换 |
| RT | Ray Tracing，射线追踪。模拟信号怎样经过直射、反射、绕射等路径到达接收端。本文用它在地图改变后重新计算信道 |
| CFM | Channel Foundation Model，信道基础模型。先从大量无线数据中学习通用规律，再迁移到定位等任务 |
| RAN | Radio Access Network，无线接入网。手机或终端通过基站接入移动网络的这一部分 |
| BS | Base Station，基站 |
| UE | User Equipment，用户设备或接收机，比如手机 |
| LoS、NLoS | Line of Sight 和 Non-Line of Sight。LoS 表示基站与接收机之间有直接视线，NLoS 表示直达路径被挡住，更多依赖反射或绕射 |
| 2.5D 地图 | 用平面网格记录占用、高度和材质。比纯二维地图信息多，但仍不是完整三维模型 |
| H | 一条完整 CSI，也就是模型看到的信道观测 |
| M | 地图输入 |
| x | 接收机的真实位置。主方法训练和推理时不把 x 直接告诉模型 |
| c | 无线配置，包括基站位姿、载频、阵列和天线等设置 |
| $\xi_{\mathrm{phys}}$ | 一对世界共享的物理条件。比较地图编辑前后时，这些条件保持不变 |
| η | 单次观测噪声。编辑边两侧独立抽样，避免模型靠复制噪声得高分 |
| AoA、AoD | Angle of Arrival 和 Angle of Departure，信号到达接收端与离开发射端时的角度。本文用它们辅助匹配编辑前后的传播路径 |

### 方法与训练

| 术语 | 通俗解释 |
|---|---|
| CSI-PAIRS | Paired Alignment and Intervention-Response Supervision，成对对齐与干预响应监督。PAIRS 也对应本文使用成对世界和成对编辑边 |
| Endpoint | 只做原有遮挡预测的基础训练版本，不加 Alignment 或 Response |
| Alignment-only | 在 Endpoint 上只加相对对齐监督，学习当前 CSI 更符合编辑边的哪一侧 |
| Response-only | 在 Endpoint 上只加响应监督，学习地图编辑后信道怎样变化 |
| Full | 同时使用 Alignment 和 Response 的完整 CSI-PAIRS |
| scene tile | 一块独立的场景区域，是数据划分和统计中的最高层单位之一 |
| scene-level world bank | 场景级世界库。同一块场景的多种地图状态组成一个库，并在许多接收位置上复用 |
| hypercube | 超立方结构。每个二进制位代表一种编辑开关，两个只差一个开关的世界由一条边连接 |
| node、edge | node 是世界库中的一个地图状态，edge 是只相差一个明确编辑的两个状态之间的连接 |
| paired intervention | 成对干预。固定位置和无线条件，只改变一个地图因素，再比较编辑前后的信道 |
| active | 物理 CSI 差异达到生效下界。teacher 敏感与否只作审计 |
| gray | 证据不一致或变化大小处于中间区域，不给强排序监督 |
| null | 两个空间都认为编辑影响接近重复测量噪声，只用来检查模型会不会凭空预测变化 |
| action、signed edit | action 是地图改了什么。signed edit 还记录从哪个世界变到哪个世界，因此带有方向 |
| zero-action | 假设地图不变的动作。它的预测误差被用来判断地图与 CSI 是否匹配 |
| branch bundle | 分支数据包。公平比较两侧时，源 CSI、配置、遮挡、查询和随机增强都完全复用 |
| mask、query | mask 是故意遮住的输入部分，query 是要求模型预测的输出位置。二者隔离可以防止直接抄答案 |
| teacher | 预先训练并冻结的教师模型。它只读取 CSI，用于判断变化是否在表征空间中也足够明显 |
| $T_{\mathrm{CSI}}$、$T_{A}$ | $T_{\mathrm{CSI}}$ 是只读 CSI 的 frozen teacher；$T_{A}$ 是把相容性分数转成概率时使用的温度参数 |
| readout | 把内部表征还原成可解释物理量的小模型。本文主要用它审计 teacher，不能代替真实物理目标 |
| encoder、F | 编码器把 CSI 和地图压成内部特征 F。训练结束后，下游定位只允许读取 F |
| predictor、P | 预测器读取 F 和动作，负责零动作预测与干预后的响应预测。下游定位不能调用它 |
| backbone | 四臂共同使用的主网络结构 |
| head | 接在主网络后的任务输出模块。临时 head 做得好，不代表最终保留的 F 也学到了信息 |
| $\mathcal L_{B}$、$\mathcal L_{A}$、$\mathcal L_{R}$ | 三类训练损失。$\mathcal L_{B}$ 是共同的基础预测损失，$\mathcal L_{A}$ 是 Alignment 损失，$\mathcal L_{R}$ 是 Response 损失 |
| NK | N 个位置乘以 K 个世界得到的原始信道数量。正文要求四臂看到同一批数据 |
| frozen | 参数冻结，只读取结果，不再根据当前实验更新 |
| probe | 探针。固定主表征后训练的一个小模型，用相同预算检查表征里有没有某类信息 |
| shortcut | 捷径。模型没有理解目标规律，却利用 ID、纹理、编辑痕迹或数据格式猜对答案 |
| leakage | 泄漏。本来不该给模型或评测流程的信息进入了训练、选模型或测试过程 |
| grounding | 模型的内部表示是否真的对应到地图几何和物理信道，而不只是记住场景标签 |
| latent | 模型内部的特征空间，不能直接当作物理真值 |
| physical target | 从 RT 或真实测量得到的物理目标，如完整 CSI 或指定 patch |
| dead-zone | 误差落在一个很小的容许区间内就不继续惩罚，适合处理接近零但并非严格为零的 null 变化 |
| no-x、oracle-x | no-x 不给接收机位置，符合本文主设定；oracle-x 把真实位置告诉模型，只作为信息上界 |
| copy、no-action、action-swap | 三个响应负对照：直接复制输入、不执行编辑、把正确编辑换成别的编辑 |
| MAE | Masked Autoencoder，遮挡自编码器。遮住输入的一部分，再训练模型恢复它 |
| JEPA | Joint Embedding Predictive Architecture，在特征空间预测被遮住或未观察部分的一类方法 |
| MLP | Multi-Layer Perceptron，多层感知机。这里主要作为结构简单、预算固定的探针 |
| ViT | Vision Transformer，把地图网格切成 token 后用 Transformer 处理的视觉骨干 |
| BN | Batch Normalization，批归一化。两条分支不能共享会把目标信息带回源分支的 BN 统计量 |
| ID、UNK | ID 是身份编号；UNK 是 unknown 的缩写，表示目标城市没有可学习新身份时使用的固定未知表征 |
| XOR | 异或规则。两个二进制状态不同时输出 1，相同时输出 0。本文用它检查模型能否只靠编辑状态猜答案 |
| official-code adaptation | 基于官方代码改到本文数据和实验协议上重新训练，不声称复现原论文的全部数字 |
| style controlled implementation | 没有完整官方实现时，按论文风格做的受控实现，必须明确它不是官方版本 |

### 评测与统计

| 术语 | 通俗解释 |
|---|---|
| CGS | 固定模型和探针预算后，检查最终表征能否在一条确认生效的 active 成对编辑边上，分清哪张候选地图更能解释当前 CSI。它只表示这种相容性是否能被固定预算的探针读出来，不是任意地图的全局真假分数 |
| compatibility | 相容性。某个 CSI 与某张候选地图在指定成对条件下是否互相匹配 |
| $q_{\mathrm{comp}}$ | 成对 active 候选内部的相对相容概率，只回答“这两张候选图里哪张更符合当前 CSI” |
| $p_{\mathrm{fail}}$ | 在冻结候选集合和源城市校准规则下，定位会失败的估计概率 |
| $A_{\mathrm{path}}$ | 路径参与度，取值从 0 到 1。越大表示越多接收功率经过被编辑表面，或因编辑出现、消失 |
| $A_{\mathrm{path,loc}}$ | 给定位样本用的路径参与度，只平均当前地图状态直接相连的编辑边 |
| I(x) | 位置 x 的几何可辨识性，用来检查不同位置是否会产生难以区分的传播路径签名 |
| AUROC | 衡量正负两类样本排序能力的指标。0.5 接近随机，越接近 1 越好 |
| NMSE | 归一化均方误差。预测误差相对真实信号能量有多大，越低越好 |
| SGCS | 比较预测变化与真实变化在方向和相对结构上是否一致的辅助指标 |
| AURC | 风险与覆盖率曲线下面积。模型逐步拒绝高风险样本后，剩余误差降得越快越好，数值通常越低越好 |
| ECE | 期望校准误差。模型说 80% 会失败的样本，实际是否大约有 80% 失败 |
| Brier | 概率预测与真实 0、1 结果之间的平方误差，越低越好 |
| NLL | 负对数似然。既惩罚预测错，也会重罚过度自信，越低越好 |
| L2 regularization | L2 正则，让过大的模型系数付出额外代价，降低少量校准数据造成的过拟合 |
| MAD | Median Absolute Deviation，中位绝对偏差。用它做标准化时，不容易被少数极端值带偏 |
| median、P90 | median 是中位数，一半样本误差低于它。P90 是第 90 百分位，能反映较差那一部分样本 |
| FLOPs | 浮点运算次数，用来近似衡量模型一次计算需要多少工作量 |
| SMD | 标准化均值差，用来检查两组样本在编辑大小、距离等条件上是否真的匹配 |
| seed | 随机种子。用不同种子重复训练，可以观察算法结果会不会随随机初始化明显变化 |
| bootstrap | 反复重抽样来估计结论有多稳定。本文使用配对多层抽样：在每个城市内重抽独立 world bank 索引，成对重抽训练 seed 索引，再成对重抽 k=8 的标签抽样索引；同一次抽到的这些索引由四臂共同使用 |
| CI、confidence interval | 置信区间，表示根据现有样本估计出的合理波动范围 |
| p 值 | 在某个零假设下观察到当前或更极端结果的概率，不等于“结论为真的概率” |
| equivalence test | 等效检验。用来证明差异小到低于实际关心的容许值 |
| non-inferiority | 非劣检验。检查一个方法是否没有比参照方法差到超过提前规定的界限 |
| Holm correction | 多重比较校正。一次做许多检验时，降低偶然出现假阳性的概率 |
| family-wise error | 一整组统计检验中至少出现一次假阳性的概率 |
| Pareto 非劣 | Full 在一个任务变好时，不能让另一个任务差到超过容许界 |
| 2×2 interaction、$\Delta_{\mathrm{int}}$ | 四臂交互效应，用来检查 Full 的收益是否超过两个单分支收益的简单相加。单个 $\Delta_{\mathrm{int}}$ 估计值大于 0 不够：第 7.3 节七道协同门必须全部通过，配对多层 bootstrap 置信区间下界还要高于 0，并且超过提前登记的最小实际效应，才能声称有正交互 |
| equal-FLOP | 让对照方法与 Full 的计算量相同，排除“只是算得更多”的解释 |
| retention | 信息保留检查。训练结束后只读 F，看学到的几何和响应信息是否还在 |
| calibration | 校准。让模型给出的概率与真实发生频率相符 |
| source support、outside support | source support 是源城市校准数据覆盖的特征范围。outside support 表示目标样本超出了这个范围，概率不应被当作同样可靠 |
| risk-coverage | 按风险从高到低拒绝样本，观察保留比例下降时定位误差是否同步下降 |
| k-shot、k=0 | k 表示目标城市提供了多少个不同位置的有标签样本。k=0 表示完全不给目标城市位置标签 |
| support、query | support 是少量可用于适配的目标样本，query 是只用于最终测试的样本，两组必须按完整位置隔离 |
| P0、P1 | P0 是决定论文能否成立的必做项，P1 是有价值但可以根据时间取舍的增强项 |
| RQ1 至 RQ5 | Research Question 1 至 5，也就是本文五个主要实验问题 |

### 数据权限和论文写作

| 术语 | 通俗解释 |
|---|---|
| source、target | source 是用于训练和定规则的源城市，target 是严格留出的目标城市 |
| source-encoder-train | 训练主编码器的数据 |
| source-probe-train | 训练统一探针的数据 |
| source-probe-selection | 选择探针配置的数据 |
| source-calibration-fit | 拟合概率校准器的数据 |
| source-calibration-selection | 选择校准器设置和阈值的数据 |
| source-method-selection | 选择方法、阈值和预注册规则的数据 |
| source-final-unseen-bank | 源城市内部最终留出的世界库，只做评测 |
| comparison set、proposal set | 给当前地图准备的比较候选集合。它必须只按地图侧信息提前生成，不能看目标 CSI、真实位置或模型结果后再挑 |
| calibrator | 把原始分数转换成概率的校准器。本文在源城市训练，到了目标城市冻结 |
| `Cal_risk` | 本文定位失败风险所用的校准器名称，输入相容性差值和定位头自身的不确定性 |
| Stage 0 | 正式 CSI-PAIRS 四臂训练前的共同预训练阶段 |
| manifest | 数据清单，记录每个场景、世界、编辑、位置、配置和文件来源，便于审计与复现 |
| claim | 论文想公开声称的结论。每个 claim 都要与相应证据绑定 |
| Claim-Evidence Gate | 结论证据门。证据没有通过时，对应结论必须删掉或降级 |
| Go、No-Go | Go 表示证据足够，可以继续下一阶段；No-Go 表示先修问题、换路线或缩小结论 |
| headline | 论文最主要的方法、实验或下游任务 |
| related work | 与本研究最接近的已有论文，以及本文与它们的准确差别 |
| rebuttal | 论文评审后的答辩阶段。主证据不能故意拖到这个阶段再补 |
| simulator-consistent | 结论与当前模拟器定义的物理变化一致，但还不能自动代表所有真实环境 |
| label-efficient | 只用很少目标位置标签也能完成任务 |
| training-cheap | 训练计算量很低。本文不能使用这个说法，因为 Full 明显多做了监督任务 |
| complementary、synergy | complementary 表示两项任务互相补充；synergy 表示联合收益超过简单相加，后者必须由正交互证据支持 |
