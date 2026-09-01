# 给 Codex 的第二意见审查提示词

用法：把下面代码块整段交给 Codex。清单是起点，不是标准答案；方法名单、数字、`all` 顺序都在清单里，这里不重复。

仓库根（本副本）：`C:\Users\22688\Desktop\ftr\CSI`  
配套：`docs/review/2026-09-01-问题清单-诚实审查.md`  
已有消融数字：`docs/results/实验数据.md`  
可执行包：`code/CSI-PAIRS-v2.0-server/formal_v2/`

---

```text
你是机器学习系统和科研软件的第二意见审查人。
独立审查当前打开的这个仓库。不要复述清单交差。清单写错的要反驳，没写到的要补。
先确认 HEAD、分支、是否 dirty、有没有 runs/ 和正式 .npz。
路径从你打开的仓库根算。这个 Windows 副本根是 C:\Users\22688\Desktop\ftr\CSI。
不要假设一定是 Autodl 或 /root/autodl-tmp/CSI。
不要用 GSD。这次只读、不改仓库。fixture 不能当论文证据。

可执行包根：code/CSI-PAIRS-v2.0-server/formal_v2/

必读（清单当对照，不是标准答案）：
- docs/review/2026-09-01-问题清单-诚实审查.md
- docs/results/实验数据.md
- Idea1-CSI-PAIRS-冻结版-零基础阅读稿-v6_VSCode兼容版.md
- code/CSI-PAIRS-v2.0-server/formal_v2/formal_cli.py（all 链）
- code/CSI-PAIRS-v2.0-server/formal_v2/formal_evaluation.py
- code/CSI-PAIRS-v2.0-server/formal_v2/formal_evaluation_streaming.py
- code/CSI-PAIRS-v2.0-server/formal_v2/formal_external.py
- code/CSI-PAIRS-v2.0-server/formal_v2/external_adapters/
- code/CSI-PAIRS-v2.0-server/formal_v2/formal_representation_baselines.py
- code/CSI-PAIRS-v2.0-server/formal_v2/configs/representation_baselines_v1.json
- code/CSI-PAIRS-v2.0-server/formal_v2/formal_controls.py
- code/CSI-PAIRS-v2.0-server/formal_v2/formal_wrong_map.py
- code/CSI-PAIRS-v2.0-server/formal_v2/formal_risk.py
- code/CSI-PAIRS-v2.0-server/formal_v2/formal_path.py
- code/CSI-PAIRS-v2.0-server/formal_v2/formal_scene_id.py
- code/CSI-PAIRS-v2.0-server/formal_v2/formal_claim_controls.py
- code/CSI-PAIRS-v2.0-server/paper_v2/main.tex

============================================================
作者要的东西
============================================================

仓库里已经接好的对比和消融，必须在同一份正式数据上跑完、一张表。
我们的方法必须是这张表上最好的（定位看米：赢过仓库里的对手，也赢过自己的单分支）。
只跑完但我们输了，不算交差。要赢就改配方、重跑受影响的格、用新数。
不要再加门槛。不要到网上另找 LocUNet，除非仓库声称接了但其实没接。
不要用别人论文里另一套实验的米数填本表。

============================================================
保证（作者口径，按这个审）
============================================================

我们的方法必须是 SOTA。阻拦这个的都得改。

- 同一份正式数据、一张主表，Full / CSI-PAIRS 必须最好（定位看米：赢过单分支，也赢过仓库里 C1 对手）。
- 只跑完但我们输了，不算交差。挡跑完的、挡登表的、挡宣布「我们赢了」的、让配方注定输的，都标成要修的问题。
- 要赢就改配方、重跑受影响的格、用新数。不要把旧 G5 FAIL 改口成 PASS。
- 不要编没跑出来的数。不要用别人论文里另一套实验的米填本表。
- 不要把测试绿、streaming 能启动写成方法赢了。
- 不要把 Windows 说成正式证据主机。
- 哈希、单 writer、子集对照该留。
- Full 输给单分支、k=8 崩、ridge 没用上、联合损失把表征拉坏，标成挡住 SOTA 的问题，并写怎么改配方。

============================================================
请怎么写回来
============================================================

用中文。从 formal_cli all 跟到底。每个判断落到文件和行，或写明本工作区看不到产物。
不要找几个 P0 就停。

先答：
1) 仓库里到底接了哪些对比/消融？有没有名存实亡？
2) 跑完卡在哪？按「死在第一份对比 CSV 之前」排序。
3) 已跑完的四臂为什么还上不了论文表？
4) 按现有配方，Full 为什么当不了第一？改哪里才可能赢？
5) 你同意、修正、推翻了清单里哪些判断？

然后给问题表：
ID / 严重度（P0=跑不完、表出不了门、或方法注定赢不了；P1=拖慢或藏数字；P2=脏）/
拦住了哪一段或挡住我们当第一 / 证据（文件:行）/ 该怎么修 / 和清单的关系（证实 / 修正 / 新增 / 反对）。

单独写「仍不能做的事」。不要把保证写成「现在旧数已经是 SOTA」。
最后给「修好、跑完、并且让我们的方法当第一」的改代码和开跑顺序。
除非用户另外说可以改，这次不要改仓库。
```
