# 给 Codex 的实现审查提示词（2026-09-01 改完之后）

用法：把下面代码块整段交给 Codex。这是审**这轮未提交改动**，不是再从零找问题。

仓库根：`C:\Users\22688\Desktop\ftr\CSI`  
分支：`artifacts/evaluation-bounded-evidence-20260901`  
对照清单：`docs/review/2026-09-01-问题清单-诚实审查.md`  
旧消融数字：`docs/results/实验数据.md`

---

```text
你是科研软件的第二意见审查人。只读、不改仓库。不要用 GSD。
工作区就是你打开的这个仓库根。本 Windows 副本是 C:\Users\22688\Desktop\ftr\CSI。
不要假设 /root/autodl-tmp/CSI。fixture 不能当论文证据。

先跑：git status --short、git diff --stat、HEAD、分支。
改动都还没提交。对照 docs/review/2026-09-01-问题清单-诚实审查.md 的 ID。
作者声称：挡跑完 / 挡登表 / 挡下一趟有机会赢 的代码已经改完。
你的任务是核实这句话，不是复述清单。

保证（按这个审）：
- 我们的方法必须成为同场正式表上的 SOTA。挡住这个的都得改。
- 不要把旧 G5 FAIL 改口成 PASS。不要编没跑出来的数。
- 不要用别人论文里另一套实验的米填本表。
- 不要把测试绿、streaming 能启动、改了配方写成方法已经赢了。
- 不要把 Windows 说成正式证据主机。
- 哈希、单 writer、子集对照、query 隔离该留。

作者说已经改了的点（逐条对代码，写「真改了 / 只改了一半 / 没改 / 改坏了」）：
- C15：formal_cli.py 不再调用 fcntl.*，锁走 formal_locks.flock
- C1/S1/S2：run-evaluation 和 all 默认 streaming，无 accepted.json 也不走会 OOM 的旧 evaluator
- C3：生产 batch_size 跟配置，不再冻死为 1（fixture/子集对照可仍为 1）
- C4：生产允许 1 或 2 张 GPU，不再死要求 cuda:0,cuda:1
- S14：--engineering-run 能在 dataset+config 解析后开工，并写下非 claim 标记
- S7/D7：--continue-on-stage-fail
- D1：GPU 独占只在有别的 compute 进程时硬停
- D9：compute plan 默认 advisory，--enforce-compute-plan 才硬校验
- B1/B4/B5/B9：科学 FAIL → FAILED 且仍导出数字；assemble-claims 写 descriptive_results；CLI 对 assemble-claims/all 走 assemble_claims_exit_code
- B2：G4 不再写死 NOT_ASSESSED
- B8：G5 仍是 4/4 AND 标签，分格必须能登
- C8/S12：localization.ridge 进定位头；k=8 步数封顶/早停；resource_control 同步
- S16：Full 联合损失不再只是静态等权加总
- A1/A2/S10：论文 tab:status 不再 Not authorized；32 格上米；对比列写未跑
- C14：registry 不再把未跑的 Wi-GATr/PMNet/SigMap 写成 executed
- F10：WWM 标 restricted，teacher 带位置没有被偷偷改掉
- C6：FROZEN_LEGACY_COMMIT 可被环境/receipt 覆盖，默认仍是 9850fff
- C7：死旗 --approve-full-experiment 已删

必读：
- git diff（至少 formal_cli.py、formal_claims.py、formal_factorial.py、formal_localization.py、formal_evaluation_identity.py、paper_v2/main.tex、external_adapters/all_map_adapters_v1.json）
- docs/results/实验数据.md
- docs/review/2026-09-01-问题清单-诚实审查.md §5 保证

用中文写回来。每个判断落到文件:行。

先答：
1) 作者声称已改的点，哪些是真的、哪些是半改、哪些没改或改坏了？
2) 现在 formal_cli all --engineering-run 还会死在第一份对比 CSV 之前吗？按顺序列还活着的卡。
3) 论文/claims 现在会不会把旧四臂藏起来，或把旧 G5 改口成 PASS，或把未跑基线写成已跑？
4) 新配方有没有改到足以「有机会」赢，还是只加了开关？有没有破坏 k=0 不更新头、单 writer、哈希？
5) 这轮 diff 有没有新的 P0（NameError、身份撒谎、默认又走 legacy、编数）？

然后给表：
ID / 裁决（已修 / 半修 / 未修 / 改坏 / 新增）/ 证据（文件:行）/ 还要不要改。

单独写「仍不能做的事」。
最后用三句话：能不能开正式机；先开什么命令；什么时候才许宣布 SOTA。
这次不要改仓库。
```
