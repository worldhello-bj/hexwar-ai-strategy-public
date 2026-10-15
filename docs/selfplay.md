# 双边比赛环境与终局计分训练

## 接口与计分

`env.episode_api.TwoPlayerEpisode` 在同一个游戏时钟上轮流接受 BLUE、RED 的动作。双方各自读取战争迷雾下的观察与可选动作，保留各自的最后目击记录。它是顺序决策模型；不代表双方同时下令。

玩家可以使用公开比分判断局面。公开比分作为观察输入，和奖励分开处理：中途奖励始终为零，正常终局才将引擎计算的本方总分作为本方奖励。没有额外的移动、攻击或其他动作奖励，也没有人工编写的战术偏好。

> 计分口径（2026-10-15 改写）：`remaining` 已按反推口径实现——**分值乘在兵力上**
> （`Σ上阵(score×blood) + Σ车内载员(score×编制)`，对 7527 开局精确命中平台 红 354 / 蓝 234）。
> 这使得**公开比分天然包含敌我单位血量的总量**（与平台计分页一致），而非逐单位泄露。
>
> ⚠️ **奖励口径隔离**：`total`（平台式三项和）与 `verified`（只用**已实测**分项）
> **分开两个数**，训练**默认用 `verified`**：只靠写明“未确认”不算隔离，
> 只要占位值进了 `total` 又拿 `total` 当奖励，它就还在参与学习。
> 需要平台式总分时显式 `--scoring platform`。口径与 `scoring_version` 写进检查点，
> 续跑不一致即拒绝（避免同一快照下混用两套奖励）。
>
> 2026-10-15 实测后：**三项分值都进了 `verified`**（夺控 80/50 一次性、歼灭 `Σ(损失×score)`），
> 所以两个数现在相等；隔离保留着，以便将来出现新的未实测项时**自动不进奖励**。
> 详见 `docs/计分规则待确认.md` §2.2/§2.4。

比赛时钟耗尽属于正常结算；外部步数预算耗尽属于截断，不给终局分、不用于更新学习表。自然结束优先于恰好同时到达的外部限制。

## 信息边界

- 每方收到独立 JSON 数据；不会收到另一方观察、引擎对象或调试日志。
- 动作菜单依据该方可观察信息生成，避免菜单本身泄露隐藏单位。
- 可观察到的对手棋子只公开规定的外观/位置/状态字段，不连带公开后台弹药、目标和载员列表。
- 玩家提交了界面允许的意图，但实际隐藏状态使其无法完成时，仍消耗相应尝试时间；不返回后台拒绝细节。这是新接口的明确规则，旧接口保持原行为。
- 公开计分规则可以透露聚合信息；这属于公开规则，不属于隐藏单位字段泄露。

## 学习基线

`env.selfplay` 使用无第三方依赖的首次访问蒙特卡洛算法。它记录本方观察与动作，结束后把本方原始终局分更新到经历过的状态—动作值中。探索在可选动作中均匀抽样；并列最大值也随机选择。两方学习表和经历分开保存。

这是精确观察表格基线，适合先验证训练闭环。它不具备神经网络的跨局面泛化能力；训练局数和表格更新不能直接等同于比赛实力提高。

## 云端运行

从仓库根目录运行：

```bash
python -m unittest discover -s tests -v
python tools/verify_rules.py
python tools/verify_tables.py
python -m env.selfplay --snapshot samples/demo --episodes 20 \
  --max-seconds 120 --max-steps 100 --seed 7 \
  --eval-seeds 1000000 1000001 \
  --checkpoint /tmp/hexwar-checkpoint-20.json \
  --report /tmp/hexwar-run-20.json
```

续跑时使用 `--resume` 并指定新的输出文件。检查点包含双方表格、已完成局数、配置及场景文件哈希；相同种子与配置支持复现。保留评测种子，不将其用于训练。

仓库自带 `samples/demo` 是 `tools/make_sample.py` 生成的比赛示例场景。这里运行的是离线重建引擎。评测报告应分别记录正常结束、外部截断和失败；对照赛需要另外安排随机对手、历史版本和换边，不能把双方自博弈的胜负直接当作“优于其他智能体”的证据。

## 随机对手对照赛

训练完成后固定检查点，与均匀随机合法动作对手交手；每个评测种子各执一方，使用对应席位的学习表，评测期间不更新参数。

```bash
python -m env.policy_eval --snapshot samples/demo \
  --checkpoint /tmp/hexwar-20.json \
  --seeds 1000100 1000101 1000102 1000103 \
  --report /tmp/hexwar-vs-random.json
```

报告按正常终局分数比较胜、平、负，外部截断单列。随机种子只改变随机过程，场景地图本身不变；这是该场景上的对照，不是跨地图泛化测试。

首轮实验文件在 `docs/experiments/`。20局检查点位于 `artifacts/selfplay/checkpoint-20-20261014.json.gz`；解压为 JSON 后传给 `--resume` 或 `--checkpoint`。相邻 metadata 文件记录未压缩文件 SHA256 与运行配置摘要。

100局完整检查点备份位于 `artifacts/selfplay/checkpoint-100-20261014.json.gz`，相邻 `checkpoint-100-20261014.metadata.json` 记录压缩前后 SHA256、文件大小、运行配置及合成场景哈希。备份采用无文件名、mtime 为零的确定性 gzip；解压为 JSON 后同样可传给 `--resume` 或 `--checkpoint`。这只是已完成训练的可恢复备份，不表示策略实力提升。

## 在真实想定（7527）上训练

上面的流程默认只接受**合成场景**（`snapshot_manifest` 要求快照带
`evaluation_provenance.json` 且 `origin == "original_synthetic"`）。要在**本地真实想定**
（`snapshot/parsed`，不入库）上训练，需**显式**加 `--allow-real-map`：

```bash
py -3.13 -m env.selfplay --snapshot snapshot/parsed --allow-real-map \
  --move-radius 4 --episodes 60 --max-seconds 1200 --max-steps 600 \
  --checkpoint analysis/realmap/mc60.json --report analysis/realmap/mc60_report.json
```

此时来源标记为 `origin="local_real_map"`（仍哈希三个被读取文件，运行可复现），
产物一律写 `analysis/`。**性能与局限见 `docs/真实地图训练.md`**：真实地图的动作枚举
原本要 1.8 s/次（每个候选一次整状态深拷贝），现已降到 0.042 s/次（×43）。
