# 完整公开地图与结构化坐标

旧观察接口只包含单位、目标点、比分和时间，没有完整地图。合法动作能透露一部分局部道路与耗时，但不能代替整张棋盘。

`PublicObservationAdapter` 是独立的、可选的第2版观察接口。它补充模拟场景的完整公开静态地图，同时保持原来的战争迷雾、合法动作和奖励规则。

## 提供什么

- 每个静态格子的字符串ID、数值行列坐标、高程、地形类别码、静态通行类别码及地图范围。
- 本方、当前可见对手、最后目击对手和目标点，按稳定ID组织为字典。
- 单位和目标点的结构化坐标；类别与标识符使用字符串，避免把它们误当成连续数值。
- 每个候选动作的原始JSON，以及独立的公开空间上下文。路径仍按实际顺序排列。

完整静态地图是这个离线模拟的公开规则。输出只包含白名单字段，不复制原始快照、后台单位、隐藏敌方位置、弹药或目标信息。通行类别码不表示任何单位此刻必然能够进入；真正的动作规则仍由环境裁决。

**公开分包的边界（2026-10-15 计分改写后）**：`score` 是**汇总值**，而平台的「剩余兵力分」本身就是公开的，因此敌我单位血量的**总量**会反在 `score` 里。这是刻意保留的公开信息，**不等于**泄露逐单位状态：敌方单位的身份、位置、弹药、压制与目标信息仍不出现在观察里，只有最后目击的位置（带 `stale` 标记）例外。

**`score` 的口径**：与奖励同一个口径（默认 `verified`）。三项分值（剩余兵力 / 夺控 / 歼灭）
**现已全部平台实测**，所以它**等于**平台计分页上的总分；两套口径（`verified` / `platform`）
现在数值相同，保留区分是为了将来出现新的未实测项时能自动排除（见 `docs/计分规则待确认.md` §2.2）。

## 使用

```python
from env.episode_api import TwoPlayerEpisode
from env.public_observation import PublicObservationAdapter

episode = TwoPlayerEpisode("samples/demo", max_seconds=1800, max_steps=1000)
view = PublicObservationAdapter(episode)
status = view.reset(seed=7)
side = view.current_actor
observation = view.observe(side)
candidates = view.candidate_views(side)

# 策略读取 observation 与候选项的 public_context。
# 提交的是原始 action，不提交上下文，也不改写动作。
status = view.step(candidates[0]["action"], side=side)
```

`schema_version` 为2。返回的数据可以安全修改而不会改变游戏状态。实体排序变化不会改写它们的身份，路径顺序仍有意义。

## 与已有模型的关系

已有表格和神经网络检查点保持原观察格式，尚未使用这个新适配器。本次没有用新地图输入重新训练，也没有声称补齐地图就提高了胜率。神经网络训练新增可选的 `--observation-schema 2`，分别编码静态地图、动态实体和候选动作，避免把320格地图简单拼入同一个哈希向量而稀释动态信息。新模式的训练与评测结果单独记录，见 `docs/neural-learning.md`。

## 已覆盖的回归

全图白名单覆盖、奇偶行坐标与邻接、返回值隔离、远端公开地形变化、隐藏状态不干扰（逐字段，仅汇总 `score` 除外——见上文「公开分包的边界」）、最后目击信息保持陈旧、实体顺序不变性、路径顺序敏感性、原始动作身份和可执行性，以及零中间奖励和原始终局计分。
