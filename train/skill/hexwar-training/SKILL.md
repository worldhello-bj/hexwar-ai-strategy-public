---
name: hexwar-training
description: 在「比赛初设」上做兵棋规则强化训练与评测。当需要跑自博弈/蒙特卡洛基线、训练神经策略（PPO/REINFORCE）、同种子对照评测、检查点续训、规则符合性门（训练前必须过）、或排查"训练了但没提升"时使用。适用于 train/ 自包含训练环境（含 engine 裁决、env 环境、configs 配置、run_train.py 入口）。关键词：强化学习、自博弈、PPO、蒙特卡洛、奖励、检查点、评测、规则门、想定 7527、比赛初设。不适用于"操控单个算子打一局"（那是 hexwar-operator 的职责）。
---

# 兵棋规则强化训练

## 何时用

- 在这套环境里**训练**：自博弈基线（MC）、神经策略（PPO/REINFORCE）、续训
- **评测**：同一批种子对比初始策略 vs 训练后策略，判断是否真有提升
- **开训前的规则门**：确认引擎/裁决表没被改坏（奖励来自规则，规则错=学错）
- 排查"跑了很久但没提升"（先看是不是样本量/奖励设计问题，不是训练器问题）

## 一、核心约定（先记住）

1. **唯一想定 = 比赛初设**：想定 7527「城市攻防作战（2V2）」，地图与编成在
   `../scenario/7527/`（2721 格 / 36 单位 / 3 夺控点，已对齐平台开局）。**没有 demo 样例**。
   配置不写 `snapshot` 时自动用它。
2. **先过规则门，再开训**：`run_train.py train` 默认先跑
   自检 + `verify_tables/rules/rules_extra/unfavorable` + `hex_score_audit`；
   任一不过就**拒绝开训**。这些校验断言的是"引擎与线上实测规则一致"。
3. **奖励口径默认 `verified`**：三项分值（夺控 主要80/次要50、剩余兵力、歼灭 Σ损失×score）
   都已平台实测，`verified` 才进奖励；`platform` 口径会把同一损失在净胜分里算两遍，只用于对账。
4. **`--allow-real-map` 必须显式给**：否则训练器会拒绝在真实想定上训练（防止把服务端数据
   当合成样例静默使用）。配置里已带 `allow_real_map: true`。
5. **别把席位当战绩**：开局红蓝分差固定（蓝 75 : 红 60），一局只约 11–14 次决策。

## 二、最小闭环

```bash
cd train
py -3.13 run_train.py check                                     # 副本是否与核心同步
py -3.13 run_train.py rules                                     # 规则门（必过）
py -3.13 run_train.py mc --config configs/smoke_realmap.json     # 3 局冒烟（比赛初设）
py -3.13 run_train.py train --algo ppo --config configs/ppo_realmap.json
py -3.13 run_train.py eval --config configs/eval_realmap.json \
        --checkpoint analysis/realmap/ppo160.json
```

`--dry-run` 先看命令行（含配置映射结果）；`--live` 让规则门加上线上样本对账。

## 三、配置（configs/*.json）

键即 CLI 参数（`snapshot` / `episodes` / `move_radius` / `max_seconds` / `max_steps` /
`scoring` / `seed` / `eval_seeds` / `checkpoint` / `report`，PPO 另有
`batch_episodes` / `hidden_dim` / `learning_rate` / `update_mode` / `ppo_clip` /
`update_epochs` / `buckets` / `baseline_offset`）。
**未支持的键会报错退出**（不静默忽略）；`note` / `_xxx` 当文档字段。

| 配置 | 用途 | 关键参数 |
|---|---|---|
| `smoke_realmap.json` | 冒烟（3 局） | `move_radius 4` |
| `mc_realmap.json` | 自博弈基线（60 局，可比 `docs/真实地图训练.md`） | `move_radius 4`、`eval_seeds [1000000,1000001]` |
| `ppo_realmap.json` | PPO 训练（160 局） | `move_radius 3`、`batch_episodes 8`、`update_mode ppo_clip` |
| `eval_realmap.json` | 同种子对照评测 | `seeds [1000000,1000001]`（检查点走命令行） |

## 四、怎么读结果

* 报告是 JSON（`--report` 指定），检查点记 `scoring_version` / `scoring_mode`；
  **口径或版本不一致会被拒绝加载**（避免旧口径的值被当新口径初值继续训）。
* 评测要拿"同一批种子"的对比：初始策略 vs 训练后策略的平均分差。
  历次基线的实测数字见 `docs/真实地图训练.md` §4（含"无可测量提升"的分析）。
* 环境枚举有快/慢两种口径（`move_radius`、`deep_validate`）；差分工具
  `run_train.py diff` 用来确认快口径与深度校验结果一致。

## 五、常见坑

| 症状 | 原因 | 处理 |
|---|---|---|
| `--allow-real-map` 报错 | 没显式允许真实想定 | 配置里加 `allow_real_map: true` |
| 训练很久没提升 | 单局决策少、开局席位决定胜负、样本量不足 | 先加局数/改奖励或课程，别先怀疑训练器（见 docs §4） |
| `check` 报漂移 | 主仓库改了 engine/env，train/ 没同步 | `py -3.13 tools/build_train_env.py` |
| 找不到快照 | 公开克隆没有 `snapshot/parsed` | 用 `../scenario/7527`（默认），或设 `HEXWAR_SNAP` |
| 检查点加载被拒 | `scoring_version` 不一致 | 重新训练，别混口径续训 |

## 六、边界

* 本环境**只做训练与评测**，不操控真实服务器；需要"指挥单个算子"用 `hexwar-operator` skill。
* 真实采集/逆向过程已封存到 `../archive/exploration/`（只读留档）。
* 地图数据的版权与使用范围见 `../scenario/7527/README.md`。
