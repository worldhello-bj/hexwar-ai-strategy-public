# 规则强化训练环境（train/）

**这个环境只有一个想定：比赛初设。** 想定 **7527「城市攻防作战（2V2）」** 的真实地图、
编成与夺控点，随仓库公开在 [`../scenario/7527/`](../scenario/7527/README.md)
（2721 格 / 36 个单位 / 3 个夺控点，已对齐平台开局）。**不含任何合成 demo 样例**。

## 为什么叫"规则强化"

奖励 = 规则。这里所有分值、战果、压制、夺控判定都出自 `engine/`，
而 `engine/` 的裁决表与计分口径是**用线上真实数据对到零残差**的
（86 条实测裁决：战斗结果 / 修正结果 / 最终战果 三链路；计分三项分值均为平台实测）。
所以训练流程被设计成：

```
py -3.13 run_train.py train --algo ppo --config configs/ppo_realmap.json
        └─ ① 先过规则门（自检 + 4 份规则校验 + 计分对账）
           └─ ② 过了才开训（规则不对就别训——否则学的是错规则）
              └─ ③ 训练完用 eval 在同一批种子上对比初始策略与训练后策略
```

## 快速开始

```bash
cd train

py -3.13 run_train.py rules                      # 规则门（30/30 + 30/30 + 20/20 + 31/31 + 计分对账）
py -3.13 run_train.py selftest                    # 引擎自检
py -3.13 run_train.py mc --config configs/smoke_realmap.json    # 3 局冒烟（就是比赛初设）
py -3.13 run_train.py mc  --config configs/mc_realmap.json      # 60 局自博弈基线
py -3.13 run_train.py ppo --config configs/ppo_realmap.json     # 160 局 PPO
py -3.13 run_train.py eval --config configs/eval_realmap.json \
        --checkpoint analysis/realmap/ppo160.json               # 同种子对照评测
py -3.13 run_train.py check                       # train/ 与核心是否同步（防漂移）
```

`--dry-run` 只打印将要执行的命令行；`--live` 让规则门额外跑线上样本对账
（需要封存区 `../archive/exploration/` 里的样本，没有会自动跳过）。

★ **公开克隆里的用法一样**（`cd train` 后同样三条命令）。差别只是：公开仓不含生成副本，
入口会**自动改用仓库根的核心**（见下）。

### 规则门能保证什么、不能保证什么

* **能保证**：那些**已经被线上实测证过**的点（裁决表 30/30、规则 30/30、新增规则 20/20、
  不利修正、计分 354:234）**没有被某次改动静默改坏**；缺脚本 / 跑不动会**硬失败**。
* **不能保证**：引擎整体是对的。没有实测样本覆盖的地方（如表4 攻级 1–6/8、间瞄、
  反应装甲、终局结算）它一律不过问；它也不能代替你自己看数据。
* 换句话说：**规则门是一道回归网，不是正确性证明**。

它对外部可复跑的入口有两个（都比“相信维护者自觉”可靠）：

```bash
python tools/verify_scenario_units.py     # 初设数据不变式（CI 也在跑）
python train/run_train.py rules           # 训练入口的门（CI 也在跑）
```

CI 里还跑 `mc --config train/configs/smoke_realmap.json`（真跑 3 局）——
**“能训练”这句话由机器证，不由人（或 AI）声明。**

## 目录

| 路径 | 说明 | 是否入库 |
|---|---|---|
| `run_train.py` | **统一入口**（规则门 / 训练 / 评测 / 自检 / 漂移检查） | ✅ 手工维护 |
| `configs/*.json` | 训练配置（冒烟 / MC / PPO / 评测），全部指向比赛初设 | ✅ 手工维护 |
| `skill/hexwar-training/` | **给 AI 用的 skill**：怎么在这个环境里训练与评测 | ✅ 手工维护 |
| `engine/` `env/` `offline/` `tests/` `tools/` `docs/` | 与核心仓库同步的**自包含副本**（公开仓里没有 ⇒ 自动改用仓库根） | ⛔ 生成（见 `MANIFEST.json`） |
| `scenario/7527/` | **比赛初设**（想定 7527 地图与编成）的副本 —— 真源在仓库根 `../scenario/7527/` | ⛔ 生成 |
| `snapshot` | 指向主仓库 `snapshot/` 的数据目录联接（完整采集，不入库） | ⛔ 生成 |
| `analysis/` | 训练产物（检查点、报告） | ⛔ 生成 |

## 与主仓库的关系（重要）

★ **两种布局都支持，入口自己判别**（看 `train/engine` 与 `train/tools/verify_rules.py` 在不在）：

| 布局 | 何时 | 核心在哪 | 规则门跑谁的脚本 |
|---|---|---|---|
| **A 自包含副本** | 主仓库里跑 `tools/build_train_env.py` 后 | `train/` 自己 | `train/tools/verify_*.py` |
| **B 直接用仓库根** | 公开克隆（`train/` 里只有本文件与 configs） | 仓库根 | `../tools/verify_*.py` |

判别不出核心时**直接报错**；规则门遇到脚本缺失也算**不通过**（不会"跳过"后报成功——
那种静默假成功曾在公开克隆里把 5 项检查全跳过还宣布通过）。

`train/` 自包含形态的价值：可以整目录拷给别人、丢进 CI 跑，不受探索脚本影响。
代价是它与主仓库有重叠，所以用 `MANIFEST.json`（逐文件 sha1 + 源版本）管住漂移：

```bash
py -3.13 tools/build_train_env.py            # 在**主仓库**里同步（手工文件不会被覆盖）
py -3.13 tools/build_train_env.py --check    # 报告漂移
py -3.13 run_train.py check                  # 在 train/ 里做同一件事（布局 B 下说明"无需检查"）
```

> 为什么不是"直接 import 主仓库的 engine"：那样 `train/` 就不自包含了；
> 为什么不是"只留一份"：规则引擎必须只有一个真源（主仓库的 `engine/`）。
> 所以选"生成 + 清单 + 漂移检查"。**改了主仓库的 rules/env，记得重新同步。**

## 数据来源

* 想定数据：`../scenario/7527/`（公开练习服 `www.hexwar.cn` 的
  `Map_Get.asp` / `Obj_Get.asp` / `City_Get.asp` 解析结果；权利说明见该目录 README）
* 本地完整采集（含原始响应与会话产物）在 `../snapshot/parsed/`，**不入库**；
  它在就用本地，没有就用 `scenario/7527`（`engine/paths.py` 决定，`HEXWAR_SNAP` 可覆盖）

## 已知边界（别把这些当"训练失败"）

* 想定开局红蓝分差是**席位决定的**（蓝 75 : 红 60 起步），胜负别当战绩读。
* 一局只有约 11–14 次决策（真实地图机动很贵），单局信号弱 ⇒ 要看多局统计。
* `docs/真实地图训练.md` 记录了历次基线的实测数字与"没练出东西"的原因分析。
