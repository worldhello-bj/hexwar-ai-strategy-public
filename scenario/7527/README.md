# 想定 7527 · 城市攻防作战（2V2）—— 地图与编成

本目录是**实际比赛地图**（比赛服与练习服为同一份想定 7527），
离线复现与规则强化训练都用它：`engine/`（裁决）与 `train/`（训练环境）直接读这里的 JSON。

## 内容

| 文件 | 规模 | 字段 |
|---|---|---|
| `map_hexes.json` | 2721 格 | `elevation` 高程(m) / `cond` 地形 / `step` 通行等级 / `note` 道路 / `hex` 贴图索引 |
| `units.json` | 36 个单位 | `name`/`army` 编制、`typeid`（1 人员 2 车辆 3 空中 4 工事）、`data` 装备、`pos` 开局格、`blood` 兵力、`score` 权重分值 |
| `cities.json` | 3 个夺控点 | `name` 主要/次要、`pos` 所在格 |

行列数 100×18；红蓝各 18 个单位。`units.json` 已对齐**平台真实开局**
（`RoomStart.asp` 之后的 `Obj_Get.asp`），不是抓包时的中局状态。

## 数据来源与获取方式

来自官方**公开**练习服 `http://www.hexwar.cn/WARGAMEAI/`（同一想定在比赛服
`43.143.209.228/WARGAMEAI` 上逐字节一致）：

| 字段 | 接口 |
|---|---|
| 格高程/地形/通行/道路 | `Map_Get.asp`（客户端 `LoadCurrentCity()` 用的同一份数据） |
| 单位编制与位置 | `Obj_Get.asp` / `ObjAjax.asp` |
| 夺控点 | `City_Get.asp` |
| 列×行/格宽 | `User_Get.asp` 字段 6/7/32 |

采集与解析脚本见 `tools/`（`hex_transport_capture.py` 等）；解析产物
`snapshot/parsed/` 是本地完整采集目录（含原始响应），**不入库**。

## 权利与使用范围

* 地图与编成数据由原厂商（济南棋战 / hexwar.cn）生成，权利归其所有。
* 本目录**只包含互操作所需的事实数据**（格高程、单位位置等），
  **不含**官方前端代码/美术资源，**不含**比赛手册原文或其表格衍生物，
  **不含**任何会话 Cookie、账号或签名信息。
* 用途：离线复现、部队/院校教学、算法训练与研究。如权利人提出异议，删除本目录即可
  （`snapshot/` 里的本地采集不受影响）。
* 若你要在公开场合分发，请自行确认你与权利人之间的许可范围。

## 零依赖校验

```bash
py -3.13 tools/selftest_engine.py          # 读本目录跑引擎自检
HEXWAR_SNAP=scenario/7527 py -3.13 tools/hex_score_audit.py   # 计分对账
py -3.13 tools/publish_scenario.py --check # 本目录与本地采集是否一致
```

`manifest.json` 里有每个文件的 sha1、规模与**敏感扫描结论**（逐条断言无 Cookie/账号命中）。
