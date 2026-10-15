# Simulator contract v8: static-unit eligibility audit

The v7 offline engine treated type-4 fortifications as mobile at 999 seconds per hex and allowed them to capture objectives. That produced an invalid +80 objective reward in the reviewed run. Pre-v8 learned rewards and comparisons that rely on this behavior are not evidence of valid performance. Start fresh training; the checkpoint-bound scoring contract is now 8.

The unchanged 7527 source has 36 units: 9 personnel (type 1), 15 vehicles (type 2), 9 aircraft (type 3), 3 fortifications (type 4). Exactly 33 are mobile types; exactly 24 satisfy unit-level ground capture eligibility at opening. Objective location and contested-zone checks still apply. Initial totals remain BLUE 234 / RED 354.

- Type 1: movement; capture unless existing artillery exclusions apply; load into own vehicle; equipped/ammunition-qualified grenade and attacks.
- Type 2: movement; capture unless artillery; unload actual passengers; equipped attacks.
- Type 3: movement; no capture, infantry load, vehicle unload or grenades; equipped attacks.
- Type 4: no movement or capture, infantry load, vehicle unload or grenades. Attack remains equipment-dependent; all three source bunkers have no weapons. Existing hide/stop behavior is unchanged: static does not by itself prove a universal prohibition on these actions.
- Unknown types: no invented movement speed or capture eligibility.

Raw unit `step` is 924/128/46 for mobile source types and 0 for bunkers. Its general units/semantics are not verified here; it is not used as a blanket `step == 0` ban. Map-cell `step == 0` remains the separate existing impassable-terrain rule.

The shared movement table excludes type 4. Guards cover step queries, committed steps, queued-path registration and resumption, environment candidate enumeration and explicit action submission, and the local bridge (which calls commit_step). Capture uses one shared eligibility predicate for engine application, candidate enumeration and explicit validation. Six new tests failed before correction and now pass, including the real Obj_100170 identity and all three source bunkers. No source scenario files changed.

## All opening units

These are initial-state advertised action kinds with radius 1, not all future actions. A missing capture action may simply mean the unit is not on an eligible objective.

| ID | Name | Side | Type | Raw step | Mobile type | Capture eligible | Advertised actions |
|---|---|---|---|---|---|---|---|
| Obj_100136 | 便携式反坦克导弹 | BLUE | 1 | 924 | True | True | hide,move |
| Obj_100137 | 便携式反坦克导弹 | BLUE | 1 | 924 | True | True | hide,move |
| Obj_100138 | 便携式防空导弹 | BLUE | 1 | 924 | True | True | hide,move |
| Obj_100139 | 坦克 | BLUE | 2 | 128 | True | True | hide,move |
| Obj_100140 | 步兵班 | BLUE | 1 | 924 | True | True | hide,load,move |
| Obj_100141 | 装甲车 | BLUE | 2 | 128 | True | True | hide,move |
| Obj_100142 | 坦克 | BLUE | 2 | 128 | True | True | hide,move |
| Obj_100143 | 步兵班 | BLUE | 1 | 924 | True | True | hide,load,move |
| Obj_100144 | 装甲车 | BLUE | 2 | 128 | True | True | hide,move |
| Obj_100145 | 坦克 | BLUE | 2 | 128 | True | True | hide,move |
| Obj_100146 | 步兵班 | BLUE | 1 | 924 | True | True | hide,load,move |
| Obj_100147 | 装甲车 | BLUE | 2 | 128 | True | True | hide,move |
| Obj_100148 | 自爆无人机 | BLUE | 3 | 46 | True | False | hide,move |
| Obj_100149 | 自爆无人机 | BLUE | 3 | 46 | True | False | hide,move |
| Obj_100150 | 自爆无人机 | BLUE | 3 | 46 | True | False | hide,move |
| Obj_100151 | 中型坦克 | RED | 2 | 128 | True | True | hide,move |
| Obj_100152 | 中型坦克 | RED | 2 | 128 | True | True | hide,move |
| Obj_100153 | 中型坦克 | RED | 2 | 128 | True | True | hide,move |
| Obj_100154 | 步兵班 | RED | 1 | 924 | True | True | grenade,hide,load,move |
| Obj_100155 | 无人察打车 | RED | 2 | 128 | True | True | hide,move |
| Obj_100156 | 火力支援车 | RED | 2 | 128 | True | True | hide,move |
| Obj_100157 | 旋翼无人机 | RED | 3 | 46 | True | False | hide,move |
| Obj_100158 | 巡飞弹 | RED | 3 | 46 | True | False | hide,move |
| Obj_100159 | 步兵班 | RED | 1 | 924 | True | True | grenade,hide,load,move |
| Obj_100160 | 无人察打车 | RED | 2 | 128 | True | True | hide,move |
| Obj_100161 | 火力支援车 | RED | 2 | 128 | True | True | hide,move |
| Obj_100162 | 旋翼无人机 | RED | 3 | 46 | True | False | hide,move |
| Obj_100163 | 巡飞弹 | RED | 3 | 46 | True | False | hide,move |
| Obj_100164 | 步兵班 | RED | 1 | 924 | True | True | grenade,hide,load,move |
| Obj_100165 | 无人察打车 | RED | 2 | 128 | True | True | hide,move |
| Obj_100166 | 火力支援车 | RED | 2 | 128 | True | True | hide,move |
| Obj_100167 | 旋翼无人机 | RED | 3 | 46 | True | False | hide,move |
| Obj_100168 | 巡飞弹 | RED | 3 | 46 | True | False | hide,move |
| Obj_100169 | 掩体 | BLUE | 4 | 0 | False | False | hide |
| Obj_100170 | 掩体 | BLUE | 4 | 0 | False | False | hide |
| Obj_100171 | 掩体 | BLUE | 4 | 0 | False | False | hide |
