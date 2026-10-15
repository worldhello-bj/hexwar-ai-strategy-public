"""计分口径回归测试（`engine/scoring.py`）。

用**原创合成 fixture**（不依赖真实想定数据）。覆盖：

1. `remaining` = `Σ上阵(score×blood)` + `Σ车内载员(score×编制)`（已反推并对账）；
2. **载员归属用稳定载具 ID**：载具机动后分值守恒；同格多辆载具按 `son`↔`ico` 配对；
   定不下来要显式报出；未知型号编制 = `None`（不按 0 静默计分）；
3. 下车后恢复阵营与兵力，分值不消失；
4. **已实测的夺控分**：主要 80 / 次要 50，一次性；未测到的名称不计分且报出；
5. **已实测的歼灭分**：`Σ(损失兵力 × score)`，与剩余兵力分逐点互补；
6. `total`/`verified` 两个口径 + `unverified` 清单 + 口径版本。
"""
import json
import tempfile
import unittest
from pathlib import Path

from engine import Game, Unit, geometry as G
from engine import scoring
from env.hexwar_env import HexwarEnv

CARRIER_HEX = G.make_mapid(10, 10)
BLUE_HEX = G.make_mapid(12, 12)
RED_HEX = G.make_mapid(9, 9)
FAR_HEX = G.make_mapid(11, 13)


def build(snapshot, *, extra_carrier=False, unknown_data=False, city_name="Synthetic"):
    """合成想定：蓝 1 个上阵单位 / 红 1 个上阵单位 + 1 辆载具 + 1 个车内载员。

    载具带 `son` 图标串、载员带 `ico`（与真实快照同形状），用于验证图标配对。
    """
    hexes = {str(G.make_mapid(r, c)): {"cond": 0, "elevation": 0, "step": 1}
             for r in range(6, 16) for c in range(6, 16)}
    units = [
        dict(id="blue", data=0, typeid=1, color="BLUE", blood=4, score=3, pos=BLUE_HEX),
        dict(id="red", data=0, typeid=1, color="RED", blood=2, score=5, pos=RED_HEX),
        dict(id="truck", data=0, typeid=2, color="RED", blood=3, score=10,
             pos=CARRIER_HEX, ico="ZBD100", son="PLA2-191,VUT5,UGVS"),
        # 载员：无阵营、pos=0、blood=0，只有 pos2 指向载具格；ico 命中载具 son
        dict(id="squad", data=382, typeid=1, color="", blood=0, score=3, pos=0,
             pos2=CARRIER_HEX, ico="PLA2-191"),
    ]
    if extra_carrier:
        # 同格第二辆载具，图标串里**没有**该载员的 ico ⇒ 必须能区分开
        units.append(dict(id="truck2", data=0, typeid=2, color="RED", blood=3,
                          score=10, pos=CARRIER_HEX, ico="M113TW",
                          son="COD-M4,COD-FK"))
    if unknown_data:
        units[3]["data"] = 99999
    (snapshot / "map_hexes.json").write_text(json.dumps(hexes))
    (snapshot / "cities.json").write_text(json.dumps(
        [{"id": "obj", "pos": G.make_mapid(13, 13), "name": city_name, "flag": None}]))
    (snapshot / "units.json").write_text(json.dumps(units))


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        build(self.root)
        self.game = Game(seed=1, snapshot=str(self.root))

    def make(self, **kwargs):
        kwargs.setdefault("scoring_mode", "verified")
        env = HexwarEnv(snapshot=str(self.root), **kwargs)
        env.reset(seed=1)
        return env

    def sides(self, game=None):
        return scoring.both_scores(game or self.game)

    def unit(self, uid):
        return self.game.units[uid]

    # ---------- remaining ----------

    def test_remaining_is_score_times_blood_plus_carried_establishment(self):
        both, _ = self.sides()
        self.assertEqual(both["BLUE"]["remaining"], 12)      # 3×4
        self.assertEqual(both["RED"]["remaining"], 49)       # 5×2 + 10×3 + 3×3

    def test_blood_attrition_moves_remaining_by_score_times_loss(self):
        self.unit("red").blood = 1
        both, _ = self.sides()
        self.assertEqual(both["RED"]["remaining"], 44)       # 49 − 5
        self.assertEqual(both["BLUE"]["killed"], 5)

    def test_total_is_invariant_under_attrition(self):
        both, _ = self.sides()
        total = both["BLUE"]["total"] + both["RED"]["total"]
        self.assertEqual(total, 61)
        self.unit("red").blood = 1
        both, _ = self.sides()
        self.assertEqual(both["BLUE"]["total"] + both["RED"]["total"], total)

    # ---------- 载员-载具关联 ----------

    def test_snapshot_carried_unit_is_linked_to_a_stable_carrier_id(self):
        self.assertEqual(self.unit("squad").detail["carrier"], "truck")
        self.assertEqual(self.unit("truck").detail["carried"], ["squad"])

    def test_carried_points_survive_carrier_movement(self):
        """载具开走（离开原格）后，载员分值**必须守恒**。"""
        both, _ = self.sides()
        before_red = both["RED"]["remaining"]
        before_total = both["BLUE"]["total"] + both["RED"]["total"]
        self.unit("truck").pos = FAR_HEX                 # 载具机动到别处
        after, issues = self.sides()
        self.assertEqual(after["RED"]["remaining"], before_red)
        self.assertEqual(after["BLUE"]["total"] + after["RED"]["total"], before_total)
        self.assertEqual(issues["carried_unlinked"], [])
        self.assertEqual(self.unit("squad").pos2, CARRIER_HEX)   # pos2 是旧格，但不参与计分

    def test_carrier_movement_through_env_action_conserves_score(self):
        env = self.make()
        before = env.score()
        env.game.units["truck"].pos = FAR_HEX
        after = env.score()
        self.assertEqual(after["own"]["remaining"], before["own"]["remaining"])
        self.assertEqual(after["enemy"]["remaining"], before["enemy"]["remaining"])

    def test_two_carriers_in_one_hex_are_disambiguated_by_icon(self):
        build(self.root, extra_carrier=True)
        game = Game(seed=1, snapshot=str(self.root))
        self.assertEqual(game.units["squad"].detail["carrier"], "truck")
        self.assertEqual(game.carried_ambiguous, [])
        both, issues = scoring.both_scores(game)
        self.assertEqual(both["RED"]["remaining"], 5 * 2 + 10 * 3 * 2 + 9)
        self.assertEqual(issues["carried_ambiguous"], [])

    def test_unlinkable_carried_unit_is_reported_not_silently_dropped(self):
        build(self.root)
        units = json.loads((self.root / "units.json").read_text())
        for entry in units:
            if entry["id"] == "truck":
                entry["pos"] = FAR_HEX          # 载具既不在 pos2 指的那格，图标也不匹配
        (self.root / "units.json").write_text(json.dumps(units))
        game = Game(seed=1, snapshot=str(self.root))
        both, issues = scoring.both_scores(game)
        self.assertEqual(game.carried_unlinked, ["squad"])
        self.assertEqual(issues["carried_unlinked"], ["squad"])
        self.assertIn("carried_unlinked", both["RED"]["unverified"])
        self.assertEqual(both["RED"]["remaining"], 40)      # 只剩 5×2 + 10×3

    def test_explicit_missing_carrier_never_rebinds_by_stale_position(self):
        self.unit("squad").detail["carrier"] = "missing"
        both, issues = self.sides()
        self.assertEqual(both["RED"]["remaining"], 40)
        self.assertIn("squad", issues["carried_unlinked"])

    def test_ambiguous_carried_identity_is_not_invented(self):
        build(self.root, extra_carrier=True)
        units = json.loads((self.root / "units.json").read_text())
        for entry in units:
            if entry["id"] == "truck2":
                entry["son"] = "PLA2-191"
                entry["color"] = "BLUE"
        (self.root / "units.json").write_text(json.dumps(units))
        game = Game(seed=1, snapshot=str(self.root))
        self.assertNotIn("carrier", game.units["squad"].detail)
        both, issues = self.sides(game)
        self.assertTrue(issues["carried_ambiguous"])
        self.assertEqual(both["RED"]["remaining"], 40)
        self.assertEqual(both["BLUE"]["remaining"], 42)
        for side in scoring.SIDES:
            self.assertIn("carried_ambiguous", both[side]["unverified"])
            self.assertEqual(scoring.side_score(game, side), both[side])

    def test_icon_evidence_is_consumed_once_across_passengers(self):
        build(self.root, extra_carrier=True)
        units = json.loads((self.root / "units.json").read_text())
        squad = next(entry for entry in units if entry["id"] == "squad")
        units.append(dict(squad, id="squad2"))
        (self.root / "units.json").write_text(json.dumps(units))
        game = Game(seed=1, snapshot=str(self.root))
        self.assertEqual(game.units["squad"].detail["carrier"], "truck")
        self.assertNotIn("carrier", game.units["squad2"].detail)
        self.assertTrue(game.carried_ambiguous)
        game.link_carried()
        self.assertNotIn("carrier", game.units["squad2"].detail)
        self.assertEqual(game.units["truck"].detail["carried"], ["squad"])

    def test_side_score_includes_unattributed_global_issues(self):
        self.unit("squad").detail["carrier"] = "missing"
        self.unit("squad").pos2 = FAR_HEX
        both, _ = self.sides()
        for side in scoring.SIDES:
            self.assertEqual(scoring.side_score(self.game, side), both[side])

    def test_verified_components_are_actually_filtered(self):
        from unittest.mock import patch
        self.unit("red").blood = 1
        with patch.object(scoring, "VERIFIED_COMPONENTS", ("remaining",)):
            both, _ = self.sides()
        self.assertEqual(both["BLUE"]["verified"], 12)
        self.assertEqual(both["BLUE"]["total"], 17)

    # ---------- 未知型号 / 编制来源 ----------

    def test_carried_points_die_with_carrier(self):
        """实测（`loadkill_20261015-223027`）：真装进车里的乘员**随车被歼**。

        平台数：RED 354 → **298**（= 车 10×4 + 乘员 4×4 = 40 + 16），蓝方歼煍 +56；
        之后 `Obj2_Out` 返回 `0`（放不出来）。本 fixture 对应：车 10×3 + 乘员 3×3 = 39。
        """
        both, _ = self.sides()
        self.assertEqual(both["RED"]["remaining"], 49)
        self.unit("truck").blood = 0                     # 载具被歼（3 血全打掉）
        both, _ = self.sides()
        self.assertEqual(both["RED"]["remaining"], 49 - 39)      # 车 30 + 乘员 9
        self.assertEqual(both["BLUE"]["killed"], 39)             # 歼煍分一并计入乘员
        self.assertEqual(self.unit("squad").detail["carrier"], "truck")   # 记录仍在（平台也不删）

    def test_carrier_death_ignores_carried_that_is_not_linked(self):
        """没关联上（`carried_unlinked`）的载员不能被当成“随车被歼”静默扣分。"""
        build(self.root)
        units = json.loads((self.root / "units.json").read_text())
        for entry in units:
            if entry["id"] == "truck":
                entry["pos"] = FAR_HEX
        (self.root / "units.json").write_text(json.dumps(units))
        game = Game(seed=1, snapshot=str(self.root))
        game.units["truck"].blood = 0
        both, issues = scoring.both_scores(game)
        self.assertEqual(issues["carried_unlinked"], ["squad"])
        # 载员没关联 ⇒ 它既不在 deployed 也不在 carried 里，**任何一方都不计它的分**
        # （并由 `unverified` 显式报出）；车体 0 血 ⇒ RED 只剩红方那个 5×2
        self.assertEqual(both["RED"]["remaining"], 10)

    def test_unknown_carried_establishment_is_none_and_reported(self):
        build(self.root, unknown_data=True)
        game = Game(seed=1, snapshot=str(self.root))
        self.assertIsNone(scoring.carried_establishment(99999))
        both, issues = scoring.both_scores(game)
        self.assertEqual(issues["carried_establishment_unknown"], [("squad", 99999)])
        self.assertEqual(both["RED"]["remaining"], 40)      # 未知型号**不按 0 计入**
        self.assertIn("carried_establishment_unknown", both["RED"]["unverified"])

    def test_establishment_provenance_is_always_flagged(self):
        both, _ = self.sides()
        self.assertIn("establishment_from_load_blood", both["RED"]["unverified"])

    def test_aboard_unit_counted_as_deployed_is_flagged(self):
        """引擎的中途装载不把载员 blood 置 0；这种口径差异必须报出来。"""
        both, _ = self.sides()
        self.assertNotIn("aboard_counted_as_deployed", both["BLUE"]["unverified"])
        self.unit("blue").detail["carrier"] = "truck"       # 假装蓝方步兵上了车
        both, _ = self.sides()
        self.assertIn("aboard_counted_as_deployed", both["BLUE"]["unverified"])

    # ---------- 下车恢复身份 ----------

    def test_unload_restores_color_and_strength_so_points_do_not_vanish(self):
        env = self.make(side="RED")
        before = env.score()["own"]["remaining"]
        _, _, _, info = env.step({"type": "unload", "unit": "truck"})
        self.assertFalse(info.get("invalid_action"), info)
        squad = env.game.units["squad"]
        self.assertEqual(squad.color, "RED")
        self.assertEqual(squad.blood, 3)                    # 编制表补回
        self.assertNotIn("carrier", squad.detail)
        self.assertEqual(env.score()["own"]["remaining"], before)

    # ---------- 已实测的夺控分 ----------

    def test_capture_points_use_measured_values(self):
        """实测：主要 +80 / 次要 +50（一次性）；**未测到的名称不计分**。"""
        self.assertEqual(scoring.capture_point_value("主要"), 80)
        self.assertEqual(scoring.capture_point_value("次要"), 50)
        self.assertIsNone(scoring.capture_point_value("Synthetic"))
        env = self.make()
        env.game.cities[0]["name"] = "次要"
        env.game.cities[0]["flag"] = "BLUE"
        self.assertEqual(env.score()["own"]["captured"], 50)
        env.game.cities[0]["name"] = "主要"
        self.assertEqual(env.score()["own"]["captured"], 80)
        env.game.cities[0]["name"] = "未知目标"
        self.assertEqual(env.score()["own"]["captured"], 0)
        self.assertIn("capture_point_name_unknown", env.score()["unverified"])

    def test_capture_bonus_counts_once_and_survives_losing_ownership(self):
        """实测：

        - 占住 60 s 不增长（一次性）；
        - **失守会把该点的分从原主扣除、重夺方加上**（实测：红 50 / 蓝 0）。
        """
        env = self.make()
        env.game.cities[0]["name"] = "次要"
        env.game.cities[0]["flag"] = "BLUE"
        first = env.score()["own"]["captured"]
        self.assertEqual(first, 50)
        env.game.second = 600                               # 十分钟后（模拟占住）
        self.assertEqual(env.score()["own"]["captured"], first)
        env.game.cities[0]["flag"] = "RED"                  # 归属被夺走
        self.assertEqual(env.score()["own"]["captured"], 0)
        self.assertEqual(env.score()["enemy"]["captured"], 50)

    # ---------- 已实测的歼灭分 ----------

    def test_measured_kill_points_complement_remaining(self):
        """实测：掉 1 兵力 ⇒ 输方 remaining −score、赢方 killed +score（diff 变化 2×score）。"""
        env = self.make()
        before = env.score()["diff"]
        env.game.units["red"].blood = 1                     # 掉 1 兵力（score=5）
        after = env.score()
        self.assertEqual(after["own"]["killed"], 5)
        self.assertEqual(after["enemy"]["remaining"], 44)
        self.assertEqual(after["diff"] - before, 10)
        self.assertEqual(after["own"]["verified"], after["own"]["total"])

    def test_verified_equals_total_now_that_all_three_are_measured(self):
        env = self.make()
        env.game.cities[0]["name"] = "次要"
        env.game.cities[0]["flag"] = "BLUE"
        env.game.units["red"].blood = 1
        report = env.score()
        self.assertEqual(report["own"]["verified"], report["own"]["total"])
        self.assertEqual(report["own"]["verified"],
                         12 + 50 + 5)                       # remaining + captured + killed

    # ---------- 口径与版本 ----------

    # ---------- 上车 / 下车（平台实测：上车 blood=0、按编制计分；下车按编制恢复）----------

    def test_load_keeps_score_but_zeroes_blood(self):
        """上车不改变计分（平台实测：装车那一刻计分页不变），但单位变成 blood=0。

        证据：`tools/hex_load_kill_probe.py` 里 `上车 ObjInto=True` 后计分页仍为 354/234。
        """
        units = json.loads((self.root / "units.json").read_text())
        for entry in units:
            if entry["id"] == "squad":          # 把乘员改成“开局同格部署”（平台真实开局）
                entry.update(color="RED", pos=CARRIER_HEX, blood=3)
                entry.pop("pos2", None)
        (self.root / "units.json").write_text(json.dumps(units))
        env = self.make(side="RED")
        before = env.score()["own"]["remaining"]
        _, _, _, info = env.step({"type": "load", "unit": "squad", "carrier": "truck"})
        self.assertFalse(info.get("invalid_action"), info)
        squad = env.game.units["squad"]
        self.assertEqual(squad.blood, 0)                     # 车内读不到兵力
        self.assertEqual(squad.establishment, 3)             # 编制留着
        self.assertEqual(env.score()["own"]["remaining"], before)   # 分值不变
        self.assertNotIn("aboard_damage_forgotten", env.score()["unverified"])

    def test_damaged_load_roundtrip_preserves_remaining_and_killed(self):
        units = json.loads((self.root / "units.json").read_text())
        for entry in units:
            if entry["id"] == "squad":
                entry.update(color="RED", pos=CARRIER_HEX, blood=3)
                entry.pop("pos2", None)
        (self.root / "units.json").write_text(json.dumps(units))
        env = self.make(side="RED")
        env.game.units["squad"].blood = 1
        before = env.score()
        for action in ({"type": "load", "unit": "squad", "carrier": "truck"},
                       {"type": "unload", "unit": "truck"}):
            _, _, _, info = env.step(action)
            self.assertFalse(info.get("invalid_action"), info)
            after = env.score()
            self.assertEqual(after["own"]["remaining"], before["own"]["remaining"])
            self.assertEqual(after["enemy"]["killed"], before["enemy"]["killed"])
            self.assertIn("damaged_transport_unverified", after["unverified"])
        self.assertEqual(env.game.units["squad"].blood, 1)
        self.assertEqual(env.game.units["squad"].establishment, 3)
        self.assertNotIn("carrier", env.game.units["squad"].detail)
        env.step({"type": "load", "unit": "squad", "carrier": "truck"})
        env.game.units["truck"].blood = 0
        after = env.score()
        self.assertEqual(after["own"]["remaining"], 10)
        self.assertEqual(after["enemy"]["killed"], 39)

    def test_loaded_unit_dies_with_carrier_after_engine_load(self):
        """引擎自己的 `load` 也要跟平台一致：车毁 ⇒ 乘员一并被歼（分扣除并记给对手）。"""
        units = json.loads((self.root / "units.json").read_text())
        for entry in units:
            if entry["id"] == "squad":
                entry.update(color="RED", pos=CARRIER_HEX, blood=3)
                entry.pop("pos2", None)
        (self.root / "units.json").write_text(json.dumps(units))
        env = self.make(side="RED")
        env.step({"type": "load", "unit": "squad", "carrier": "truck"})
        env.game.units["truck"].blood = 0                   # 载具被歼
        report = env.score()
        self.assertEqual(report["own"]["remaining"], 10)    # 只剩红方那个 5×2
        self.assertEqual(report["enemy"]["killed"], 39)     # 车 30 + 乘员 9

    def test_measured_tables_and_version_are_explicit(self):
        self.assertEqual(scoring.CAPTURE_POINT_SCORE, {"主要": 80, "次要": 50})
        self.assertEqual(scoring.PENDING_RULES, ())      # 三项 + 夺控边界都已实测
        self.assertEqual(scoring.VERIFIED_COMPONENTS,
                         ("remaining", "killed", "captured"))
        self.assertIsInstance(scoring.SCORING_VERSION, int)
        self.assertGreaterEqual(scoring.SCORING_VERSION, 5)   # v5 = 载员随车被歼的双重实测
        self.assertEqual(HexwarEnv.CAPTURE_POINT_SCORE, {"主要": 80, "次要": 50})
        self.assertEqual(HexwarEnv.capture_point_value("主要"), 80)

    def test_score_reports_version_mode_and_rules(self):
        report = self.make().score()
        self.assertEqual(report["scoring_version"], scoring.SCORING_VERSION)
        self.assertEqual(report["mode"], "verified")
        self.assertEqual(report["pending_rules"], list(scoring.PENDING_RULES))
        self.assertEqual(report["pending_rules"], [])
        self.assertIn("issues", report)


if __name__ == "__main__":
    unittest.main()
