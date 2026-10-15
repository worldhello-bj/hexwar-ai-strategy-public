"""比赛初设（scenario/7527/units.json）必须就是平台真实开局。

锁定住 2026-10 修掉的那类错：数据来自**中局**抓包（乘员已上车），
而"对齐开局"的脚本漏了 `sons`（车内单位数）⇒ 发布出去的初设里
装甲车 sons=1、火力支援车 sons=4，而真实开局恒为 0。

判据（都不依赖被封存的抓包，公开克隆里也能跑）：
* 36 个实体全部在场：`blood>0` 且 `pos>0`（没有谁被写成"已上车"）；
* **所有 `sons == 0`**（开局无人在车内）；
* 红蓝各 18，`pos` 都落在地图格里；
* 乘员 `blood` == 编制表（`engine.equip.CARRIED_ESTABLISHMENT`）。
"""
import io
import json
import os
import unittest
from collections import Counter

from engine import paths as _paths
from engine.equip import CARRIED_ESTABLISHMENT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITS = os.path.join(ROOT, "scenario", "7527", "units.json")
MAP = os.path.join(ROOT, "scenario", "7527", "map_hexes.json")


def load_units():
    with io.open(UNITS, encoding="utf-8") as fh:
        return json.load(fh)


class OpeningOrderOfBattleTests(unittest.TestCase):
    def setUp(self):
        self.units = load_units()
        with io.open(MAP, encoding="utf-8") as fh:
            self.mapids = {int(k) for k in json.load(fh)}

    def test_36_units_18_per_side(self):
        self.assertEqual(len(self.units), 36)
        c = Counter(u["color"] for u in self.units)
        self.assertEqual(c["BLUE"], 18)
        self.assertEqual(c["RED"], 18)

    def test_nobody_is_aboard_at_opening(self):
        """`sons` = 车内单位数；开局谁也没上车，必须全 0。"""
        bad = [(u["id"], u["army"], u["sons"]) for u in self.units
               if int(u["sons"]) != 0]
        self.assertEqual(bad, [], "开局不该有单位在车内（车内数 sons 应为 0）：%s" % bad)

    def test_every_unit_is_on_the_map(self):
        off = [u["id"] for u in self.units if int(u["blood"]) <= 0 or int(u["pos"]) <= 0]
        self.assertEqual(off, [], "这些单位开局不在场（blood/pos <= 0）：%s" % off)
        outside = [u["id"] for u in self.units if int(u["pos"]) not in self.mapids]
        self.assertEqual(outside, [], "这些单位的位置不在地图格里：%s" % outside)

    def test_crew_strength_matches_establishment(self):
        bad = [(u["id"], u["data"], u["blood"], CARRIED_ESTABLISHMENT[int(u["data"])])
               for u in self.units
               if int(u["data"]) in CARRIED_ESTABLISHMENT
               and int(u["blood"]) != CARRIED_ESTABLISHMENT[int(u["data"])]]
        self.assertEqual(bad, [], "乘员兵力与编制不符：%s" % bad)

    def test_published_copy_is_the_one_engine_reads(self):
        """公开克隆里没有本地采集，引擎会回退到 scenario/7527；确认这条路径生效。"""
        snap = _paths.snapshot_dir()
        self.assertTrue(os.path.isdir(snap))
        self.assertTrue(os.path.exists(os.path.join(snap, "units.json")))


if __name__ == "__main__":
    unittest.main()
