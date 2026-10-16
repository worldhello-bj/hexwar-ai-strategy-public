# -*- coding: utf-8 -*-
"""六角格几何：mapid ↔ (行,列)、邻接、距离。

地图为 pointy-top、奇数行右移半格的偏移布局（与客户端 LoadCurrentCity 一致）：
    x = HexWidth*col + (HexWidth/2 if row 奇数)
    y = (HexRowHeight/2) * row
mapid = 10000*(row//2) + 2*col + (row % 2)
"""

import math
from functools import lru_cache

HEX_W = 52
HEX_ROW_H = 90


@lru_cache(maxsize=65536)
def _split_mapid(mapid):
    b = mapid // 10000
    c = mapid % 2
    row = 2 * b + c
    col = js_round((mapid - 10000 * b) / 2) - (1 if c else 0)
    return row, col


def js_round(x):
    """复刻 JavaScript Math.round（.5 一律向上进位，Python 的 round 是银行家舍入）。"""
    return int(math.floor(x + 0.5))


def split_mapid(mapid):
    """mapid → (row, col)。

    ★ `lru_cache`（纯函数，无副作用）：实时训练下这个函数会被调上百万次
    （枚举邻格 / 测距 / 通视），1 Hz 实时训练的主要成本就包括这里的
    `make_mapid`/`split_mapid` 字符串↔整数往返（实测单局 190 万次调用）。
    """
    return _split_mapid(int(mapid))


@lru_cache(maxsize=65536)
def make_mapid(row, col, firstid=80014):
    """按客户端 LoadCurrentCity 的编号方式由 (row,col) 生成 mapid。"""
    b = row // 2
    c = row % 2
    return 10000 * b + 2 * col + c


def pos_of(mapid):
    """返回该格的像素坐标 (x, y)（与客户端一致）。"""
    row, col = split_mapid(mapid)
    x = HEX_W * col + (HEX_W // 2 if row % 2 else 0)
    y = (HEX_ROW_H // 2) * row
    return x, y


@lru_cache(maxsize=65536)
def to_cube(mapid):
    row, col = _split_mapid(mapid)
    # odd-r 偏移转立方坐标
    q = col - (row - (row & 1)) // 2
    r = row
    return q, r, -q - r


def distance(a, b):
    """两格之间的六角格距离（格）。"""
    aq, ar, as_ = to_cube(a)
    bq, br, bs = to_cube(b)
    return (abs(aq - bq) + abs(ar - br) + abs(as_ - bs)) // 2


@lru_cache(maxsize=65536)
def _neighbors(mapid):
    row, col = _split_mapid(mapid)
    if row % 2 == 0:
        offs = [(0, -1), (0, 1), (-1, -1), (-1, 0), (1, -1), (1, 0)]
    else:
        offs = [(0, -1), (0, 1), (-1, 0), (-1, 1), (1, 0), (1, 1)]
    out = []
    for dr, dc in offs:
        r2, c2 = row + dr, col + dc
        if r2 < 0 or c2 < 0:
            continue
        out.append(make_mapid(r2, c2))
    return tuple(out)


def neighbors(mapid):
    """邻格（返回新 list，调用方可以随意改；纯函数体走缓存）。"""
    return list(_neighbors(int(mapid)))


def hexid(mapid):
    """显示用行列号 RRCC。"""
    b = int(mapid) // 10000
    c = int(mapid) % 2
    d = "%02d" % ((2 * b) if c == 0 else (2 * b + 1))
    a = js_round((int(mapid) - 10000 * b) / 2)
    if c == 1:
        a -= 1
    return d + "%02d" % a
