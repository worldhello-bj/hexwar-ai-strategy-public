# -*- coding: utf-8 -*-
"""构建 / 校验「干净的规则强化训练环境」`train/`。

设计取舍（写下来免得以后又纠结）
--------------------------------
* **为什么要生成而不是手拷一堆副本**：训练环境必须**自包含**（能整目录拷给别人、
  丢进 CI、离线跑），但**规则引擎只能有一个真源**（仓库根的 `engine/`）。
  所以 `train/` 里与核心重叠的部分由本脚本从仓库根**同步**，并写
  `train/MANIFEST.json` 记下**逐文件 sha1 + 源版本**。
* **`--check` 报告漂移**（核心改了、train/ 没同步，或 train/ 里被人手改过）。
* **手工维护的文件不会被覆盖**：`README.md` / `run_train.py` / `configs/` / `skill/`。
* 不拷的东西（有意为之）：`offline/site|assets|mock`（官方前端镜像，版权归原厂商）、
  `docs/WARGAMEAI.pdf`（大二进制，仍以 `../docs/WARGAMEAI.pdf` 为准）、
  封存区 `archive/`、采集数据 `snapshot/`（训练时用 `--snapshot ../snapshot/parsed` 指过去）。

用法：
  py -3.13 tools\\build_train_env.py             # 同步（覆盖式重建生成物）
  py -3.13 tools\\build_train_env.py --check     # 只检查漂移，不改文件
  py -3.13 tools\\build_train_env.py --smoke     # 同步后顺手跑一遍引擎自检
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "train")

# 与核心重叠、由脚本同步的部分：(源, 目标, 白名单/None=整目录)
COPY_SPEC = [
    ("engine", "engine", None),
    ("env", "env", None),
    ("offline", "offline", ("server.py", "engine_bridge.py")),
    ("tests", "tests", None),
    ("tools", "tools", None),
    # ★ 不拷 `samples/`：那里是早期的**合成**测试夹具，只给主仓库单测用。
    #   训练环境的想定就是比赛初设 `scenario/7527/`（用户要求：不要其他 demo）。
    ("docs", "docs", "*.md"),
    # ★ 比赛初设（想定 7527 的地图与编成）已随仓库公开，直接拷进去 —— 
    #   训练环境的想定就是它，不依赖主仓库的 snapshot/。
    ("scenario", "scenario", None),
]
# 数据目录（联接进来，不算生成物/不参与漂移检查）
DATA_LINKS = ("snapshot",)
# 手工维护、绝不被脚本覆盖的文件/目录
AUTHORED = ("README.md", "run_train.py", "configs", "skill", "requirements.txt",
            "MANIFEST.json", ".gitignore")
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git"}
SKIP_EXT = {".pyc", ".pyo"}


def sha1(path):
    h = hashlib.sha1()
    with io.open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(65536), b""):
            h.update(blk)
    return h.hexdigest()


def wanted(name, whitelist, is_dir):
    if is_dir:
        return name not in SKIP_DIRS
    if os.path.splitext(name)[1] in SKIP_EXT:
        return False
    if whitelist is None:
        return True
    if isinstance(whitelist, str):
        if whitelist.startswith("*."):
            return name.endswith(whitelist[1:])
        return name == whitelist
    return name in whitelist


def iter_sources():
    """产出 (绝对源路径, train/ 下的相对路径)。"""
    for src_rel, dst_rel, whitelist in COPY_SPEC:
        src_dir = os.path.join(ROOT, src_rel)
        if not os.path.isdir(src_dir):
            continue
        for base, dirs, files in os.walk(src_dir):
            dirs[:] = [d for d in dirs if wanted(d, whitelist, True)]
            for name in files:
                if not wanted(name, whitelist, False):
                    continue
                full = os.path.join(base, name)
                rel = os.path.relpath(full, ROOT)
                yield full, os.path.join(dst_rel, os.path.relpath(full, src_dir))


def sync():
    files = {}
    for src, dst_rel in iter_sources():
        dst = os.path.join(TRAIN, dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        files[dst_rel.replace(os.sep, "/")] = sha1(src)

    # 依赖清单（训练用）
    req_src = os.path.join(ROOT, "requirements-neural.txt")
    req_dst = os.path.join(TRAIN, "requirements.txt")
    if os.path.exists(req_src):
        body = io.open(req_src, encoding="utf-8").read()
    else:
        body = "numpy\ntorch\n"
    io.open(req_dst, "w", encoding="utf-8").write(
        "# 由 tools/build_train_env.py 从 requirements-neural.txt 同步\n" + body)

    # 清掉 train/ 里过期的生成物（不在本次清单里的）
    for base, dirs, fs in os.walk(TRAIN):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS
                   and d not in AUTHORED]
        rel_dir = os.path.relpath(base, TRAIN)
        if rel_dir == ".":
            continue
        if rel_dir.split(os.sep)[0] in AUTHORED:
            continue
        for name in fs:
            rel = os.path.relpath(os.path.join(base, name), TRAIN).replace(os.sep, "/")
            if rel not in files:
                os.remove(os.path.join(base, name))

    rev = ""
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_rev": rev,
        "note": ("train/ 里与核心重叠的部分由 tools/build_train_env.py 同步生成；"
                 "手工文件（README/run_train.py/configs/skill）不在本清单内。"
                 "核心改了而 train/ 没同步时，--check 会报出来。"),
        "files": dict(sorted(files.items())),
    }
    io.open(os.path.join(TRAIN, "MANIFEST.json"), "w", encoding="utf-8").write(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    total = sum(os.path.getsize(os.path.join(TRAIN, r)) for r in files)
    print("同步 %d 个文件（%.1f KB）→ %s" % (len(files), total / 1024.0, TRAIN))
    print("源版本 %s" % (rev[:12] or "（非 git 检出）"))
    return 0


def check():
    mpath = os.path.join(TRAIN, "MANIFEST.json")
    if not os.path.exists(mpath):
        print("train/ 还没生成过：先跑 py -3.13 tools\\build_train_env.py")
        return 1
    man = json.load(io.open(mpath, encoding="utf-8"))
    files = man["files"]
    missing, changed = [], []
    for rel in sorted(files):
        src = os.path.join(ROOT, rel)
        dst = os.path.join(TRAIN, rel)
        if not os.path.exists(dst):
            missing.append(rel)
            continue
        if os.path.exists(src) and sha1(src) != sha1(dst):
            changed.append(rel)
    current = {os.path.relpath(os.path.join(b, n), TRAIN).replace(os.sep, "/")
               for b, d, fs in os.walk(TRAIN) for n in fs
               if not any(x in b.split(os.sep) for x in SKIP_DIRS)
               and not any(x in b.split(os.sep) for x in DATA_LINKS)
               and "analysis" not in b.split(os.sep)}      # analysis/ 是运行产物目录
    extra = sorted(r for r in current
                   if r not in files and r.split("/")[0] not in AUTHORED)
    print("train/ MANIFEST 版本 %s（源 %s）"
          % (man["generated_at"], (man.get("source_rev") or "")[:12]))
    print("- 缺文件 %d：%s" % (len(missing), missing[:5]))
    print("- 与核心不一致 %d：%s" % (len(changed), changed[:5]))
    print("- 多出来的生成物 %d：%s" % (len(extra), extra[:5]))
    if missing or changed or extra:
        print("⇒ 有漂移：跑 py -3.13 tools\\build_train_env.py 重新同步")
        return 1
    print("⇒ 一致：train/ 与核心同版本")
    return 0


def link_data():
    """把**有意不拷**的数据目录以联接/软链方式挂进来（不占空间、不会入 git）。

    `snapshot/` 是服务端数据（想定 7527 的解析结果），训练/自检都要用它，
    但属版权与体积问题不能进仓库 —— 所以在 train/ 里做一个指向源目录的
    **目录联接**（Windows junction，无需管理员；POSIX 下用 symlink）。
    """
    out = []
    for name in DATA_LINKS:
        src = os.path.join(ROOT, name)
        dst = os.path.join(TRAIN, name)
        if not os.path.isdir(src):
            out.append((name, "源目录不存在（跳过）"))
            continue
        if os.path.exists(dst):
            out.append((name, "已存在（跳过）"))
            continue
        if os.name == "nt":
            rc = subprocess.run(["cmd", "/c", "mklink", "/J", dst, src],
                                capture_output=True, text=True)
            out.append((name, "junction" if rc.returncode == 0 else "失败：%s"
                        % (rc.stderr or rc.stdout).strip()[:80]))
        else:
            os.symlink(src, dst)
            out.append((name, "symlink"))
    for name, r in out:
        print("数据目录 %-12s %s" % (name, r))
    return 0


def smoke():
    exe = sys.executable
    for rel in ("tools/selftest_engine.py",):
        os.makedirs(os.path.join(TRAIN, "analysis"), exist_ok=True)
        p = subprocess.run([exe, rel], cwd=TRAIN, capture_output=True, text=True)
        tail = (p.stdout or "").strip().splitlines()[-1:] or \
               (p.stderr or "").strip().splitlines()[-1:]
        print("smoke %-28s exit=%d %s" % (rel, p.returncode, tail))
        if p.returncode != 0:
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查漂移，不写文件")
    ap.add_argument("--smoke", action="store_true", help="同步后跑一遍引擎自检")
    args = ap.parse_args()
    if args.check:
        return check()
    rc = sync()
    if rc == 0:
        link_data()
        os.makedirs(os.path.join(TRAIN, "analysis"), exist_ok=True)
    if rc == 0 and args.smoke:
        rc = smoke()
    return rc


if __name__ == "__main__":
    sys.exit(main())
