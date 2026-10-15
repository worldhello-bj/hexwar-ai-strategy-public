# -*- coding: utf-8 -*-
"""Hexwar 离线练习 —— 本地桩服务器。

用途：完全离线地跑起官方前端，数据来自 2026-10-14 抓取的对局快照。
不连接 www.hexwar.cn / cdn1.hexwar.cn。

用法：
    py -3.13 d:\\兵推\\offline\\server.py
    浏览器打开 http://127.0.0.1:8080/
"""
import os
import sys
import posixpath
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from security_helpers import confined_path, safe_log_target

ROOT = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(ROOT, "site")
MOCK = os.path.join(ROOT, "mock")
WEBROOT = "/WARGAMEAI7"
PORT = int(os.environ.get("HEXWAR_PORT", "8080"))
LOG = os.path.join(ROOT, "server_access.log")

# 裁决引擎开关：HEXWAR_LIVE=0 时退回纯静态快照
LIVE = os.environ.get("HEXWAR_LIVE", "1") != "0"
BRIDGE = None
if LIVE:
    try:
        import engine_bridge
        BRIDGE = engine_bridge.Bridge()
    except Exception as _e:  # 引擎不可用时退回快照模式
        BRIDGE = None
        print("[warn] 引擎加载失败，退回快照模式: %r" % (_e,))

MIME = {
    ".html": "text/html",
    ".js": "application/javascript",
    ".css": "text/css",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".json": "application/json",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
}

# 需要按文件真实编码声明 charset 的后缀
TEXT_MIME = {".html", ".js", ".css", ".txt", ".json"}


def detect_charset(b):
    """判断文本字节的真实编码：纯 ASCII 不声明，其余返回 utf-8 或 gbk。

    本站在线时部分 .js/.css 是 GBK（含中文的 jAlert、war.min.js 等），
    若无脑声明 utf-8，浏览器会把 GBK 中文字节解成乱码。
    """
    if not b or all(c < 128 for c in b):
        return None
    try:
        b.decode("utf-8")
        return "utf-8"
    except Exception:
        return "gbk"

LOGF = open(LOG, "a", encoding="utf-8", buffering=1)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOGF.write(s + "\n")


class Handler(BaseHTTPRequestHandler):
    server_version = "HexwarOffline/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # 用自定义日志

    def _send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = b""
        if n:
            try:
                body = self.rfile.read(n)
            except Exception:
                body = b""
        self._handle("POST", body)

    def _handle(self, method, raw_body=b""):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)

        log("%-4s %s" % (method, safe_log_target(self.path)))

        # 根路径
        if path in ("/", "/WARGAMEAI", "/WARGAMEAI/"):
            self.send_response(302)
            self.send_header("Location", WEBROOT + "/main.html")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        # 客户端登录跳转
        if path.rstrip("/") == WEBROOT + "/Login/Login_Transfer":
            return self._serve_file(os.path.join(SITE, "main.html"))

        rel = path
        if rel.startswith(WEBROOT + "/"):
            rel = rel[len(WEBROOT) + 1:]
        elif rel.startswith("/cdn1/"):
            rel = "cdn1/" + rel[len("/cdn1/"):]
        elif rel.startswith("/login"):
            return self._serve_file(os.path.join(SITE, "main.html"))
        else:
            rel = rel.lstrip("/")

        rel = posixpath.normpath(rel.replace("\\", "/"))
        try:
            site_path = confined_path(SITE, rel)
            mock_path = confined_path(MOCK, rel)
        except (ValueError, OSError, RuntimeError):
            return self._send(403, "forbidden", "text/plain")

        # 0) 裁决引擎优先接管动态接口
        if BRIDGE is not None and rel.lower().endswith(".asp"):
            try:
                post_body = raw_body.decode("utf-8", "replace") if raw_body else ""
            except Exception:
                post_body = ""
            try:
                res = BRIDGE.handle(rel, urllib.parse.parse_qs(parsed.query), method, post_body)
            except Exception as e:
                log("   !! 引擎异常 %s: %r" % (rel, e))
                res = None
            if res is not None:
                body, ctype = res
                return self._send(200, body, ctype)

        # 1) 桩数据优先（.asp）
        if rel.lower().endswith(".asp"):
            for cand in (mock_path, confined_path(MOCK, os.path.basename(rel))):
                if os.path.isfile(cand):
                    b = open(cand, "rb").read()
                    # 官方 ASP 输出为 GBK 字节，必须声明 gbk，否则客户端解码乱码
                    return self._send(200, b, "text/html; charset=gbk")
            log("   -> 无桩，返回空串")
            return self._send(200, "", "text/html; charset=gbk")

        # 2) 静态文件
        p = site_path
        if os.path.isfile(p):
            return self._serve_file(p)

        # 3) 兜底：以 / 结尾当目录
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "main.html")):
            return self._serve_file(os.path.join(p, "main.html"))

        log("   -> 404")
        return self._send(404, "not found: " + rel, "text/plain; charset=utf-8")

    def _serve_file(self, p):
        try:
            p = confined_path(SITE, os.path.relpath(p, SITE))
        except (ValueError, OSError, RuntimeError):
            return self._send(403, "forbidden", "text/plain")
        ext = os.path.splitext(p)[1].lower()
        ctype = MIME.get(ext, "application/octet-stream")
        try:
            b = open(p, "rb").read()
        except Exception as e:
            return self._send(500, "file unavailable", "text/plain")
        if ext in TEXT_MIME:
            cs = detect_charset(b)
            if cs:
                ctype += "; charset=" + cs
        return self._send(200, b, ctype)


def main():
    os.makedirs(SITE, exist_ok=True)
    os.makedirs(MOCK, exist_ok=True)
    log("=" * 60)
    log("Hexwar 离线练习服务器启动于 http://127.0.0.1:%d/" % PORT)
    log("站点目录: %s" % SITE)
    log("桩数据目录: %s" % MOCK)
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log("已停止")


if __name__ == "__main__":
    main()
