# 编码判据与修法

## 1. 先测字节，别信"看起来"

中文乱码有两种方向，**先分清方向再动手**：

| 现象 | 成因 | 修法 |
|---|---|---|
| `锛庛紪……`（一堆生僻汉字） | UTF-8 字节被按 **GBK** 解码 | 改成按 UTF-8 解 |
| `���` / `ȷ��`（替换符或西欧重音字母） | GBK 字节被按 **UTF-8/Latin-1** 解码 | 改成按 GBK 解 |
| `\uXXXX` 原样显示 | 转义没被解析 | 不是编码问题，是转义层 |

判据脚本：

```python
def probe(b):
    out = {}
    for enc in ("utf-8", "gb18030"):
        try:
            b.decode(enc); out[enc] = "ok"
        except Exception:
            out[enc] = "FAIL"
    if all(c < 128 for c in b):
        out["note"] = "纯 ASCII"
    return out
```

- 只有 utf-8 成功 → UTF-8
- 只有 gb18030 成功 → GBK
- **两个都成功** → 可能是纯 ASCII，也可能内容太短有歧义 → 用 `note` 字段判断

## 2. 同一站点会混用编码

实测过的情况：一个站点里

- 入口页 HTML：GBK
- 部分 `.js`（含中文提示语、注释）：GBK
- 部分 `.js`（纯 ASCII）：无所谓
- `.asp` 接口输出：GBK 字节

**所以桩服务不能全局声明一种 charset**，要**逐文件探测**：

```python
TEXT_EXT = {".html", ".js", ".css", ".txt", ".json"}

def detect_charset(b):
    if not b or all(c < 128 for c in b):
        return None          # 纯 ASCII：不声明
    try:
        b.decode("utf-8")
        return "utf-8"
    except Exception:
        return "gbk"

# 发送时：
ctype = MIME[ext]
cs = detect_charset(data)
if cs:
    ctype += "; charset=" + cs
```

**为什么不能统一声明 utf-8**：浏览器只要看到 `charset=utf-8` 就照做，
GBK 文件里的中文会立刻变成 `ȷ��`。而**不声明**时浏览器回退到文档编码，
反而常常是对的——但显式声明正确的编码更可靠。

## 3. `.asp` 输出必须与线上一致

Classic ASP 站点的接口**通常返回 GBK 字节**。桩服务要：

```python
return text.encode("gb18030"), "text/html; charset=gbk"
```

用 `gb18030` 而不是 `gbk` 更宽容（覆盖 GBK 扩展区）。

## 4. 与 jQuery 的交互

老代码常把选项写成 `datatype`（小写 t）——**这不是有效选项**，
jQuery 会忽略它并按 `Content-Type` 猜类型。后果：

| 桩服务发的 Content-Type | 客户端拿到 | 结果 |
|---|---|---|
| `text/html; charset=gbk` | 字符串 | ✅ 与线上一致 |
| `application/json` | 已解析的对象 | ❌ 回调里 `.split()` 崩 |

**结论：桩服务的 `.asp` 一律发 `text/html`。**

## 5. 本地文件与其编码

- 用 `open(p, "rb")` 读、`open(p, "r", encoding=...)` 写，**别依赖系统默认编码**。
- Windows 上 Python 默认编码受区域影响，处理中文时显式指定或设 `PYTHONUTF8=1`。
- PowerShell 里跑原生命令前先 `[Console]::OutputEncoding=[System.Text.Encoding]::UTF8`，
  否则日志里的中文全是乱码；**结果宁可写 UTF-8 文件再读**，不要靠 stdout。
- PowerShell 管道会把中文输出在 CJK 字符间插入空格（渲染伪影），
  **不要**据此判断文件内容被改坏，用 grep 复核。

## 6. 日期字段的类型陷阱

一个字段如果既当"是否已开始"的**真值**用（`if(!Time1)`），
又被拿去做**日期差**（`Date.parse(Time1)`），那它必须是**日期串**。

填 `1` 的后果：真值判断正常，但时钟显示 `NaN:NaN:NaN`。

排查方法：搜该字段的所有出现位置，看有没有 `Date.parse(x)` / `new Date(x)` /
`x.replace(/-/g,"/")` 这类用法。
