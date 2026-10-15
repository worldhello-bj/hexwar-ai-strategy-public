# 从 PDF 手册重建裁决表

> ⚠️ 本文件只讲**方法论**。手册正文与表格的著作权属于原出版方，
> 不要把手册内容或整表誊抄进公开仓库；只保留实现互操作性所必需的功能性数值，
> 并在 `NOTICE` 里写明出处。

## 1. 先判断 PDF 是"文字版"还是"扫描版"

```python
import fitz
doc = fitz.open(pdf)
for i in range(doc.page_count):
    t = doc[i].get_text("text") or ""
    print(i+1, len(t), len(doc[i].get_images()))
```

- 每页字符数几百以上 → 有文本层，可直接抽。
- 字符数极少但图片数 > 0 → 扫描件，需要 OCR（另见本机的 markitdown / liteparse skill）。

**顺便算乱码率**：用字符白名单（ASCII + 常用汉字 + 中文标点）算"可疑字符占比"。
正常中文通常 < 3%，CID 字体缺 `ToUnicode` 的乱码页会 > 30%，阈值取 0.12 左右。

## 2. 纯文本抽取的列结构不可信

`page.get_text("text")` 会把宽表**按阅读顺序拉平**，多位数与空单元格全部错位。

两个可靠办法：

**A. 按词坐标聚类成行**

```python
words = page.extract_words()
rows = []
for wd in sorted(words, key=lambda w: (round(wd["top"],1), wd["x0"])):
    for r in rows:
        if abs(r["y"] - wd["top"]) < 6:
            r["ws"].append(wd); break
    else:
        rows.append({"y": wd["top"], "ws": [wd]})
```

**B. 用表格网格线（最准）**

```python
vs, hs = [], []
for e in page.edges:                       # 含 line / rect / curve 的边
    if e.get("orientation") == "v" and (e["y1"]-e["y0"]) > 3:
        vs.append(e["x0"])
    elif e.get("orientation") == "h" and (e["x1"]-e["x0"]) > 3:
        hs.append(e["y0"])
```

然后用 `(x0,x1,y0,y1)` 把每个词落进单元格。
注意：**有些页的表格线不在 `page.lines` 里**，必须用 `page.edges`，
必要时再补 `page.rects` 与 `page.curves` 的边。

## 3. 用领域不变量自检（关键一步）

抽完不要直接信。**先想清楚这张表在语义上必须满足什么**，再写断言：

| 表类型 | 应有不变量 |
|---|---|
| 攻击等级表（武器 × 距离） | 射程越远，等级不升高（通常递减） |
| 战果表（随机数 × 攻击等级） | 攻击等级越高，战果不降低（**单调不减**） |
| 修正表（高度差 × 距离） | 修正值应随高度差增大而更负 |
| 概率/结果表 | 行内不应无故出现"洞"（除非确有设计） |

实战例子：某战果表的列对齐无法唯一确定，但"攻击等级升高结果不应下降"
是硬约束。于是可以**按单调性反解列偏移** ——

```python
def violations(colmap, shift):
    v = 0
    for roll in rows:
        seq = [row(roll).get(colmap[a] + shift, 0) for a in attack_levels]
        v += sum(1 for i in range(1, len(seq)) if seq[i] < seq[i-1])
    return v

best = min(range(-2, 4), key=lambda s: violations(base, s))
```

## 4. 歧义消不掉时怎么办

**不要假装还原成功**。正确做法：

1. 把**原始抽取结果**（含坐标）落盘，保留证据
2. 把**择优结果**落盘，并在文件里写 `note` 说明哪里不确定
3. 在交付文档里**明确列出**"哪张表存在网格歧义、建议对照手册第 N 页校正"
4. 宁可给用户一个"结构正确、个别单元格待校"的引擎，也不要给一个"看起来完整但错了"的引擎

## 5. 把"叙事规则"与"数值表"分开处理

- **数值表** → 直接编码成数据文件（JSON），引擎读表。
- **叙事规则**（"某情况下减半""某状态持续 N 秒"）→ 编码成**判定函数**，
  并在注释里写清依据的条款位置，方便日后核对。

好处：数值可替换、规则可单测，两者互不干扰。

## 6. 版本差异

手册常有"某年某月更新"章节。**务必抓一次官方的更新说明**
（这类站点常在首页挂一个 `readme.txt`），把差异写进代码注释。
实测中更新说明会直接给出关键数值（例如某种地形修正的具体数值），
**比从表格里推断可靠得多**。
