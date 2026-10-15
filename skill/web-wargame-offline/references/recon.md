# 探查与抓取

## 1. 会话流程（Classic ASP 站点尤其重要）

这类站点的"进入业务页面"往往是一条链，缺一步就返回会话失效标记（常见是 `-1`）：

```
GET  /                     # 拿会话 cookie
POST /login/               # 表单字段名照抄登录页的 name
POST /SetUp/SetUpSave.asp  # 选房（allow_redirects=False，自己跟）
GET  <302 Location>        # 跟随跳转
GET  /XXX/Login/Login_Transfer/?PortID=n   # 传输页，会下发业务 cookie
GET  /XXX/main.html        # 客户端主页
GET  /XXX/User_Get.asp     # 业务接口（第一步往往只需这一步）
```

**调试技巧**：业务接口只返回 `-1` / 空 / 登录页 HTML，就说明链没走完。
不要立刻怀疑账号，先把上面每一步的 URL、状态码、Set-Cookie 打出来。

## 2. 从登录页提取表单契约

```python
soup = BeautifulSoup(html, "lxml")
for form in soup.find_all("form"):
    print(form.get("action"), form.get("method"))
    for el in form.find_all(["input", "select", "textarea"]):
        print("  ", el.get("name"), el.get("type"), (el.get("value") or "")[:40])
```

同时检查页面里有没有 `md5`、`onSubmit`、隐藏 token。老系统常见是
**无 CSRF、无 md5、明文口令**——记录下来，但**不代表可以代登录他人账号**。

## 3. HAR 的接口聚合（先看谁在轮询）

```python
from collections import Counter
cnt = Counter()
for e in har["log"]["entries"]:
    u = e["request"]["url"].split("?")[0]
    cnt[u] += 1
for path, n in cnt.most_common(20):
    print(n, path)
```

判读方法：

- **次数接近"记录总数/两次轮询间隔"的路径 = 状态轮询接口**，它才是权威。
- 只出现 1 次的大响应体（几百 KB）= 地图/想定数据，值得单独落盘。
- 出现多次但响应体很小（几十字节）= 心跳/Cookie 查询，可忽略。

## 4. 落盘策略

| 内容 | 用途 | 是否可公开 |
|---|---|---|
| `.asp` 响应原始字节 | 桩数据、格式参考 | ❌ 含服务端数据 |
| 静态 JS/CSS/图片 | 前端镜像 | ❌ 版权归厂商 |
| 你自己的解析产物（结构化的地图/单位 JSON） | 引擎输入 | ❌ 仍是厂商数据 |
| 手册 PDF | 规则来源 | ❌ 不转载，让使用者自行获取 |
| 你的代码 / 文档 / 方法论 | 交付物 | ✅ |

**默认把上述 ❌ 全部写进 `.gitignore`。**

## 5. 批量探测面板类接口

单位详情、武器面板这类接口是**理解数值模型的金矿**。
把每种棋子各取一个代表，批量请求并抓取纯文本，就能反推出
"装备 ID → 名称 / 射程 / 弹药类型 / 装甲 / 权重 / 侦察范围"的权威映射，
**比从手册猜准确得多**。

```python
seen = {}
for oid, name in units:
    key = name
    if key in seen:
        continue
    seen[key] = fetch("ObjInfo.asp", {"ObjID": oid})
```

用**集合交集**消歧：若某武器名在 A 单位与 B 单位的 ID 列表中同时出现，
而两列表的交集只有一个 ID，就唯一确定了该 ID 对应的武器名。
用这个方法可以把"ID ↔ 名称"完全定死，零猜测。
