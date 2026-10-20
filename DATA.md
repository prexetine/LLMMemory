# 数据说明（DATA）

> **本仓库不含任何数据。** 数据单独发送。
> 本文说明数据是什么、需要哪几个用户、怎么放、怎么验证、怎么从原始 xlsx 重新提取。

---

## 1. 为什么数据不在仓库里

| 原因 | 说明 |
| --- | --- |
| 原始 xlsx 太大 | 4 个文件共 **718.7 MB**，单个 `定位全部数据20261014.xlsx` 就有 **446.4 MB**，超过 GitHub 100 MB 单文件上限，且 xlsx 已是压缩格式、gzip 后仅再省 3% |
| 含敏感信息 | 对话是**真实儿童对话原文**；照片表含儿童照片的 CDN URL（URL 本身即可定位真人照片） |
| 已单独发送 | 按交接约定，数据不走 git |

`.gitignore` 里已写 `data/`，**默认排除**，不需要额外操作。

---

## 2. 数据是什么

来自 **2026-10-14 / 10-15 导出的一批原始 xlsx**，经筛取后转成 memory_agent 可直接读取的 `jsonl.gz`。

| 项 | 值 |
| --- | --- |
| 批次 | `1017` |
| 目标用户 | **13 个**（5 core + 8 seed） |
| 有效对话轮 | 22,755 |
| 对话 Session | 1,623 |
| 有效图片 | 1,348 |
| 有效定位 / 通话 | 24,969 / 268 |
| 提取后体积 | **7.65 MB**（原始 718.7 MB，约 1%） |

### 原始来源文件

```
智能体对话全部数据20261014.xlsx    244.0 MB
定位全部数据20261014.xlsx          446.4 MB
拍照图片全部数据20261015.xlsx       19.1 MB
通话记录全部数据20261014.xlsx        9.1 MB
```

> 这批导出此前**未被任何脚本处理过** —— `prepare_user_batches.py` 默认读 `BASE/date`，
> `core_user_audit.py` 匹配 `J20*.xlsx`，两个目录结构都不匹配 `data/1017`。

---

## 3. 需要哪几个用户

| # | device_id | 角色 | 有效对话轮 | 对话Session | 有效图片 | 有效通话 | 有效定位 | 丢弃(no_user_text) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `12354838B66792A5` | CORE | 8,018 | 604 | 270 | 134 | 3,496 | 739 |
| 2 | `12354838B667A7F0` | CORE | 6,096 | 237 | 66 | 6 | 2,405 | 333 |
| 3 | `12354838B667A861` | CORE | 3,491 | 296 | 134 | 6 | 3,049 | 312 |
| 4 | `12354838B667950A` | CORE | 1,993 | 161 | 81 | 35 | 2,445 | 235 |
| 5 | `12354838B6679432` | CORE | 1,971 | 70 | 46 | 44 | 1,955 | 98 |
| 6 | `12354838B6678EAF` | SEED | 382 | 39 | 134 | 35 | 1,903 | 107 |
| 7 | `12354838B6678E41` | SEED | 263 | 90 | 246 | 44 | 1,592 | 145 |
| 8 | `12354838B6678E87` | SEED | 194 | 45 | 89 | 5 | 2,890 | 102 |
| 9 | `12354838B6678EAE` | SEED | 179 | 40 | 60 | 1 | 1,388 | 79 |
| 10 | `12354838B6678EB7` | SEED | 159 | 35 | 24 | 13 | 694 | 57 |
| 11 | `12354838B6678EB9` | SEED | 9 | 6 | 88 | 4 | 857 | 44 |
| 12 | `12354838B6678EBC` | SEED | **0** | **0** | 75 | 0 | 2,213 | 7 |
| 13 | `12354838B6678EC0` | SEED | **0** | **0** | 18 | 4 | 82 | 2 |
| | **合计** | | **22,755** | **1,623** | **1,348** | **268** | **24,969** | |

### ⚠️ 三个稀疏用户，用之前先知道

| 用户 | 问题 |
| --- | --- |
| `...8EBC` | **对话 0 有效轮**。只有 75 张图片 → 只会产生图片 Session |
| `...8EC0` | **对话 0 有效轮**。只有 18 张图片 → 只会产生图片 Session |
| `...8EB9` | 只有 **9 有效轮 / 6 Session**，图片 88 张。样本量过小，统计结论不可用 |

> 这 3 人在本批数据里的对话**全部**是 `no_user_text`（助手发言、无用户输入），
> 所以 `eligible_turns=0`。"有效轮"指用户真实发言，**不是丢数据**。

**只想看效果，建议先用 `...92A5`（最大，8,018 轮）或 `...9432`（较小且干净，1,971 轮 / 70 Session）。**

---

## 4. 放到哪里

数据放在仓库的 `data/1017/` 下，结构必须是：

```
data/1017/
├─ users.csv                  13 个用户名单（memory_agent 的 data_root 需要它）
├─ manifest.json              清单元数据（用户、各来源记录数、源文件）
├─ extraction_summary.json    提取口径与统计（batch、max_date、disposition 分布）
└─ candidate_data/
   └─ <device_id>/
      ├─ dialogue.jsonl.gz    ← 运行时必需
      ├─ photo.jsonl.gz       ← photos.enabled=true 时必需
      ├─ call.jsonl.gz        ← memory_agent 目前不读，保留备用
      └─ location.jsonl.gz    ← memory_agent 目前不读，保留备用
```

本机已备份一份在**包外**：`D:\project\pycharm\LLM_Memory\data\extracted_1017\`（13 用户 / 55 文件 / 7.66 MB）。
复制到 `LLMMemory\data\1017\` 即可运行。

---

## 5. 记录格式

每行一个 JSON，与 `core_users/**/candidate_data` 完全同构：

```json
{"batch":"1017","source_kind":"dialogue","source_file":"智能体对话全部数据20261014.xlsx",
 "source_relative_path":"1017/智能体对话全部数据20261014.xlsx","sheet":"Sheet1","row":123,
 "disposition":"eligible","flags":[],"occurred_at":"2026-08-24T17:24:43.000001+08:00",
 "device_ids":["12354838B66792A5"],"data":{...原始字段原样保留...},
 "record_key":"id:2345525","content_fingerprint":"...","identity_method":"primary_key"}
```

要点：

- **`data` 是原始 xlsx 字段的原样值**（含 Excel 序列号时间），未做改写
- `disposition` 由 `core_user_audit.classify` 判定：**只有 `eligible` 参与提取**
- `occurred_at` 已转 UTC+8，带时区
- 按 `(occurred_at, record_key)` 升序（运行时还会再排序）
- `row` 是 xlsx 原始行号，可回溯

### 各来源 disposition 分布

| 来源 | 总命中 | eligible | 其他 |
| --- | ---: | ---: | --- |
| dialogue | 25,015 | **22,755** | `no_user_text` 2,260 |
| photo | 1,348 | **1,348** | — |
| location | 24,969 | **24,969** | — |
| call | 385 | **268** | `not_connected` 117 |

---

## 6. 怎么跑

用随包配置 `python_app/config/memory_agent_batch13_1017.json`（已指向 `data_root: data/1017`、13 个用户）：

```powershell
# 先空跑验证（不调用模型、零费用）
& 'D:\envrionment\Python312\python.exe' -X utf8 `
  'python_app\src\memory_agent.py' `
  --config 'python_app\config\memory_agent_batch13_1017.json' --mode prepare

# 确认无误后再正式跑（真实计费）
& 'D:\envrionment\Python312\python.exe' -X utf8 `
  'python_app\src\memory_agent.py' `
  --config 'python_app\config\memory_agent_batch13_1017.json' --mode live
```

**已在数据齐备时实测通过**：13 个用户全部 `exit_code=0`，共 2,890 个 Session，106 维规则加载正常。
（`...8EBC` 的 75 个 Session 与 `...8EC0` 的 18 个全部来自图片。）

> 想只跑一个用户：把配置里 `user_selection.device_ids` 改成单个 ID，或另存一份配置。

---

## 7. 怎么自己从原始 xlsx 重新提取

提取脚本随包：`scripts/extract_1017_users.py`。

```powershell
# --data 指向放着 4 个原始 xlsx 的目录
& 'D:\envrionment\Python312\python.exe' -X utf8 scripts\extract_1017_users.py `
  --data   'D:\project\pycharm\LLM_Memory\data\1017' `
  --output 'data\1017' `
  --kinds dialogue photo call location

# 生成 users.csv 与 manifest.json
& 'D:\envrionment\Python312\python.exe' -X utf8 scripts\build_1017_roster.py --root data\1017
```

脚本行为：

- 复用仓库自身的 `core_user_audit.StreamBook`（**流式解析 XML，不整表载入内存**）、
  `classify`、`devices_for` 与 `prepare_user_batches.identity`，
  保证 `disposition` / `record_key` / `content_fingerprint` 口径与 `core_users` 既有数据**完全一致**
- 每个来源单独扫描并缓存到 `--scratch`，重复运行可秒级跳过（`--force` 强制重扫）
- **不修改原始 xlsx**，不调用任何模型
- 全量扫描实测耗时：对话 211s + 图片 25s + 通话 10s + 定位 372s ≈ 10 分钟

### 改目标用户

改 `scripts/extract_1017_users.py` 顶部的 `CORE` 与 `SEED` 两个列表。

### `--max-date` 注意

默认 `2026-10-15`（本批导出日期）。
**不要沿用 `python_app/config/core_user_audit.json` 里的 `2026-09-07`** ——
那会把 9 月之后的记录全判为 `invalid_time`。脚本已单独处理，不读那个文件。

### 源文件名必须一致

在 `scripts/extract_1017_users.py` 顶部的 `FILES` 字典里：

```
智能体对话全部数据20261014.xlsx
拍照图片全部数据20261015.xlsx
通话记录全部数据20261014.xlsx
定位全部数据20261014.xlsx
```

---

## 8. 合规提醒

- `dialogue.jsonl.gz` 含**真实儿童对话原文**
- `photo.jsonl.gz` 含儿童照片的 CDN URL（**URL 本身即可定位真人照片**）
- 分享数据时确认走内部合规通道，**不要推到公开仓库或公开网盘**
