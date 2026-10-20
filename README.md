# LLMMemory —— 儿童记忆提取 Agent 交接包

本仓库是 **ToyMemory 记忆提取 Agent** 的完整交接包：可运行代码、运行参数、提示词、原始产品方案，以及数据提取脚本。

> **本仓库不含数据**（含儿童对话，体积也太大），数据单独发送。
> 数据规格、用户名单、放置位置、重提方法见 **`DATA.md`**。

> 目标：接手人读完 `docs/01_操作手册.md` 即可跑通；读完 `docs/02_设计说明.md` 即可理解设计与已知问题。

---

## 目录结构

```
LLMMemory/
├─ README.md                    ← 本文件
├─ DATA.md                      ★ 数据说明（本仓库不含数据）
├─ config.example.yaml          凭证模板（真实 key 不入库）
├─ .gitattributes               文本/二进制属性（*.gz 按 binary，防换行损坏）
├─ .gitignore                   含 data/，默认排除数据
├─ J20核心用户记忆维度与提取存储规划.md   ★ 106 维规则来源（运行时必需，BASE 根级）
├─ python_app/
│  ├─ src/                      生产代码（53 个模块，仅标准库）
│  ├─ config/                   运行参数（31 个 JSON，含 batch13_1017）
│  ├─ prompts/                  提示词 TXT
│  ├─ assets/                   HTML 模板与前端脚本
│  ├─ tests/                    52 个测试 + 10 个离线校验脚本
│  └─ requirements*.txt         依赖清单
├─ scripts/                     ★ 数据提取脚本（随包，数据自备时用它重建）
│  ├─ extract_1017_users.py     从原始 xlsx 筛出目标用户 → jsonl.gz
│  └─ build_1017_roster.py      生成 users.csv 与 manifest.json
├─ docs/
│  ├─ 01_操作手册.md            ★ 环境、运行、配置、排障、产物说明
│  ├─ 02_设计说明.md            ★ 架构、不变量、实测数据、已知问题
│  ├─ 项目架构.md               原仓库文档
│  ├─ 技术文档.md               原仓库文档（最详细，1515 行）
│  ├─ 更新日志.md               版本历史
│  └─ J20核心用户记忆维度与提取存储规划.md   同上（留档副本，代码实际读根级那份）
├─ 产品文件/
│  ├─ 记忆维度设计-v1.0.xlsx    ★ 106 维的原始设计出处
│  └─ ...（产品方案 V3.2 / V0.2 / 推荐方案 V1.0）
└─ core_users/                  用户名单 CSV（不含对话与照片原始数据）
```

> **注意**：`rules_file` 在配置里写作 `J20核心用户记忆维度与提取存储规划.md`，
> 而代码用 `BASE = parents[2]` 解析，所以**必须放在仓库根级**（已放好）。
> `docs/` 下那份只是留档副本，改动它不会生效。

---

## 快速开始

### 1. 确认环境

```powershell
& 'D:\envrionment\Python312\python.exe' -X utf8 -c "import yaml,sys;print(sys.version)"
```

主链路**仅依赖标准库** + `PyYAML`（读凭证）。`json_repair` 可选，缺失时优雅降级。详见 `docs/01_操作手册.md` §1。

### 2. 配置凭证

```powershell
copy config.example.yaml config.yaml
# 编辑 config.yaml，填入真实 api_key 与推理接入点 ID (ep-xxxx)
```

### 3. 先跑 prepare（**不调用模型、零费用**）

**先把数据放到 `data/1017/`**（数据单独发送，放置方式见 `DATA.md` §4），然后：

```powershell
& 'D:\envrionment\Python312\python.exe' -X utf8 `
  'python_app\src\memory_agent.py' `
  --config 'python_app\config\memory_agent_batch13_1017.json' --mode prepare
```

预期输出：13 个用户各一行 `{"status": "prepared_not_extracted", ..., "sessions": N, "dimensions": 106}`，退出码 0。
（数据齐备时实测共 2,890 个 Session。有 3 个用户样本很稀疏，见 `DATA.md` §3 的提醒。）

> 数据还没到手时，可以先用 `python_app/config/memory_agent_generic_flow.json`
> 或任意 `core_users/` 下的名单验证环境是否装好，但那同样需要对应的数据目录。

### 4. 再跑 live（真实计费）

```powershell
& 'D:\envrionment\Python312\python.exe' -X utf8 `
  'python_app\src\memory_agent.py' `
  --config 'python_app\config\memory_agent_batch13_1017.json' --mode live
```

> ⚠️ `live` 会产生真实 API 费用。该配置 **未设 `max_calls`（无上限）**。
> 建议首次只留一个 `device_ids` 并把 `max_sessions` 改成 `3` 验证。

### 5. 离线回归测试（不调用模型）

```powershell
& 'D:\envrionment\Python312\python.exe' -X utf8 -m unittest discover -s 'python_app\tests' -q
```

---

## 数据

> ⚠️ **本仓库不含数据。** 数据单独发送，完整规格见 **`DATA.md`**。

数据来自 2026-10-14 / 10-15 导出的一批**原始 xlsx**（对话 244MB / 定位 446MB / 拍照 19MB / 通话 9MB，共 718.7 MB），
筛出 **13 个用户**（5 core + 8 seed），转成 `memory_agent` 可直接读取的 `jsonl.gz`，压缩到 **7.65 MB（约 1%）**。

| 项 | 值 |
| --- | --- |
| 有效对话轮 | 22,755 |
| 对话 Session | 1,623 |
| 有效图片 | 1,348 |
| 有效定位 / 通话 | 24,969 / 268 |

**不进仓库的原因**：含真实儿童对话与照片地址；且 446MB 单文件超 GitHub 100MB 上限，
xlsx 已是压缩格式、再压仅省 3%，git-lfs 也避不开体积。

数据放到 `data/1017/` 后即可运行；`.gitignore` 已排除 `data/`，不会误提交。
用 `scripts/extract_1017_users.py` 可从原始 xlsx 重新提取 —— 用户名单、口径、耗时见 `DATA.md` §7。

---

## 阅读顺序建议

| 顺序 | 文档 | 作用 |
| --- | --- | --- |
| 1 | `docs/01_操作手册.md` §0–§2 | 知道包里有什么、怎么跑起来 |
| 2 | `DATA.md` | **数据单独发送**：哪 13 个用户、放哪里、怎么跑、怎么重提 |
| 3 | `docs/02_设计说明.md` §4 | 理解关键不变量（改代码前必读） |
| 4 | `docs/02_设计说明.md` §7–§9 | **已知问题与改进方向** |
| 5 | `docs/技术文档.md` | 需要字段级细节时查阅 |
| 6 | `产品文件/记忆维度设计-v1.0.xlsx` | 讨论维度体系时对照产品口径 |
| 7 | `docs/项目架构.md` | 了解历史演进与版本边界 |

---

## 需要知道的四件事

### ① 生产只有一个入口

`python_app/src/memory_agent.py`。其余如 `main.py`、`behavior_service.py`、`recommendation_service.py` 属于**早期推荐规则原型，不在本链路**；`tests/legacy_*.py` 是**冻结的历史实现夹具**，仅供对照，不要导入或修改。

### ② 有两套流程版本，不能混用

`generic-flow-v2` 与 `generic-flow-v3-relations`（含本地 BGE 召回）。**跨版本禁止续跑**，换版本必须用新输出目录。当前配置用的是 v3-relations。

### ③ 已知问题尚未修复

`docs/02_设计说明.md` §8 列出了实测确认的问题，其中三条最影响产出：

| 级别 | 问题 | 症状 |
| --- | --- | --- |
| **P0-1** | 维度体系与控制逻辑耦合 | `UNCLASSIFIED` 占 65~84%，106 维实际只用到 72 个 |
| **P0-2** | 价值判断依赖 `scope`（四跳耦合） | 「单次会话说出的喜好」拿不到长期身份，用户间长期层产出率相差 26 倍 |
| **P0-3** | `pending` 是不可回捞的黑洞 | 候选被静默丢弃，无复核路径 |

**接手后如果要改进，建议先做 §9.6 的评估体系（零模型成本），再动机制。**

### ④ 技术失败与语义待确认必须分开看

`partial` / `awaiting_retry` / `failed` 是**技术状态**，可以 `--resume` 重跑；
候选级的 `pending` 是**语义待确认**，不是 bug。详见 `docs/01_操作手册.md` §5。

---

## 合规与提交注意

### ⚠️ 敏感数据

| 内容 | 状态 |
| --- | --- |
| `config.yaml`（真实 API key） | **不在本包内**，从 `config.example.yaml` 生成 |
| **数据（对话 / 图片 / 定位 / 通话）** | **本仓库完全不含**，单独发送。其中对话是真实儿童原文、照片表含可定位真人照片的 CDN URL |
| `core_users/**/*.csv` | 仅名单，不含对话。**在仓库内** |
| `photo_assets/`（儿童照片本体） | 本包未包含，**任何情况下不要入库** |
| `memory.sqlite3` / `model_calls.jsonl` | 本包未包含，含原始对话与模型输出 |
| `output/`、`bge_runtime/` | 本包未包含（运行产物、含本地服务 token） |

> 数据按约定单独发送。**不要把数据加到 git** —— `.gitignore` 已排除 `data/`，
> 若确实需要入库请先确认合规，并注意 446MB 单文件超 GitHub 上限。

### 提交前自查

```powershell
# 确认没有凭证
Get-ChildItem -Recurse -File | Select-String -Pattern 'api_key|sk-[a-z0-9]{20,}' | Select-Object Path,LineNumber

# 确认没有环境与缓存残留
Get-ChildItem -Recurse -Directory -Include __pycache__,.venv,.pytest_cache
```

### 本包已排除（不会进 git）

`.venv/`、`__pycache__/`、`*.pyc`、`data/`（原始 xlsx 与提取数据，**数据单独发送**）、
`core_users/**/candidate_data/`（既有批次的对话数据包）、`output/`、`bge_runtime/`、`config.yaml`。

---

## 一句话总结现状

工程侧（证据凭证链、事务重放、缓存幂等、Session 隔离）**扎实且可审计**；
主要问题在**产品侧设计**：把「分类标签」当成了「控制闸门」，导致长期记忆产出率仅 **8.17%**，且 65% 的事实没有维度。
详见 `docs/02_设计说明.md`。
