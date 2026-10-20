# ToyMemory 记忆-内容-推荐规则核心

## 当前记忆 Agent：单链路 generic-flow-v2

当前唯一入口：按Skill提取 → Session汇总 → 来源审核与一次修订 → 同源一致性 → 独立分类 → 历史维护 → 原子提交 → 双HTML。技术失败与语义待确认分别显示，已完成阶段支持缓存恢复。

唯一当前配置为 `python_app/config/memory_agent_generic_flow.json`：8EAF用户、0907全部对话和图片、豆包2.1 Lite、并行1人，新目录为 `output/memory_agent_single_flow_0907`。旧执行配置停止接受，旧结果仍可查看导出；不自动迁移。真实运行由用户启动。

```powershell
& 'D:\envrionment\Python312\python.exe' -X utf8 'D:\project\pycharm\LLM_Memory\python_app\src\memory_agent.py' --mode live
```

续跑在上述命令末尾加 `--resume`，不能用于旧版本目录。详见 [项目架构](docs/项目架构.md) 和 [技术文档](docs/技术文档.md) 当前章节。双HTML为批次根目录的 `index.html` 和 `index_report.html`，数据内嵌，可离线查看。唯一重试参数 `stage_max_attempts` 为1～3，默认3；网络与协议修正共享预算，不再嵌套重试。

以下内容保留早期推荐规则原型说明；其中推荐、家长交互和计分不属于当前记忆提取链路。

这是根据《ToyMemory记忆-内容-推荐整体方案V0.1》实现的可运行规则核心。当前版本聚焦可审计闭环：行为采集、记忆回流、家长审核、内容召回、硬过滤、排序和推荐留痕。

## 已实现

- 四类记忆的数据结构、五个需求状态，以及停止使用状态；
- 记忆证据、来源、置信度和调用记录；
- 完整收听、播放、跳过、接受推荐、拒绝、不再推荐六类事件；
- 主动点播、接受推荐后播放、系统闲置推送三类来源权重；
- 3分生成候选、初始置信度65、待确认封顶84和保底40；
- 事件幂等、对话/行为证据去重、每批与每周候选频控；
- 同义归一、上下位实体召回；
- 黑名单、家长禁用主题、年龄、内容状态和敏感记忆硬过滤；
- 确定性排序、推荐依据和推荐运行日志；
- SQLite 持久化和命令行现场演示。

## 环境

代码仅使用 Python 标准库，已在 Python 3.12 下验证，不需要安装第三方包。仓库中的 `.venv/` 被忽略，不属于交付资产。

快速脚本按以下顺序选择环境：

1. `-PythonExe` 显式传入的解释器；
2. 项目本地 `.venv`；
3. `CONDA_ENV_NAME` 指定的 Conda 环境；
4. Windows `py -3.12` 启动器。

## 快速运行

```powershell
& .\scripts\quick_start.ps1
```

显式指定团队环境中的解释器：

```powershell
& .\scripts\quick_start.ps1 -PythonExe "C:\path\to\python.exe"
```

演示会依次完成：创建孩子和内容快照、写入三次海豚完整收听、回流候选记忆、模拟家长确认、生成带依据的推荐。运行数据库保存在 `output/toy_memory.sqlite3`，不会进入版本库。

## 测试

```powershell
& .\scripts\test_python.ps1
```

## CLI

```powershell
& .\.venv\Scripts\python.exe .\python_app\src\main.py --config .\python_app\config\default.json init-db
& .\.venv\Scripts\python.exe .\python_app\src\main.py --config .\python_app\config\default.json demo
& .\.venv\Scripts\python.exe .\python_app\src\main.py --config .\python_app\config\default.json process
& .\.venv\Scripts\python.exe .\python_app\src\main.py --config .\python_app\config\default.json list-memories --child-id demo_child
& .\.venv\Scripts\python.exe .\python_app\src\main.py --config .\python_app\config\default.json recommend --child-id demo_child --entities 海豚 --limit 2
```

家长审核需要先从 `list-memories` 结果取得 `memory_id`：

```powershell
& .\.venv\Scripts\python.exe .\python_app\src\main.py --config .\python_app\config\default.json review --memory-id <memory_id> --action confirm
```

## 目录

```text
python_app/config/default.json  规则参数
python_app/src/                 领域模型、存储与业务服务
python_app/tests/               自动化测试
scripts/                        快速运行与测试入口
docs/                           架构、技术说明和更新日志
output/                         本地数据库和运行产物（不入库）
```

## 当前边界

- 内容混合标签可通过 `ContentCatalogService.upsert_from_mixed_tags` 自动拆分并写入本地快照；正式内容平台检索接口尚未提供，因此没有伪造网络客户端。
- 大模型话术尚未接入；系统输出 `speech_mode`、内容和推荐依据，供智能体生成话术。
- 原推荐核心仍通过 `MemoryService.create_memory` 接收外部抽取结果；新增的中心Agent Demo使用独立数据库，尚未自动接入推荐核心。
- 冷启动和兴趣衰减尚无需求中的确定参数，当前未擅自实现。

更完整的数据表、算法和限制说明见 [项目架构](docs/项目架构.md) 与 [技术文档](docs/技术文档.md)。

## 中心记忆提取Agent Demo

入口为`python_app/src/memory_agent.py`，与旧推荐CLI独立。基于已筛选用户`12354838B6679292`最早4个连续会话：中心LLM动态选择8类业务Skill，加载原方案的106项维度规则，按需读取先前会话记忆，独立复核后写入SQLite、106维度书页和关键词索引。向量仅登记disabled状态，没有伪造向量。

环境复用`D:\envrionment\Python312\python.exe`（Python 3.12.3），新增入口使用现有PyYAML 6.0.2读取本地凭证，可选复用已安装json_repair处理JSON语法，缺少该库也能启动且仍拒绝非法输出；未安装新包。原规则核心仍只依赖标准库。模型配置读取根目录`config.yaml`，只允许Doubao/Deepseek服务及豆包、DeepSeek模型别名；不读取其中其他模型运行。当前运行参数在`python_app/config/memory_agent.json`，默认豆包lite、关闭深度思考。凭证配置、用户数据和output均已忽略，不提交凭证；不含密钥的运行参数文件可版本管理。

```powershell
# 仅检查数据和106项映射，不调用模型；输出目录须为空或不存在
& 'D:\envrionment\Python312\python.exe' -X utf8 'D:\project\pycharm\LLM_Memory\python_app\src\memory_agent.py' --mode prepare --output 'D:\project\pycharm\LLM_Memory\output\agent_prepare_check'

# 真实调用：重跑请换新的输出目录，避免覆盖已有结果
& 'D:\envrionment\Python312\python.exe' -X utf8 'D:\project\pycharm\LLM_Memory\python_app\src\memory_agent.py' --mode live --output 'D:\project\pycharm\LLM_Memory\output\agent_live_repeat'

# 全部离线测试，不调用模型
& 'D:\envrionment\Python312\python.exe' -X utf8 -m unittest discover -s 'D:\project\pycharm\LLM_Memory\python_app\tests' -v

# 独立、只读检查默认实跑产物，不调用模型
& 'D:\envrionment\Python312\python.exe' -X utf8 'D:\project\pycharm\LLM_Memory\python_app\src\validate_agent_run.py' 'D:\project\pycharm\LLM_Memory\output\memory_agent_demo_v1'

# 若因网络或预算在会话提交前中断，可校验原输入后续跑，已完成会话不再调用模型
& 'D:\envrionment\Python312\python.exe' -X utf8 'D:\project\pycharm\LLM_Memory\python_app\src\memory_agent.py' --mode live --resume
```

输出目录内优先看`运行报告.md`与`记忆书页.md`；`manifest.json`保存样本边界与维度映射，`memory.sqlite3`是权威记忆，`report.json`、`trace.jsonl`、`model_calls.jsonl`供问题复核。原始模型响应含用户内容，只保留本地，不应上传公共仓库。HTTP/预算/校验失败不降级成假结果，退出码2代表部分完成或失败。

此Demo只验证对话链路，不代表全量用户、图片/定位/通话或106维度的准确率已经验收。共享设备不等同唯一儿童。详细字段、限制和参数见技术文档第10节。

### 选择一个、多个或全部种子用户

编辑python_app/config/memory_agent.json中的user_selection（不要同时保留旧device_id字段）：

| 用途 | user_selection配置 |
|---|---|
| 一个用户 | `{"mode":"selected","device_ids":["12354838B6679292"]}` |
| 多个用户 | `{"mode":"selected","device_ids":["12354838B6679292","12354838B6679160"]}` |
| 全部用户 | `{"mode":"all","device_ids":[]}` |

mode选择指定列表或全部；device_ids填写设备ID，不是linked_user_ids。全部指data_root下core_50_provisional.csv中的全部核心初选设备，不是原始日志全部用户。指定列表不能为空、不能重复，也不能填写名单外ID。运行前检查选中用户的对话包存在。

max_sessions为正整数时表示每用户最早几个会话，为"all"时表示全部有效会话。max_calls和max_total_input_chars是每用户、每次执行的累计预算，null表示用户已授权不设累计上限。parallel_users当前为3（支持1至3）：最多3个用户独立进程并行，同用户会话仍按时间串行。单用户失败记入batch_report.json并继续其他用户，不混用历史、数据库和模型计数。

当前配置由用户选择两位设备12354838B6679432、12354838B667A7F0，max_sessions="all"处理各自全部有效会话；人数及路径以配置文件为准。2026-09-29起使用有界上下文v2：原文完整分块、规则与候选分页、历史按需限量、Planner不带完整执行记录；逐条Review核对全部session窗口后，才通过原存储门禁提交。旧90000字符配置已废弃，不能理解为允许无限请求。当前保守输入预算49440，包含提示、数据及schema；超预算拆分任务而非截断原文。模型调用累计预算仍为null，单次输出12000 token、60秒超时和Agent步数限制保留。此改造未调用真实模型、未修改已有结果；新版请使用独立目录，例如：

```text
output/memory_agent_rank2_context_v2/
  index.html                唯一汇总HTML，页面内切换全部用户
  batch_manifest.json       本次用户范围、配置指纹和版本
  batch_report.json         每个用户执行状态、耗时与结果路径
  12354838B6679432/
    memory.sqlite3          此用户独立记忆库
    运行报告.md
    记忆书页.md
    ...                     原有审核、证据与模型日志
```

--output指定批次根目录。先用--mode prepare检查选择和会话，不调用模型；prepare目录不能直接用于live，请另指定新目录。新版REVIEW_VERSION为item-review-v2-bounded，不支持旧v1目录直接--resume。v2同目录可续跑，复用完成的窗口，重试未提交窗口；只允许单独调整parallel_users，其他配置、输入、用户集合及规则版本必须一致。切换用户或处理模式请换输出目录。旧device_id配置仍支持原单用户目录布局，但同样校验新版指纹。

若仍有旧进程运行，先在原窗口按一次Ctrl+C，等待保存报告并返回PowerShell提示符；修改源码不会替换已加载的旧代码。不要强制关窗口或同时启动新旧两份程序。下方--resume命令仅用于已经创建的v2批次，不能用于旧v1升级；v2首次运行见后面的不带--resume命令。批次使用.batch.lock互斥，文件保留但进程退出后系统锁释放。并行时Ctrl+C停止派发新用户，正在进行的请求返回或超时后在下一调用/会话边界退出并保存。

```powershell
& 'D:\envrionment\Python312\python.exe' -X utf8 'D:\project\pycharm\LLM_Memory\python_app\src\memory_agent.py' --mode live --output 'D:\project\pycharm\LLM_Memory\output\memory_agent_rank2_context_v2' --resume
```

旧版本已提交的partial不自动覆盖或迁移。v2的未完成窗口仅写context_work暂存，正式sessions只接收全窗口提取完成且经过逐条Review的批次；同配置可以复用暂存重试。defer仍表示待定，不自动唤醒。日志含[user=设备ID]；batch_report的parallel_users表示并行上限、worker_pid标明进程。主进程统一写报告与HTML。单次请求受控不代表总内存/费用固定；原始用户数据、候选及报告仍占本地内存，逐条跨窗口审核可能增加调用成本。

export_html为布尔开关：true时，每位用户live运行后更新批次根目录唯一的index.html；false时不导出。该文件内嵌全部用户数据，顶部下拉框切换用户，无需其他HTML或服务器。数据库和日志仍按用户隔离，不在子目录额外生成HTML。页面只展示当前版本Review通过的正式记忆，审核事件单列；待运行、失败或零记忆结果明确标记。源报告缺失时显示对应用户状态；导出失败记入batch_report.json。导出不调用模型、不修改数据库，汇总文件含所有用户原文，请勿公开分享。每次更新会读取已有用户结果并重写此派生文件，内存和文件大小随已处理用户总量增长。

已有批次可单独重新导出，不重新调用模型：

```powershell
& 'D:\envrionment\Python312\python.exe' -X utf8 'D:\project\pycharm\LLM_Memory\python_app\src\export_memory_html.py' --input 'D:\project\pycharm\LLM_Memory\output\memory_agent_rank2_context_v2'
```

首次运行当前配置的两用户全会话（用户已授权取消累计预算上限，使用新目录，不加--resume）：

```powershell
& 'D:\envrionment\Python312\python.exe' -X utf8 'D:\project\pycharm\LLM_Memory\python_app\src\memory_agent.py' --mode live --output 'D:\project\pycharm\LLM_Memory\output\memory_agent_rank2_context_v2'
```

### 强制逐条Review（说明）

新运行使用“中心提取 → 会话Auditor补提检查 → 每条独立Review → 存储凭证校验 → 正式库”。Review只允许approve进入正式库；revise修订一次后重新审核，defer/reject仅留审核记录。正式历史只召回当前版本有通过凭证的reviewed记忆，不再召回旧candidate/hypothesis。

完整流程图见[项目架构第7节](docs/项目架构.md)，字段与边界见[技术文档10.5节](docs/技术文档.md)。新增模块为python_app/src/agent_review.py，新增门禁测试为python_app/tests/test_agent_review.py。

原memory_agent_demo_v1目录和用户HTML保留原样，不自动把34条旧候选升级。新版实跑请使用新输出目录，旧版本不支持直接--resume。机器审核通过不等于人工确认或客观事实核实；待定记录没有自动重审任务，遗忘控制仍需治理。审核增加模型调用，沿用原预算而不自动放宽。

历史v1实跑产物见`output/memory_agent_demo_v1/运行报告.md`和`记忆书页.md`：4个会话、86条有效话轮、34条候选；抽查发现的主体和维度问题仍是历史结果的质量限制，不能直接作为确认画像。这不是新版真实模型验收结果。

### 有界上下文配置与验证

配置文件新增context_window_tokens=65536、context_safety_tokens=4096、session_chunk_tokens=8192、history_context_tokens=4096、rule_page_tokens=8192、context_split_retries=6。数值采用保守UTF-8字节估值，非供应商精确token计数；含schema的总输入再预留输出与安全空间后检查，服务端仍可能拒绝并触发有界拆分。原文不静默截断，超长单话轮也按区间完整覆盖。

完整字段、调用内容、恢复边界及成本见[技术文档10.6节](docs/技术文档.md)。新增agent_context.py、agent_execution.py、test_agent_context.py。本次没有新增依赖，没有重跑真实模型，也没有实现每日增量调度。每日调度与历史全量处理是上层运行策略，不应通过扩大单次上下文解决。

离线回归命令（包含真实数据分块核验及模拟模型边界测试，不发模型请求）：

```powershell
& 'D:\envrionment\Python312\python.exe' -X utf8 -m unittest discover -s 'D:\project\pycharm\LLM_Memory\python_app\tests' -q
```
