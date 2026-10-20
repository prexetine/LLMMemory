"""读取阶段一产物，验证分包并生成中文交付报告，不进行记忆提取。"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def make_report(root: Path) -> None:
    started = time.perf_counter()
    summary = json.loads((root / 'run_summary.json').read_text(encoding='utf-8'))
    candidates = read_csv(root / 'candidates_100.csv')
    manifest = read_csv(root / 'export_manifest.csv')
    validation = {'source_hashes_unchanged': {}, 'exports': [], 'errors': []}
    for kind, source in summary['files'].items():
        digest = hashlib.sha256()
        with Path(source['path']).open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        valid = digest.hexdigest() == source['sha256']
        validation['source_hashes_unchanged'][kind] = valid
        if not valid:
            validation['errors'].append('源文件变化: ' + kind)
    counts = Counter()
    temporal_review = {}
    for item in manifest:
        device = item['device_id']
        for kind in ('dialogue', 'photo', 'call', 'location'):
            path = root / 'candidate_data' / device / (kind + '.jsonl.gz')
            actual = 0
            seconds = Counter()
            examples = []
            with gzip.open(path, 'rt', encoding='utf-8') as handle:
                for line in handle:
                    record = json.loads(line)
                    actual += 1
                    if device not in record['device_ids'] or record['source_kind'] != kind:
                        validation['errors'].append(f'归属错误:{device}/{kind}:{actual}')
                    if not isinstance(record.get('row'), int) or record['row'] < 2:
                        validation['errors'].append(f'源行错误:{device}/{kind}:{actual}')
                    if kind == 'dialogue' and record['disposition'] == 'eligible':
                        seconds[record['occurred_at']] += 1
                        if len(examples) < 5:
                            examples.append({'sheet': record['sheet'], 'row': record['row'], 'time': record['occurred_at']})
            expected = int(item[kind + '_exported_rows'])
            if actual != expected:
                validation['errors'].append(f'条数不符:{device}/{kind}:{actual}!={expected}')
            validation['exports'].append({'device_id': device, 'kind': kind, 'rows': actual, 'expected': expected})
            counts[kind] += actual
            if kind == 'dialogue':
                temporal_review[device] = {'max_same_timestamp': max(seconds.values(), default=0),
                                           'timestamps_ge10': sum(n >= 10 for n in seconds.values()),
                                           'example_refs': examples}
        print(f'校验候选 {item["rank"]}/{len(manifest)}', flush=True)
    db = sqlite3.connect(root / 'audit.sqlite3')
    review_rows = []
    for candidate in candidates:
        counted = db.execute('SELECT sum(n) FROM dates WHERE device=?', (candidate['device_id'],)).fetchone()[0]
        if counted != int(candidate['dialogue_turns']):
            validation['errors'].append('对话统计不一致: ' + candidate['device_id'])
        unique = db.execute('SELECT count(*) FROM utterances WHERE device=?', (candidate['device_id'],)).fetchone()[0]
        t = temporal_review[candidate['device_id']]
        batch_suspect = int(candidate['active_days']) == 1 and t['timestamps_ge10'] >= 3 and unique / counted <= 0.2
        review_rows.append({'rank': candidate['rank'], 'device_id': candidate['device_id'],
                           'selection': candidate['selection'], 'rule_flags': candidate['risk_reasons'],
                           'unique_user_texts': unique, 'max_same_timestamp': t['max_same_timestamp'],
                           'timestamps_ge10': t['timestamps_ge10'], 'batch_pattern_suspect': batch_suspect,
                           'decision': 'hold_reserve_for_business_review' if batch_suspect else 'retain_provisional',
                           'reason': '单日+多批次同时间大量请求+低文本多样性，疑似批量测试/数据回放，需业务确认' if batch_suspect else '没有足以确认测试身份的组合证据；不因高频或单句重复直接排除'})
    validation['sqlite_integrity'] = db.execute('PRAGMA quick_check').fetchone()[0]
    db.close()
    with (root / 'review_decisions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]) if review_rows else ['device_id', 'decision', 'reason'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(review_rows)
    (root / 'temporal_review_evidence.json').write_text(json.dumps(temporal_review, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    if validation['sqlite_integrity'] != 'ok':
        validation['errors'].append('SQLite完整性检查失败')
    for kind, audit in summary['audit'].items():
        if sum(audit['dispositions'].values()) != audit['rows']:
            validation['errors'].append('清洗计数不守恒: ' + kind)
    validation['seconds'] = time.perf_counter() - started
    validation['passed'] = not validation['errors']
    (root / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    sizes = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    flagged = [r for r in candidates if r['risk_reasons']]
    core = candidates[:summary['core_provisional']]
    lines = ['# J20 阶段一：数据审计与核心用户初筛报告', '',
             '## 1. 结果与状态', '',
             f'- 已扫描四份源工作簿，按设备识别 {summary["devices"]:,} 个非空主体标识，其中 {summary["rankable_devices"]:,} 个设备有符合口径的用户文本对话。',
             f'- 保存前 {len(candidates)} 位候选及其四类原始记录分包；前 {len(core)} 位为核心初选，尚未完成业务方测试设备身份确认。',
             f'- 前100候选中 {len(flagged)} 位有规则风险提示，核心初选中 {sum(bool(r["risk_reasons"]) for r in core)} 位有提示。风险不等于测试账号，未据此自动排除。',
             '- 未下载图片、未转写音频、未提取记忆、未创建向量或推荐能力。',
             f'- 流式扫描及分包耗时 {summary["elapsed_seconds"]:.1f} 秒；分包复核耗时 {validation["seconds"]:.1f} 秒；当前产物约 {sizes/1024**2:.1f} MiB（包括中间缓存）。',
             f'- 校验：{"通过" if validation["passed"] else "失败，请查看validation.json"}；源文件哈希、清洗分流、候选对话统计、分包条数与归属均已检查。', '',
             '## 2. 数据概况', '',
             '| 来源 | 全量数据行 | 工作表数 | 符合本来源口径 | 最早时间 | 最晚时间 | 候选分包行数 |',
             '|---|---:|---:|---:|---|---|---:|']
    labels = {'dialogue': '对话', 'photo': '拍照', 'call': '通话', 'location': '定位'}
    for kind, a in summary['audit'].items():
        lines.append(f'| {labels[kind]} | {a["rows"]:,} | {len(a["sheets"])} | {a["dispositions"].get("eligible",0):,} | {a["first"]} | {a["last"]} | {counts[kind]:,} |')
    lines.extend(['', '候选分包保留原始行及处理标签，包含不参与排名的记录；两设备通话可出现在双方包内，因此跨设备包不能直接求和当唯一通话数。', '',
                  '### 清洗分流', '', '| 来源 | 处理状态 | 行数 |', '|---|---|---:|'])
    for kind, a in summary['audit'].items():
        for status, count in a['dispositions'].items():
            lines.append(f'| {labels[kind]} | {status} | {count:,} |')
    lines.extend(['', '### 核心50位与100位候选的数据规模', '',
                  '| 范围 | 有效用户对话 | 原始对话行 | 拍照元数据行 | 通话行 | 定位行 |',
                  '|---|---:|---:|---:|---:|---:|'])
    for title, group in [('核心初选', core), ('全部候选', candidates)]:
        totals = [sum(int(r[k]) for r in group) for k in ('dialogue_turns','dialogue_raw','photo_raw','call_raw','location_raw')]
        lines.append('| ' + title + ' | ' + ' | '.join(f'{n:,}' for n in totals) + ' |')
    lines.extend(['', '分流原因互斥且按校验顺序归类，因此“缺少设备且已删除”的记录只计入missing_device；删除标志的原始分布详见run_summary.json的enums。',
                  'duplicate表示相同源主键且全行一致，不是按相似坐标或近邻时间合并；源记录保留在原工作簿，候选包内保留处理标签。',
                  'disposition中eligible为符合本来源计数条件；no_user_text为没有用户文本；missing_device为设备缺失；deleted_or_unknown_delete为删除或删除枚举未识别；not_connected为没有接通时间；invalid_device为设备标识不合格式。', ''])
    lines.extend(['', '## 3. 排名口径与限制', '',
                  '1. 用户已确认：text为用户输入、content为助手回复，status=0正常，del_flag=0未删除。',
                  '2. 主排名为去重后有效用户文本对话记录数，不按唯一chat_id计数；dialog_id按设备范围作为记录主键。audio-only不算文本轮数，另行保留。',
                  '3. 并列按对话活跃天数、有效拍照元数据数、已接通会话数、设备ID排序。定位不参与排名。',
                  '4. Unix秒/毫秒按UTC转换到UTC+8；Excel日期序列暂按UTC+8本地时间解释。混合时间格式已分别识别，业务时区仍需确认。',
                  '5. 图片仅检查未删除、设备、时间、URL语法和系统素材命名嫌疑；未访问URL，不保证可访问或为真实用户照片。validity_time含义未确认，不据其直接删除。',
                  '6. 接通计数要求connected_at存在、状态为ENDED/CONNECTED且时间顺序正常；completed但无接通时间不计入。通话时长字段不作为接通唯一依据。',
                  '7. 重复主键且全行一致记duplicate；同主键内容变化记key_conflict，保留首条暂计，所有版本保留。存在冲突时需业务复核后再做记忆提取。',
                  '8. 无设备、设备格式异常和异常时间不参与排名。相同文本发生于不同事件不自动删掉，避免误删真实重复需求。',
                  '9. 非空设备标识总数含异常标识；仅16位十六进制且有有效对话的设备进入排名。按设备分组不是对真实个人身份的认证。', '',
                  '## 4. 核心50位初选', '',
                  '| 排名 | 设备 | 有效对话 | 活跃天 | 有效拍照元数据 | 接通通话 | 风险提示 |',
                  '|---:|---|---:|---:|---:|---:|---|'])
    for r in core:
        lines.append(f'| {r["rank"]} | {r["device_id"]} | {r["dialogue_turns"]} | {r["active_days"]} | {r["photo_records"]} | {r["connected_calls"]} | {r["risk_reasons"].replace("|", "；") or "无规则提示（不等于身份核验）"} |')
    lines.extend(['', '## 5. 测试嫌疑与评审', '',
                  '以下阈值是审计启发式，不是测试账号事实：测试话术至少20次且占比≥5%；单日≥500轮；单句重复≥15%；设备关联多个拍照账号。',
                  '多个账号可能是家庭共享或历史绑定；高重复可能是儿童反复点播。不得仅凭这些指标删除用户。',
                  'candidate_review_evidence.json保存本地复核证据；其中对话是个人数据，勿作为公开材料传播。核心初选按原排名，不以未确认风险自动替换。', '',
                  '补充复核检查同时间戳密集写入：同一天、至少3个时间戳各有10条以上请求、独立文本占比不超过20%时标记批量测试/数据回放嫌疑。仅作为复核证据，不自动认定测试身份。',
                  '符合该组合的设备：' + ('、'.join(r['device_id'] for r in review_rows if r['batch_pattern_suspect']) or '无') + '。详见 [复核决策](review_decisions.csv) 和 temporal_review_evidence.json。', '',
                  '## 6. 交付文件与字段', '',
                  '- [前100位候选](candidates_100.csv)：保留排名、有效对话/天数/拍照/通话、各来源原始数量、风险与状态。',
                  '- [核心50位初选](core_50_provisional.csv)：前50位子集，需业务确认测试账号后冻结正式名单。',
                  '- [全量设备排名](all_device_ranking.csv)：所有具有有效对话的设备。',
                  '- [各设备来源计数](device_source_counts.csv)：包括没有有效对话但有其他来源的设备。',
                  '- [导出清单](export_manifest.csv)：每设备四类数据实际分包数量。',
                  '- [候选数据目录](candidate_data/)：每设备四个jsonl.gz文件，图片/音频只保留链接，不包含媒体文件。',
                  '- [运行审计](run_summary.json)、[验证结果](validation.json)：文件哈希、分流、枚举、耗时、配置与校验结果。',
                  '- audit.sqlite3：去重依据、设备对话频次和按日统计；*.strings.sqlite3与*.staging.jsonl.gz为本地处理中间产物。', '',
                  '### 排名表字段', '',
                  '| 字段 | 含义 |', '|---|---|',
                  '| rank / device_id | 按已确认口径得到的名次 / 设备分组标识 |',
                  '| dialogue_turns / active_days | 符合口径的用户文本记录数 / 有效对话自然日数 |',
                  '| photo_records / connected_calls | 通过元数据校验的拍照数量 / 满足接通判断的会话数量 |',
                  '| dialogue_raw / photo_raw / call_raw / location_raw | 明确关联本设备的各类原始行数，含未参与排名记录 |',
                  '| location_eligible | 通过本次坐标、精度和时间基础校验的定位数，不参与排名 |',
                  '| first_dialogue / last_dialogue | 有效用户文本的首次 / 最后观察时间 |',
                  '| max_daily_turns | 单个自然日内的有效对话最大值 |',
                  '| test_phrase_count / top_text_count | 测试话术规则命中数 / 最常出现的同一规范文本次数 |',
                  '| linked_user_ids | 拍照记录中与设备共同出现的账号，不表示已确认真实用户身份 |',
                  '| risk_reasons | 风险提示，竖线分隔；空值仅表示没有命中已配置规则 |',
                  '| selection | core_provisional为核心初选，reserve为候补 |',
                  '| review_status | 是否需业务复核；没有规则提示也不等于完成身份核验 |', '',
                  '复核表中unique_user_texts是规范化后的独立文本数；max_same_timestamp是相同时间戳的最大有效行数；timestamps_ge10是出现至少10行的时间戳数量；batch_pattern_suspect为批量模式嫌疑，decision/reason为暂定处理和依据。', '',
                  '### 数据包字段', '',
                  '| 字段 | 含义 |', '|---|---|',
                  '| source_kind / source_file | 来源类别及原始文件名 |',
                  '| sheet / row | 原工作表、Excel行号，表头为第1行 |',
                  '| disposition / flags | 排名有效性分流及附加风险标签 |',
                  '| occurred_at | 标准化观察时间，包含时区 |',
                  '| device_ids | 明确关联的设备列表，通话可有两个设备 |',
                  '| data | 原始列值，字符串保存；Excel日期序列原值未覆盖 |', '',
                  'CSV采用UTF-8，若Excel直接打开乱码，请通过“从文本/CSV”选择UTF-8导入；文件中设备ID按文本使用。', '',
                  '## 7. 待确认及后续边界', '',
                  '业务方确认测试设备名单、Excel本地时区、同主键冲突以及多账号绑定含义。确认后在100位候选内顺序替补，冻结正式50位名单。',
                  '本阶段交付可复核的初筛和数据分包，不将业务身份确认未完成报告为已完成。后续维度规范、小样本提取与媒体下载均未执行。', ''])
    (root / '阶段一数据审计与核心用户初筛报告.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    if validation['errors']:
        raise RuntimeError(validation['errors'][:10])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    make_report(parser.parse_args().output)
