"""J20 阶段一：流式工作簿审计、设备排名及候选原始数据分包。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import posixpath
import re
import sqlite3
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Any
from urllib.parse import urlparse

NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
LOG = logging.getLogger(__name__)


def dump(path: Path, value: Any) -> None:
    """以 UTF-8/LF 保存机器可读结果。"""
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


class StreamBook:
    """共享字符串落盘，逐行解析 XML；不依赖 Excel 全表载入。"""

    def __init__(self, path: Path, cache: Path) -> None:
        self.path = path
        self.zip = zipfile.ZipFile(path)
        self.db = sqlite3.connect(cache)
        self.db.execute('PRAGMA journal_mode=OFF')
        self.db.execute('PRAGMA cache_size=-8192')
        self.db.execute('CREATE TABLE IF NOT EXISTS strings(id INTEGER PRIMARY KEY,value TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT)')
        signature = f'{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}'
        saved = self.db.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
        if saved != (signature,):
            self.db.execute('DELETE FROM strings')
            batch = []
            if 'xl/sharedStrings.xml' in self.zip.namelist():
                with self.zip.open('xl/sharedStrings.xml') as handle:
                    context = ET.iterparse(handle, events=('start', 'end'))
                    _, root = next(context)
                    index = 0
                    for event, element in context:
                        if event == 'end' and element.tag == NS + 'si':
                            value = ''.join(t.text or '' for t in element.iter(NS + 't'))
                            batch.append((index, value))
                            index += 1
                            root.clear()
                            if len(batch) >= 10000:
                                self.db.executemany('INSERT INTO strings VALUES (?,?)', batch)
                                batch.clear()
                    self.db.executemany('INSERT INTO strings VALUES (?,?)', batch)
            self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', ('signature', signature))
            self.db.commit()
        self.string = lru_cache(maxsize=32768)(self._string)

    def _string(self, index: int) -> str:
        row = self.db.execute('SELECT value FROM strings WHERE id=?', (index,)).fetchone()
        if row is None:
            raise ValueError(f'共享字符串索引不存在: {index}')
        return row[0]

    def sheets(self) -> list[tuple[str, str]]:
        rels = ET.fromstring(self.zip.read('xl/_rels/workbook.xml.rels'))
        mapping = {r.attrib['Id']: r.attrib['Target'] for r in rels}
        book = ET.fromstring(self.zip.read('xl/workbook.xml'))
        result = []
        for sheet in book.findall('.//' + NS + 'sheet'):
            target = mapping[sheet.attrib[REL + 'id']]
            target = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
            result.append((sheet.attrib['name'], target))
        return result

    def rows(self) -> Iterator[tuple[str, int, dict[str, str | None]]]:
        for name, target in self.sheets():
            headers = None
            with self.zip.open(target) as handle:
                context = ET.iterparse(handle, events=('start', 'end'))
                parent = None
                for event, element in context:
                    if event == 'start' and element.tag == NS + 'sheetData':
                        parent = element
                    if event != 'end' or element.tag != NS + 'row':
                        continue
                    values = {}
                    for cell in element:
                        ref = cell.attrib['r']
                        col = 0
                        for letter in re.match(r'[A-Z]+', ref).group():
                            col = col * 26 + ord(letter) - 64
                        node = cell.find(NS + 'v')
                        value = node.text if node is not None else None
                        if cell.attrib.get('t') == 's' and value is not None:
                            value = self.string(int(value))
                        elif cell.attrib.get('t') == 'inlineStr':
                            value = ''.join(t.text or '' for t in cell.iter(NS + 't'))
                        values[col - 1] = value
                    number = int(element.attrib['r'])
                    if headers is None:
                        headers = [values.get(i) for i in range(max(values, default=-1) + 1)]
                        if not headers or any(not key for key in headers) or len(set(headers)) != len(headers):
                            raise ValueError(f'非法表头: {name} {headers}')
                        LOG.info('%s / %s 表头: %s', self.path.name, name, headers)
                    else:
                        if any(i >= len(headers) for i in values):
                            raise ValueError(f'超出表头的列: {name}:{number}')
                        yield name, number, {key: values.get(i) for i, key in enumerate(headers)}
                    if parent is not None:
                        parent.clear()

    def close(self) -> None:
        self.string.cache_clear()
        self.zip.close()
        self.db.close()


def inspect(inputs: Path, out: Path, limit: int) -> None:
    """读取有限样例及枚举，供核对字段方向；不访问媒体。"""
    result = {}
    for path in sorted(inputs.glob('J20*.xlsx')):
        LOG.info('检查 %s', path.name)
        book = StreamBook(path, out / (path.stem + '.strings.sqlite3'))
        samples = []
        enums: dict[str, Counter] = defaultdict(Counter)
        count = 0
        for sheet, row, record in book.rows():
            count += 1
            for key in ('status', 'del_flag', 'source', 'provoke_type', 'state', 'end_reason', 'caller_type', 'callee_type', 'loc_type'):
                if key in record:
                    enums[key][str(record[key])] += 1
            if len(samples) < 8 or (record.get('text') and len(samples) < 20):
                samples.append({'sheet': sheet, 'row': row, 'data': record})
            if count >= limit:
                break
        result[path.name] = {'examined': count, 'sheets': book.sheets(), 'enums': enums, 'samples': samples}
        book.close()
    dump(out / 'inspection.json', result)


def normalize(value: Any) -> str:
    return re.sub(r'\s+', ' ', str(value or '')).strip()


def parse_time(value: Any, offset: int = 8) -> tuple[datetime | None, str]:
    """识别 Unix 秒/毫秒、Excel 日期序列及 ISO 日期，返回本地观察时间。"""
    if value is None or not str(value).strip():
        return None, 'missing'
    tz = timezone(timedelta(hours=offset))
    try:
        number = float(value)
        if number >= 1e12:
            return datetime.fromtimestamp(number / 1000, tz), 'unix_ms'
        if number >= 1e9:
            return datetime.fromtimestamp(number, tz), 'unix_s'
        if 30000 <= number <= 100000:
            return datetime(1899, 12, 30, tzinfo=tz) + timedelta(days=number), 'excel_serial_assumed_local'
    except (TypeError, ValueError, OverflowError, OSError):
        pass
    try:
        dt = datetime.fromisoformat(str(value))
        return (dt.replace(tzinfo=tz) if dt.tzinfo is None else dt.astimezone(tz)), 'iso'
    except ValueError:
        return None, 'invalid'


def devices_for(kind: str, row: dict[str, Any]) -> list[str]:
    if kind == 'call':
        return sorted({normalize(row.get(side + '_id')) for side in ('caller', 'callee')
                       if row.get(side + '_type') == 'device' and normalize(row.get(side + '_id'))})
    value = normalize(row.get('device_id'))
    return [value] if value else []


def classify(kind: str, row: dict[str, Any], config: dict) -> tuple[str, list[str], datetime | None]:
    """按保守口径决定是否参与排名；原始行仍然保留。"""
    flags = []
    key = {'dialogue': 'create_time', 'photo': 'create_time', 'call': 'started_at', 'location': 'timestamp'}[kind]
    dt, _ = parse_time(row.get(key), config['timezone_offset'])
    if not devices_for(kind, row):
        return 'missing_device', flags, dt
    if any(not re.fullmatch(config['device_pattern'], d) for d in devices_for(kind, row)):
        return 'invalid_device', flags, dt
    if dt is None or not config['min_date'] <= dt.date().isoformat() <= config['max_date']:
        return 'invalid_time', flags, dt
    if kind in ('dialogue', 'photo'):
        if row.get('del_flag') not in ('0', '0.0'):
            return 'deleted_or_unknown_delete', flags, dt
    if kind == 'dialogue':
        if row.get('status') not in config['dialogue_statuses']:
            return 'unknown_dialogue_status', flags, dt
        value = normalize(row.get('text'))
        if not value:
            return ('audio_only' if row.get('audio') else 'no_user_text'), flags, dt
        if value in config['system_texts']:
            return 'system_text', flags, dt
        if re.search(config['test_text_pattern'], value, re.I):
            flags.append('test_phrase')
        return 'eligible', flags, dt
    if kind == 'photo':
        url = normalize(row.get('picture_url'))
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc:
            return 'invalid_url', flags, dt
        if re.search(config['asset_url_pattern'], parsed.path, re.I):
            return 'system_asset_suspect', flags, dt
        return 'eligible', flags, dt
    if kind == 'call':
        connected, _ = parse_time(row.get('connected_at'), config['timezone_offset'])
        ended, _ = parse_time(row.get('ended_at'), config['timezone_offset'])
        if connected is None:
            return 'not_connected', flags, dt
        tolerance = timedelta(seconds=1)
        if connected < dt - tolerance or (ended and ended < connected - tolerance):
            return 'invalid_call_sequence', flags, dt
        if row.get('state') not in ('ENDED', 'CONNECTED'):
            return 'unknown_call_state', flags, dt
        return 'eligible', flags, dt
    try:
        lat, lon = float(row.get('lat')), float(row.get('lon'))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return 'invalid_coordinate', flags, dt
        accuracy = float(row.get('accuracy'))
        if accuracy < 0:
            return 'invalid_accuracy', flags, dt
        if accuracy > config['accuracy_warning']:
            flags.append('low_accuracy')
    except (ValueError, TypeError):
        return 'invalid_coordinate_or_accuracy', flags, dt
    return 'eligible', flags, dt


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        for row in rows:
            safe = {k: ("'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@')) else v)
                    for k, v in row.items()}
            writer.writerow(safe)


def rank_stats(stats: dict[str, dict]) -> list[dict]:
    rows = []
    for device, s in stats.items():
        if not s['counts']['dialogue_eligible']:
            continue
        rows.append({'device_id': device, 'dialogue_turns': s['counts']['dialogue_eligible'],
                     'active_days': len(s['days']), 'photo_records': s['counts']['photo_eligible'],
                     'connected_calls': s['counts']['call_eligible']})
    return sorted(rows, key=lambda s: (-s['dialogue_turns'], -s['active_days'],
                                      -s['photo_records'], -s['connected_calls'], s['device_id']))


def scan(inputs: Path, out: Path, config: dict) -> None:
    """全量扫描并按候选设备导出原始行，保留筛选状态和来源。"""
    started = time.perf_counter()
    if (out / 'audit.sqlite3').exists():
        raise FileExistsError('审计库已存在，请使用新的输出目录，防止覆盖或重复计数')
    db = sqlite3.connect(out / 'audit.sqlite3')
    db.executescript('''
        PRAGMA journal_mode=WAL;
        PRAGMA cache_size=-16384;
        CREATE TABLE seen(kind TEXT,key BLOB,fingerprint BLOB,PRIMARY KEY(kind,key)) WITHOUT ROWID;
        CREATE TABLE utterances(device TEXT,text TEXT,n INTEGER,PRIMARY KEY(device,text)) WITHOUT ROWID;
        CREATE TABLE dates(device TEXT,day TEXT,n INTEGER,PRIMARY KEY(device,day)) WITHOUT ROWID;
    ''')
    stats: dict[str, dict] = defaultdict(lambda: {'counts': Counter(), 'days': set(), 'users': set(),
                                                'first': None, 'last': None})
    audits = {}
    files = {}
    enums: dict[str, Counter] = defaultdict(Counter)
    candidate_ids: set[str] = set()
    bundles = out / 'candidate_data'
    bundles.mkdir(exist_ok=True)
    kinds = [('dialogue', '*智能体对话*.xlsx'), ('photo', '*拍照*.xlsx'),
             ('call', '*通话*.xlsx'), ('location', '*定位*.xlsx')]
    selected_handles = {}
    export_counts: Counter = Counter()
    for kind, pattern in kinds:
        matching = list(inputs.glob('J20' + pattern))
        if len(matching) != 1:
            raise ValueError(f'文件匹配不唯一: {pattern}')
        path = matching[0]
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        files[kind] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
        if kind == 'location':
            candidate_ids = {r['device_id'] for r in rank_stats(stats)[:config['candidate_count']]}
            for device in candidate_ids:
                folder = bundles / device
                folder.mkdir(exist_ok=True)
                selected_handles[device] = gzip.open(folder / 'location.jsonl.gz', 'wt', encoding='utf-8', newline='\n', compresslevel=1)
        book = StreamBook(path, out / (path.stem + '.strings.sqlite3'))
        audit = {'rows': 0, 'sheets': Counter(), 'dispositions': Counter(), 'flags': Counter(),
                 'nulls': Counter(), 'time_encodings': Counter(), 'first': None, 'last': None}
        spool = gzip.open(out / (kind + '.staging.jsonl.gz'), 'wt', encoding='utf-8', newline='\n', compresslevel=1) if kind != 'location' else None
        LOG.info('开始全量扫描 %s', kind)
        for sheet, number, row in book.rows():
            audit['rows'] += 1
            audit['sheets'][sheet] += 1
            for key, value in row.items():
                if value is None or value == '':
                    audit['nulls'][key] += 1
            for key in ('status', 'del_flag', 'source', 'state', 'end_reason', 'caller_type', 'callee_type', 'loc_type'):
                if key in row:
                    enums[kind + '.' + key][str(row[key])] += 1
            disposition, flags, dt = classify(kind, row, config)
            time_key = {'dialogue': 'create_time', 'photo': 'create_time', 'call': 'started_at', 'location': 'timestamp'}[kind]
            audit['time_encodings'][parse_time(row.get(time_key), config['timezone_offset'])[1]] += 1
            if dt:
                stamp = dt.isoformat()
                audit['first'] = min(audit['first'] or stamp, stamp)
                audit['last'] = max(audit['last'] or stamp, stamp)
            encoded = json.dumps(row, ensure_ascii=False, separators=(',', ':'))
            fingerprint = hashlib.sha256(encoded.encode('utf-8')).digest()
            primary = row.get({'dialogue': 'dialog_id', 'photo': 'picture_id', 'call': 'video_id', 'location': 'id'}[kind])
            natural = json.dumps([row.get('device_id'), primary], separators=(',', ':')) if kind == 'dialogue' and primary else primary
            key = hashlib.sha256(str(natural).encode('utf-8')).digest() if natural else fingerprint
            previous = db.execute('SELECT fingerprint FROM seen WHERE kind=? AND key=?', (kind, key)).fetchone()
            if previous:
                disposition = 'duplicate' if previous[0] == fingerprint else 'key_conflict'
            else:
                db.execute('INSERT INTO seen VALUES (?,?,?)', (kind, key, fingerprint))
                if not primary:
                    flags.append('missing_primary_key')
            audit['dispositions'][disposition] += 1
            audit['flags'].update(flags)
            devices = devices_for(kind, row)
            for device in devices:
                s = stats[device]
                s['counts'][kind + '_raw'] += 1
                s['counts'][kind + '_' + disposition] += 1
                if kind == 'photo' and row.get('user_id'):
                    s['users'].add(row['user_id'])
                if kind == 'dialogue' and disposition == 'eligible':
                    day = dt.date().isoformat()
                    s['days'].add(day)
                    stamp = dt.isoformat()
                    s['first'] = min(s['first'] or stamp, stamp)
                    s['last'] = max(s['last'] or stamp, stamp)
                    s['counts'].update(flags)
                    db.execute('INSERT INTO dates VALUES (?,?,1) ON CONFLICT(device,day) DO UPDATE SET n=n+1', (device, day))
                    db.execute('INSERT INTO utterances VALUES (?,?,1) ON CONFLICT(device,text) DO UPDATE SET n=n+1', (device, normalize(row['text'])))
            package = {'source_kind': kind, 'source_file': path.name, 'sheet': sheet, 'row': number,
                       'disposition': disposition, 'flags': flags, 'occurred_at': dt.isoformat() if dt else None,
                       'device_ids': devices, 'data': row}
            if spool:
                spool.write(json.dumps(package, ensure_ascii=False) + '\n')
            elif candidate_ids.intersection(devices):
                line = json.dumps(package, ensure_ascii=False) + '\n'
                for device in candidate_ids.intersection(devices):
                    selected_handles[device].write(line)
                    export_counts[(device, kind)] += 1
            if audit['rows'] % config['progress_every'] == 0:
                db.commit()
                LOG.info('%s 已扫描 %s 行，累计 %.1fs', kind, audit['rows'], time.perf_counter() - started)
        if spool:
            spool.close()
        book.close()
        db.commit()
        assert sum(audit['dispositions'].values()) == audit['rows']
        audits[kind] = audit
        dump(out / 'audit_progress.json', {'files': files, 'audit': audits, 'enums': enums})
    for handle in selected_handles.values():
        handle.close()
    ranked = rank_stats(stats)
    for rank, entry in enumerate(ranked, 1):
        device = entry['device_id']
        s = stats[device]
        maximum = db.execute('SELECT max(n) FROM dates WHERE device=?', (device,)).fetchone()[0]
        texts = db.execute('SELECT text,n FROM utterances WHERE device=? ORDER BY n DESC,text LIMIT 5', (device,)).fetchall()
        test_count = s['counts']['test_phrase']
        reasons = []
        if test_count >= config['test_min_count'] and test_count / entry['dialogue_turns'] >= config['test_ratio']:
            reasons.append('重复测试话术')
        if maximum >= config['dense_daily_turns']:
            reasons.append('单日异常高频需复核')
        if texts and texts[0][1] / entry['dialogue_turns'] >= config['repeat_ratio']:
            reasons.append('单句重复占比高')
        if len(s['users']) > 1:
            reasons.append('设备关联多个账号')
        entry.update({'rank': rank, 'dialogue_raw': s['counts']['dialogue_raw'],
                      'photo_raw': s['counts']['photo_raw'], 'call_raw': s['counts']['call_raw'],
                      'location_raw': s['counts']['location_raw'], 'location_eligible': s['counts']['location_eligible'],
                      'first_dialogue': s['first'], 'last_dialogue': s['last'], 'max_daily_turns': maximum,
                      'test_phrase_count': test_count, 'top_text_count': texts[0][1] if texts else 0,
                      'linked_user_ids': '|'.join(sorted(s['users'])), 'risk_reasons': '|'.join(reasons),
                      'selection': 'core_provisional' if rank <= config['core_count'] else 'reserve',
                      'review_status': 'pending_business_confirmation' if reasons else 'no_rule_flag_not_identity_verified'})
    candidates = ranked[:config['candidate_count']]
    write_csv(out / 'all_device_ranking.csv', ranked)
    write_csv(out / 'candidates_100.csv', candidates)
    write_csv(out / 'core_50_provisional.csv', candidates[:config['core_count']])
    write_csv(out / 'device_source_counts.csv', [dict(device_id=d, **{k:s['counts'][k] for k in
              ['dialogue_raw','dialogue_eligible','photo_raw','photo_eligible','call_raw','call_eligible','location_raw','location_eligible']}) for d,s in sorted(stats.items())])
    review = []
    for entry in candidates:
        d = entry['device_id']
        review.append({'device_id': d, 'rank': entry['rank'], 'risk_reasons': entry['risk_reasons'],
                       'top_texts': db.execute('SELECT text,n FROM utterances WHERE device=? ORDER BY n DESC,text LIMIT 15', (d,)).fetchall(),
                       'test_examples': db.execute("SELECT text,n FROM utterances WHERE device=? AND (text LIKE '%测试%' OR text LIKE '%test%' OR text LIKE '%调试%') ORDER BY n DESC LIMIT 10", (d,)).fetchall()})
    dump(out / 'candidate_review_evidence.json', review)
    for kind in ('dialogue', 'photo', 'call'):
        handles = {d: gzip.open(bundles / d / (kind + '.jsonl.gz'), 'wt', encoding='utf-8', newline='\n', compresslevel=1) for d in candidate_ids}
        with gzip.open(out / (kind + '.staging.jsonl.gz'), 'rt', encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                for d in candidate_ids.intersection(record['device_ids']):
                    handles[d].write(line)
                    export_counts[(d, kind)] += 1
        for handle in handles.values():
            handle.close()
    manifest = []
    for entry in candidates:
        d = entry['device_id']
        item = {'device_id': d, 'rank': entry['rank'], 'selection': entry['selection']}
        for kind, _ in kinds:
            n = export_counts[(d, kind)]
            assert n == stats[d]['counts'][kind + '_raw'], (d, kind, n)
            item[kind + '_exported_rows'] = n
        manifest.append(item)
    write_csv(out / 'export_manifest.csv', manifest)
    dump(out / 'run_summary.json', {'files': files, 'audit': audits, 'enums': enums,
         'devices': len(stats), 'rankable_devices': len(ranked), 'candidates': len(candidates),
         'core_provisional': min(len(candidates), config['core_count']), 'elapsed_seconds': time.perf_counter() - started,
         'config': config, 'status': 'completed_provisional_pending_business_review'})
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()
    LOG.info('完成：%s设备，%s候选，%s核心初选；耗时%.1fs', len(stats), len(candidates), min(len(candidates), config['core_count']), time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--limit', type=int, default=3000)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.inspect:
        inspect(args.input, args.output, args.limit)
    else:
        if not args.config:
            parser.error('全量扫描需要 --config')
        config = json.loads(args.config.read_text(encoding='utf-8'))
        scan(args.input, args.output, config)


if __name__ == '__main__':
    main()
