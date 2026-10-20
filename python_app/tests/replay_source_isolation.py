"""只读回放已保存模型候选，验证协议链路；不代表新的来源审核通过。"""
import copy
import hashlib
import json
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from memory_agent import BASE, load_inputs
from agent_context import windows
from agent_data import Registry
from agent_contracts import validate_candidate


def main():
    started = time.perf_counter(); tracemalloc.start()
    root = BASE / 'output/memory_agent_seed_8EAF_0907_management/12354838B6678EAF'
    paths = [root / n for n in ('report.json', 'memory.sqlite3', 'model_calls.jsonl')]
    hashes = lambda: {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    cfg = json.loads((BASE / 'python_app/config/memory_agent_seed_8EAF_0907_management.json').read_text(encoding='utf-8'))
    sessions, _ = load_inputs(cfg, '12354838B6678EAF')
    registry = Registry(BASE / cfg['rules_file'])
    by_evidence = {t['evidence_id']: s for s in sessions for t in s['turns']}
    counts = Counter(); errors = Counter(); passed_photos = set(); details = []
    with paths[2].open(encoding='utf-8') as stream:
        for line in stream:
            call = json.loads(line)
            if not call['role'].startswith('source_extractor:'):
                continue
            try:
                response = json.loads(call['raw_output'])
            except (ValueError, KeyError):
                counts['unparsed_calls'] += 1; continue
            for raw in response.get('candidates', []):
                counts['raw_candidate_outputs_including_retries'] += 1
                ids = {e.get('evidence_id') for e in raw.get('evidence', []) if isinstance(e,dict)}
                known = [by_evidence[e] for e in ids if e in by_evidence]
                if not known or len({s['session_id'] for s in known}) != 1:
                    errors['证据不能唯一定位会话'] += 1; continue
                s = known[0]; failure = '证据未落在单一来源窗口'
                for w in windows(s, cfg['session_chunk_tokens']):
                    if not ids <= {t['evidence_id'] for t in w['turns']}:
                        continue
                    try:
                        validate_candidate(copy.deepcopy(raw), call['role'].split(':',1)[1], registry, w, 'replay')
                        counts['protocol_valid'] += 1
                        if s.get('source_kind') == 'photo': passed_photos.add(s['session_id'])
                        failure = None; break
                    except (ValueError, KeyError, TypeError) as exc:
                        failure = str(exc)
                if failure:
                    errors[failure] += 1
                    details.append(dict(call=call['call'], session_id=s['session_id'], reason=failure))
    assert hashes() == before, '旧产物发生变化'
    report = dict(mode='offline_saved_output_schema_replay', real_model_calls=0, counts=dict(counts),
                  photo_sessions_protocol_valid=len(passed_photos), remaining_errors=dict(errors), details=details,
                  old_artifact_sha256=before, old_artifacts_unchanged=True,
                  elapsed_seconds=round(time.perf_counter()-started,3), python_peak_bytes=tracemalloc.get_traced_memory()[1],
                  limitation='协议通过不等于来源审核通过；不调用模型，不修改旧候选、审核结论或记忆库。')
    output = BASE / 'output/source_isolation_offline_validation'; output.mkdir(exist_ok=True)
    (output / 'saved_output_replay.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('details','old_artifact_sha256')},ensure_ascii=False))


if __name__ == '__main__': main()
