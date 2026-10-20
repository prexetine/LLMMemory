"""v4离线链路回归：模拟模型语义输出，不把模拟通过当作线上质量验收。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import sys
import io
from types import SimpleNamespace
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_source_agent_v3 import BASE, session, candidate as old_candidate
from agent_data import Registry, digest
from agent_store import MemoryStore
from agent_maintenance import V4Store, applies, storage_control, suspend, apply_decision, relevant
from agent_contracts import CHECKS, VERSION, BASE_FIELDS, META, validate_candidate, validate_signal, normalize_signal, source_result, seal, subject_id
from legacy_source_v4 import SourcePipeline, EXTRACT
from legacy_memory_agent import CentralAgent, policy_version, run_one
from agent_schema import response_schema
from export_memory_html import build_data
from memory_view_data import build_report_user


def raw(s, claim="喜欢恐龙", **changes):
    c = old_candidate(s, claim, subject="我")
    c = {k: v for k, v in c.items() if k in BASE_FIELDS | META}
    c.update(subject_role="current_user", assertion_mode="self_report", subject_key=None, attribute_key=None)
    c.update(changes)
    return c


def verdict(c, decision="pass", repair="none"):
    return dict(candidate_id=c["candidate_id"], decision=decision, checks={k: "pass" for k in CHECKS},
                reason="原话支持", evidence_ids=[e["evidence_id"] for e in c["evidence"]], needs_context=False,
                context_ids=[], repair={"kind": repair, "reason": "按独立喜好拆分" if repair != "none" else ""})


def control(s, quote, kind="withdrawal", target="session", text="", subject=None, scope="session"):
    r = dict(kind=kind, subject_key=subject, attribute_key=None, scope=scope,
             evidence=[dict(evidence_id=s["turns"][0]["evidence_id"], quote=quote)], target_type=target, target_text=text)
    validate_signal(r, s)
    return normalize_signal(r, s)


class Fake:
    def __init__(self):
        self.calls = []
        self.items = None
        self.signals = []
        self.repair = False
        self.action = "coexist"
        self.reject = False

    def ask(self, role, prompt, data):
        self.calls.append((role, copy.deepcopy(data)))
        assert data["policy"] == VERSION
        if role == "source_router":
            signals = [{k:v for k,v in s.items() if k not in ("device_id", "session_id")} for s in self.signals]
            return dict(skills=["interests_preferences"], signals=signals, reason="测试")
        if role.startswith("source_extractor:"):
            items = self.items if self.items is not None else [raw(data["session"])]
            return dict(candidates=items if any(d["dimension_id"] == "G16" for d in data["skill_rules"]) else [], notes="")
        if role == "source_reviewer":
            return {"results": [verdict(c, "reject" if self.reject else "pending" if self.repair and not c.get("parent_candidate_id") else "pass",
                                         "split" if self.repair and not c.get("parent_candidate_id") else "none") for c in data["candidates"]]}
        if role == "source_repair":
            s = dict(device_id="test-user", session_id="s1", turns=data["evidence_pack"])
            return {"results": [dict(parent_candidate_id=p["candidate"]["candidate_id"], candidates=[raw(s, "喜欢恐龙"), raw(s, "喜欢汽车")], reason="拆分") for p in data["parents"]]}
        if role == "history_compare":
            return {"results": [dict(candidate_id=c["candidate_id"], action=self.action,
                      targets=[dict(memory_id=data["history"][0]["memory_id"], revision=data["history"][0]["revision"])],
                      reason="模拟历史判断", basis=data["signals"][0]["evidence"] if self.action == "supersede" else []) for c in data["candidates"]]}
        raise AssertionError(role)


class V4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.store = MemoryStore(self.out / "memory.sqlite3")
        self.addCleanup(self.store.close)
        self.cfg = json.loads((BASE / "python_app/config/memory_agent_source_v3.json").read_text(encoding="utf-8"))
        self.cfg["review_strategy"] = "source_v4"
        self.registry = Registry(BASE / self.cfg["rules_file"])
        self.fake = Fake()
        self.agent = CentralAgent(self.fake, self.registry, self.store, self.cfg, self.out / "trace.jsonl")

    def run_session(self, s=None):
        return self.agent.run_session(s or session(text="我喜欢恐龙呀。"))

    def normalized(self, s, **kw):
        return validate_candidate(raw(s, **kw), "interests_preferences", self.registry, s, "w")

    def state(self):
        return V4Store(self.store).snapshot("test-user")

    def test_default_user_no_name_and_resume(self):
        s = session(text="我喜欢恐龙呀。")
        r = self.run_session(s)
        self.assertEqual(len(r["approved"]), 1)
        self.assertEqual(r["candidates"][0]["subject_key"], subject_id("test-user"))
        self.assertEqual(r["candidates"][0]["attribute_key"], "G16")
        self.assertEqual(set(r["reviews"][0]["checks"]), set(CHECKS))
        count = len(self.fake.calls)
        self.assertEqual(self.run_session(s), r)
        self.assertEqual(len(self.fake.calls), count)
        self.run_session(session("s2", "我喜欢恐龙呀。"))
        memory = next(iter(self.state()["memories"].values()))
        self.assertEqual(memory["support_count"], 2)
        self.assertEqual(len(self.state()["memories"]), 1)

    def test_split_positive_negative_and_no_mechanical_split(self):
        cases = [("我喜欢恐龙和汽车", ["喜欢恐龙", "喜欢汽车"]),
                 ("我喜欢恐龙，不喜欢汽车", ["喜欢恐龙", "不喜欢汽车"]),
                 ("我喜欢和爸爸一起搭积木", ["喜欢和爸爸一起搭积木"]),
                 ("比起汽车更喜欢恐龙", ["比起汽车更喜欢恐龙"])]
        for n, (text, claims) in enumerate(cases):
            s = session("split" + str(n), text)
            self.fake.items = [raw(s, c) for c in claims]
            r = self.run_session(s)
            self.assertEqual([c["claim"] for c in r["approved"]], claims)
        self.assertIn("不机械", EXTRACT)

    def test_parent_repair_once_and_new_proofs(self):
        s = session(text="我喜欢恐龙和汽车")
        self.fake.items = [raw(s, "喜欢恐龙和汽车")]
        self.fake.repair = True
        r = self.run_session(s)
        self.assertEqual(len(r["approved"]), 2)
        self.assertEqual(sum(role == "source_repair" for role, _ in self.fake.calls), 1)
        self.assertEqual(len(r["candidates"]), 2)
        self.assertEqual(len({c["parent_candidate_id"] for c in r["approved"]}), 1)
        self.assertTrue(all(c["candidate_version"] == 2 for c in r["approved"]))
        self.assertEqual(len({p["proof_hash"] for p in r["reviews"]}), 2)
        self.assertTrue(all(m["support_count"] == 1 for m in self.state()["memories"].values()))

    def test_other_and_unresolved_subjects(self):
        s = session(text="爸爸喜欢汽车，我喜欢恐龙。")
        father = self.normalized(s, subject="我爸爸", subject_role="other", claim="爸爸喜欢汽车")
        me = self.normalized(s)
        self.assertNotEqual(father["subject_key"], me["subject_key"])
        self.assertEqual(father["subject_key"], subject_id("test-user", "爸爸"))
        unknown = self.normalized(s, subject="他", subject_role="other")
        with self.assertRaises(ValueError):
            source_result({"results": [verdict(unknown)]}, [unknown], {s["turns"][0]["evidence_id"]})

    def test_explicit_speaker_switch_isolated(self):
        s = session(text="我是小明的姐姐，我喜欢恐龙。")
        self.fake.signals = [control(s, "我是小明的姐姐", kind="subject_switch", target="unresolved", subject="小明的姐姐")]
        r = self.run_session(s)
        self.assertEqual(r["approved"][0]["subject_role"], "other")
        self.assertNotEqual(r["approved"][0]["subject_key"], subject_id("test-user"))

    def test_request_and_temporary_do_not_become_long_term(self):
        for n, (text, mode) in enumerate((("想听恐龙故事", "request"), ("今天不想上学", "temporary"))):
            s = session("time" + str(n), text)
            self.fake.items = [raw(s, text, assertion_mode=mode)]
            r = self.run_session(s)
            self.assertEqual(r["approved"][0]["assertion_mode"], mode)
            self.assertIsNone(r["approved"][0]["valid_from"])

    def test_context_answer_and_missing_evidence(self):
        s = session(text="恐龙")
        s["turns"].insert(0, session("question", "你问吧")["turns"][0])
        s["turns"][0]["assistant_text"] = "你喜欢什么动物？"
        c = self.normalized(s, evidence=[dict(evidence_id="s1:1", quote="恐龙")])
        pack = SourcePipeline(self.agent, s).pack([c])
        self.assertEqual(len(pack), 2)
        with self.assertRaises(ValueError):
            self.normalized(s, evidence=[dict(evidence_id="s1:1", quote="你喜欢什么动物")])

    def test_false_withdrawals_blocked(self):
        for text in ("暂停。", "关机", "以后再讲吧", "不要讲英文", "忘了那个人叫啥了", "我刚才说的想不起来了"):
            s = session(text=text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                control(s, text)

    def test_controls_respect_item_session_and_all(self):
        s = session(text="不要记住我喜欢恐龙，其他的可以记。")
        c = self.normalized(s)
        item = control(s, s["turns"][0]["user_text"], target="item", text="喜欢恐龙")
        self.assertTrue(applies(item, c))
        self.assertFalse(applies(item, dict(c, claim="喜欢汽车")))
        self.assertFalse(applies(item, dict(c, origin_session_id="older")))
        self.assertFalse(applies(item, dict(c, subject_key=subject_id("test-user", "爸爸"))))
        unresolved = dict(item, target_type="unresolved", target_text="")
        self.assertFalse(applies(unresolved, c))
        self.assertEqual(storage_control(c, [unresolved])["status"], "unresolved")
        all_s = session(text="全部记忆都不要保存")
        all_control = control(all_s, all_s["turns"][0]["user_text"], target="all")
        self.assertTrue(applies(all_control, dict(c, origin_session_id="older")))
        self.assertFalse(applies(all_control, dict(c, device_id="another")))

    def test_pass_source_but_block_storage_and_html(self):
        s = session(text="我喜欢恐龙，这次聊天不要保存。")
        self.fake.signals = [control(s, s["turns"][0]["user_text"])]
        r = self.run_session(s)
        self.assertEqual(r["reviews"][0]["source_decision"], "pass")
        self.assertEqual(r["maintenance"][0]["storage_control"]["status"], "blocked")
        self.assertEqual(r["approved"], [])
        (self.out / "report.json").write_text(json.dumps(dict(device_id="test-user", status="completed", selected_sessions=1, results=[r], review_version=VERSION)), encoding="utf-8")
        (self.out / "manifest.json").write_text(json.dumps(dict(review_version=VERSION)), encoding="utf-8")
        formal = build_data(self.out, BASE / self.cfg["rules_file"])
        card = build_report_user(self.out, formal)["cards"][0]
        self.assertEqual(card["source_decision"], "pass")
        self.assertEqual(card["storage_control"]["status"], "blocked")
        self.assertEqual(card["claim"], "喜欢恐龙")

    def test_transaction_failure_rollback_and_receipt_recovery(self):
        with patch("agent_maintenance.suspend", side_effect=RuntimeError("disk failure")):
            with self.assertRaises(RuntimeError):
                self.run_session()
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM memories").fetchone()[0], 0)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM sessions").fetchone()[0], 0)
        r = self.run_session()
        count = len(self.fake.calls)
        self.store.db.execute("DELETE FROM context_work")
        self.store.db.commit()
        recovered = self.run_session()
        self.assertEqual(recovered["commit"], r["commit"])
        self.assertEqual(len(self.fake.calls), count)

    def test_schema_and_batch_ids(self):
        s = session()
        c = self.normalized(s)
        v = verdict(c)
        for results in ([], [v, v], [dict(v, candidate_id="extra")]):
            with self.assertRaises(ValueError):
                source_result({"results": results}, [c], {"s1:1"})
        schema = response_schema("source_reviewer", {"policy": VERSION})
        self.assertEqual(set(schema["properties"]["results"]["items"]["properties"]["checks"]["properties"]), set(CHECKS))
        self.assertEqual(policy_version(self.cfg), VERSION)

    def test_reject_fiction_and_asr_pending(self):
        self.fake.reject = True
        r = self.run_session(session(text="故事里的我是一条恐龙"))
        self.assertEqual(r["approved"], [])
        self.assertEqual(r["reviews"][0]["decision"], "reject")
        s = session("asr", "我不、喜欢恐龙")
        c = self.normalized(s)
        v = verdict(c, "pending")
        v["checks"]["semantics"] = "uncertain"
        v["reason"] = "否定词疑似ASR错误，无音频不猜补"
        source_result({"results": [v]}, [c], {"asr:1"})
        v["decision"] = "pass"
        with self.assertRaises(ValueError):
            source_result({"results": [v]}, [c], {"asr:1"})

    def test_unclassified_and_extra_keys(self):
        s = session()
        c = self.normalized(s, dimension_id="UNCLASSIFIED")
        v = verdict(c, "pending")
        source_result({"results": [v]}, [c], {"s1:1"})
        source_result({"results": [verdict(c)]}, [c], {"s1:1"})  # 来源有据不等于已归类入库。
        extra = raw(s);extra["dynamic_key"] = "不允许"
        with self.assertRaises(ValueError):
            validate_candidate(extra, "interests_preferences", self.registry, s, "w")

    def test_late_log_cannot_supersede(self):
        fact = dict(type="attribute", predicate="喜好", object="恐龙", value="喜欢", polarity="positive", cardinality="multiple", time_text="", time_value=None)
        first = session(date="2026-09-03")
        self.fake.items = [raw(first, fact=fact)]
        self.run_session(first)
        s = session("late", "改一下，喜欢恐龙说错了，我不喜欢恐龙", date="2026-09-01")
        self.fake.items = [raw(s, "不喜欢恐龙", fact=dict(fact, polarity="negative"))]
        self.fake.action = "supersede"
        self.fake.signals = [control(s, s["turns"][0]["user_text"], kind="correction", target="item", text="喜欢恐龙")]
        r = self.run_session(s)
        self.assertEqual(r["approved"], [])
        self.assertIn("迟到", r["maintenance"][0]["reason"])

    def test_user_runner_resume_and_failed_session_continuation(self):
        cfg = dict(self.cfg, device_id="test-user")
        cfg.pop("user_selection", None)
        s1,s2 = session("bad", "我喜欢恐龙"),session("good", "我喜欢恐龙")
        fake = Fake()
        original = fake.ask
        def fail(role, prompt, data):
            value = original(role,prompt,data)
            if role.startswith("source_extractor:") and data["session"]["session_id"] == "bad":
                for c in value["candidates"]:c["subject"] = None
            return value
        out = self.out / "run"
        with patch.object(fake,"ask",side_effect=fail), patch("legacy_memory_agent.JsonClient",return_value=fake), patch("legacy_memory_agent.load_inputs",return_value=([s1,s2],{"device_id":"test-user"})), redirect_stdout(io.StringIO()):
            self.assertEqual(run_one(cfg,"live",str(out)),2)
        report = json.loads((out/'report.json').read_text(encoding='utf-8'))
        self.assertEqual([r['status'] for r in report['results']],['completed','completed'])
        self.assertEqual(report['extraction_pending'], 1)
        self.assertTrue(report['results'][0]['needs_attention'])
        with patch("legacy_memory_agent.JsonClient",return_value=Fake()), patch("legacy_memory_agent.load_inputs",return_value=([s1,s2],{"device_id":"test-user"})), redirect_stdout(io.StringIO()):
            self.assertEqual(run_one(cfg,"live",str(out),resume=True),2)

    def test_photo_snapshot_reuse_no_download(self):
        from test_agent_photos import fixture, pixels, config
        from agent_photos import bind_image
        cfg = config();cfg['review_strategy']='source_v4'
        source = self.out/'images';source.mkdir()
        s = fixture(source)
        old = source/'old'/s['device_id'];old.mkdir(parents=True)
        new = source/'new';new.mkdir()
        opener = SimpleNamespace(open=lambda *a, **kw: io.BytesIO(pixels()))
        with patch('agent_photos.urllib.request.build_opener',return_value=opener):
            bind_image(s,old,cfg,self.fake)
        cfg['photo_cache_source']=str(source/'old')
        with patch('agent_photos.urllib.request.build_opener',side_effect=AssertionError('不应下载')):
            bind_image(s,new,cfg,self.fake)
        self.assertEqual(len(list((new/'photo_assets').glob('*.jpg'))),1)
        self.agent.config=cfg
        self.fake.items=[raw(s, '画面为红色',kind='observation',assertion_mode='observation',subject_role='other',subject='照片画面',scope='仅本图')]
        result=self.run_session(s)
        self.assertEqual(len(result['approved']),1)
        self.assertEqual(result['approved'][0]['subject_key'],s['turns'][0]['evidence_id'])

    def test_old_database_rejected_without_migration(self):
        self.store.db.execute("INSERT INTO sessions(batch_id,device_id,session_id) VALUES('old','test-user','old')")
        self.store.db.commit()
        before=list(self.store.db.execute('SELECT name FROM sqlite_master'))
        with self.assertRaises(ValueError):V4Store(self.store)
        self.assertEqual(list(self.store.db.execute('SELECT name FROM sqlite_master')),before)

    def test_compatible_conflict_third_member_and_update(self):
        def birth(s, value):
            return raw(s, "出生日期是" + value, fact=dict(type="attribute", predicate="出生日期", object="本人", value=value, polarity="positive", cardinality="single", time_text="", time_value=None))
        s = session(text="出生日期是2018-01-01")
        self.fake.items = [birth(s, "2018-01-01")]
        self.run_session(s)
        self.fake.action = "conflict"
        s = session("s2", "出生日期是2019-01-01")
        self.fake.items = [birth(s, "2019-01-01")]
        self.run_session(s)
        s = session("s3", "出生日期是2020-01-01")
        self.fake.items = [birth(s, "2020-01-01")]
        self.run_session(s)
        groups = self.state()["groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(next(iter(groups.values()))["members"]), 3)
        s = session("s4", "改一下，出生日期是2018-01-01说错了，是2021-01-01", date="2026-09-02")
        self.fake.items = [birth(s, "2021-01-01")]
        self.fake.action = "supersede"
        self.fake.signals = [control(s, s["turns"][0]["user_text"], kind="correction", target="item", text="出生日期是2018-01-01")]
        self.run_session(s)
        self.assertTrue(any(m["state"] == "superseded" for m in self.state()["memories"].values()))
        self.assertTrue(self.state()["versions"])


if __name__ == "__main__":
    unittest.main()
