"""真实错误片段的确定性离线回归；模拟模型输出，不代表模型准确率。"""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import test_source_agent_v4 as fixtures
from test_chain_facts import fact
from agent_data import digest
from agent_fact_policy import comparable, same_fact, relation_guard
from agent_llm import TransientModelError
from legacy_source_v4 import SourcePipeline
from agent_memory_structure import rebuild_summaries
from agent_contracts import validate_candidate
from agent_maintenance import suspend
from export_memory_html import export_user, build_data
from memory_view_data import build_report_user


def structure(kind="fact", context="real", topic="动物知识", steps=None):
    return dict(type=kind, context=context, topic=topic, steps=steps or [])


class ManagementTests(unittest.TestCase):
    def setUp(self):
        self.h = fixtures.V4Tests()
        self.h.setUp()
        self.addCleanup(self.h.doCleanups)

    def test_repair_timeout_only_blocks_affected_candidate_and_resume_idempotent(self):
        s = fixtures.session(text="我喜欢恐龙和汽车，也喜欢聊天")
        self.h.fake.items = [fixtures.raw(s, "喜欢恐龙和汽车"), fixtures.raw(s, "喜欢聊天")]
        original = self.h.fake.ask
        def ask(role, prompt, data):
            if role == "source_reviewer":
                return {"results": [fixtures.verdict(c, repair="split" if "汽车" in c["claim"] else "none") for c in data["candidates"]]}
            if role == "source_repair":
                raise TransientModelError("TimeoutError重试耗尽")
            return original(role, prompt, data)
        with patch.object(self.h.fake, "ask", side_effect=ask):
            result = self.h.run_session(s)
            self.assertEqual([c["claim"] for c in result["approved"]], ["喜欢聊天"])
            self.assertEqual(result["commit"]["counts"]["pending"], 1)
            self.assertIn("TimeoutError", next(d["reason"] for d in result["maintenance"] if d["action"] == "pending"))
            self.assertEqual(self.h.run_session(s), result)
            self.assertEqual(len(self.h.state()["memories"]), 1)

    def test_review_timeout_batch_does_not_stop_next_batch(self):
        s = fixtures.session(text="我喜欢恐龙和汽车")
        self.h.cfg["source_review_batch_size"] = 1
        self.h.fake.items = [fixtures.raw(s, "喜欢恐龙"), fixtures.raw(s, "喜欢汽车")]
        original = self.h.fake.ask
        def ask(role, prompt, data):
            if role == "source_reviewer" and data["candidates"][0]["claim"] == "喜欢恐龙":
                raise TransientModelError("timeout")
            return original(role, prompt, data)
        with patch.object(self.h.fake, "ask", side_effect=ask):
            r = self.h.run_session(s)
        self.assertEqual([c["claim"] for c in r["approved"]], ["喜欢汽车"])
        self.assertEqual(r["commit"]["counts"]["pending"], 1)

    def test_history_timeout_is_pending_not_new_or_abort(self):
        self.h.run_session()
        s = fixtures.session("s2", "我喜欢汽车")
        self.h.fake.items = [fixtures.raw(s, "喜欢汽车")]
        original = self.h.fake.ask
        def ask(role, prompt, data):
            if role == "history_compare":
                raise TransientModelError("timeout")
            return original(role, prompt, data)
        with patch.object(self.h.fake, "ask", side_effect=ask):
            r = self.h.run_session(s)
        self.assertEqual(r["commit"]["counts"]["pending"], 1)
        self.assertEqual(len(self.h.state()["memories"]), 1)

    def test_scope_aliases_same_session_but_not_different_scenes(self):
        s = fixtures.session(text="我最喜欢和你一起呆着聊天")
        a = self.h.normalized(s, scope="当前会话可见对话段", fact=fact("喜欢", "聊天", "聊天"))
        b = self.h.normalized(s, scope="本次会话窗口内", fact=fact("喜欢", "聊天", "聊天"))
        self.assertTrue(same_fact(a, b))
        self.assertFalse(same_fact(a, dict(b, scope="和爸爸一起")))
        self.assertFalse(same_fact(dict(a, scope="当前会话"), dict(b, scope="当前会话", origin_session_id="s2")))

    def test_exact_fact_model_new_or_coexist_is_reinforced(self):
        s = fixtures.session(text="我喜欢恐龙")
        c = self.h.normalized(s, fact=fact())
        m = dict(c, memory_id="m", revision=2, state="reviewed")
        for action in ("new", "coexist"):
            d = relation_guard(c, dict(action=action, targets=[], basis=[], reason="模型判断"), [m])
            self.assertEqual(d["action"], "reinforce")
            self.assertEqual(d["targets"], [dict(memory_id="m", revision=2)])

    def test_same_evidence_mode_mismatch_cannot_silently_duplicate(self):
        s = fixtures.session(text="我想玩过家家")
        a = self.h.normalized(s, fact=fact("想玩", "过家家", "过家家"))
        b = self.h.normalized(s, assertion_mode="request", fact=fact("想玩", "过家家", "过家家", "request"))
        m = dict(a, memory_id="m", revision=1)
        self.assertTrue(comparable(b, m))
        for action in ("new", "coexist", "reinforce", "supersede", "conflict"):
            d = dict(action=action, targets=[] if action == "new" else [dict(memory_id="m", revision=1)], reason="测试", basis=[])
            self.assertEqual(relation_guard(b, d, [m])["action"], "pending")

    def learning(self):
        s = fixtures.session(text="庆")
        s["turns"][0]["assistant_text"] = "不是庆，再想想。"
        turn = dict(s["turns"][0], evidence_id="s1:2", user_text="府", assistant_text="答对了。")
        s["turns"].append(turn)
        s["input_hash"] = digest(s["turns"])
        steps = [dict(kind="attempt", text="庆", evidence_id="s1:1", speaker="user"),
                 dict(kind="hint", text="不是庆，再想想。", evidence_id="s1:1", speaker="assistant"),
                 dict(kind="attempt", text="府", evidence_id="s1:2", speaker="user"),
                 dict(kind="feedback", text="答对了。", evidence_id="s1:2", speaker="assistant")]
        raw = fixtures.raw(s, "猜字谜先答庆，提示后改答府，助手反馈正确", dimension_id="E44",
                           fact=fact("猜字谜", "字谜", "两次作答", "event"),
                           evidence=[dict(evidence_id=t["evidence_id"], quote=t["user_text"]) for t in s["turns"]],
                           structure=structure("learning_episode", "game", "猜字谜", steps))
        return s, raw

    def test_learning_structure_atomic_store_and_both_html(self):
        s, raw = self.learning()
        skill = self.h.registry.dimensions[raw["dimension_id"]]["owner_skill_id"]
        c = validate_candidate(raw, skill, self.h.registry, s, "w")
        p = SourcePipeline(self.h.agent, s)
        proof = p.proof(c, fixtures.verdict(c), p.pack([c]), [])
        receipt, _ = p.maintain([c], [proof], [], {})
        self.assertEqual(receipt["counts"]["new"], 1)
        memory = next(iter(self.h.state()["memories"].values()))
        self.assertEqual(memory["structure"]["steps"], c["structure"]["steps"])
        self.assertEqual(memory["retention"], "event")
        result = dict(session_id=s["session_id"], status="completed", turns=2, approved=[c], candidates=[c], reviews=[], coverage={})
        self.write_report([result])
        before = (self.h.out / "memory.sqlite3").read_bytes()
        export_user(self.h.out, self.h.out / "index.html")
        self.assertEqual(before, (self.h.out / "memory.sqlite3").read_bytes())
        for name in ("index.html", "index_report.html"):
            text = (self.h.out / name).read_text(encoding="utf-8")
            self.assertIn("learning_episode", text)
            self.assertIn("助手反馈正确", text)
            self.assertNotIn("__MEMORY_DETAILS_JS__", text)

    def test_structure_rejects_invented_quote_extra_key_and_assistant_as_child(self):
        s, raw = self.learning()
        skill = self.h.registry.dimensions["E44"]["owner_skill_id"]
        variants = []
        c = copy.deepcopy(raw); c["structure"]["steps"][0]["text"] = "我都会了"; variants.append(c)
        c = copy.deepcopy(raw); c["structure"]["steps"][0]["evil"] = True; variants.append(c)
        c = copy.deepcopy(raw); c["structure"]["steps"][1]["speaker"] = "user"; variants.append(c)
        c = copy.deepcopy(raw); c["structure"]["steps"][1]["kind"] = "attempt"; variants.append(c)
        for c in variants:
            with self.assertRaises(ValueError):
                validate_candidate(c, skill, self.h.registry, s, "w")

    def test_story_real_isolation(self):
        s = fixtures.session(text="小老鼠摔倒了")
        a = self.h.normalized(s, fact=fact("描述", "老鼠", "摔倒"), structure=structure(context="story"))
        b = dict(a, structure=structure(context="real"))
        self.assertFalse(comparable(a, b))
        self.assertFalse(same_fact(a, b))

    def write_report(self, results):
        (self.h.out / "report.json").write_text(json.dumps(dict(device_id="test-user", status="completed", source_scope="dialogue_only", selected_sessions=len(results), results=results, review_version="source-maintenance-v4"), ensure_ascii=False), encoding="utf-8")
        (self.h.out / "manifest.json").write_text(json.dumps(dict(review_version="source-maintenance-v4")), encoding="utf-8")

    def topic_sessions(self):
        results = []
        for sid, text in (("q1", "世界上最大的鸟是什么鸟？"), ("q2", "蜘蛛是昆虫吗？")):
            s = fixtures.session(sid, text)
            self.h.fake.items = [fixtures.raw(s, text, dimension_id="E44", keywords=["动物知识"], assertion_mode="request", fact=fact("询问", text, text, "request"), structure=structure("request"))]
            original = self.h.fake.ask
            def ask(role, prompt, data):
                if role == "source_router":
                    return dict(skills=[self.h.registry.dimensions["E44"]["owner_skill_id"]], signals=[], reason="测试主题提问")
                if role.startswith("source_extractor:"):
                    return dict(candidates=self.h.fake.items if any(d["dimension_id"] == "E44" for d in data["skill_rules"]) else [], notes="")
                return original(role, prompt, data)
            with patch.object(self.h.fake, "ask", side_effect=ask):
                results.append(self.h.run_session(s))
        return results

    def test_summary_threshold_refs_invalidation_and_late_data(self):
        results = self.topic_sessions()
        state = self.h.state()
        summaries = list(state["summaries"].values())
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["session_ids"], ["q1", "q2"])
        self.assertEqual(len(summaries[0]["source_refs"]), 2)
        before = copy.deepcopy(state["summaries"])
        rebuild_summaries(state, "2020-01-01T00:00:00+08:00")
        self.assertEqual(state["summaries"], before)
        m = next(iter(state["memories"].values()))
        for status in ("conflicted", "superseded", "pending_control"):
            state2 = copy.deepcopy(state); state2["memories"][m["memory_id"]]["state"] = status
            rebuild_summaries(state2, summaries[0]["window_end"])
            self.assertEqual(next(iter(state2["summaries"].values()))["status"], "inactive")
        self.write_report(results)
        formal = build_data(self.h.out, fixtures.BASE / self.h.cfg["rules_file"])
        report = build_report_user(self.h.out, formal)
        self.assertEqual(formal["summaries"], report["summaries"])
        export_user(self.h.out, self.h.out / "index.html")

    def test_summary_no_same_session_double_count_and_expiry(self):
        s = fixtures.session(text="蜘蛛是昆虫吗？")
        self.h.fake.items = [fixtures.raw(s, "询问蜘蛛是不是昆虫", assertion_mode="request", fact=fact("询问", "蜘蛛", "昆虫", "request"), structure=structure("request"))]
        self.h.run_session(s); self.h.run_session(s)
        self.assertFalse(self.h.state()["summaries"])
        self.topic_sessions()
        state = self.h.state()
        rebuild_summaries(state, "2030-01-01T00:00:00+08:00")
        self.assertTrue(all(s["status"] == "inactive" for s in state["summaries"].values()))
        rebuild_summaries(state, "2020-01-01T00:00:00+08:00")
        self.assertTrue(all(s["status"] == "inactive" for s in state["summaries"].values()))

    def test_shared_work_retains_child_contribution_not_personal_event(self):
        s = fixtures.session(text="再帮他做一个草莓蛋糕")
        raw = fixtures.raw(s, "共同故事中提出帮角色再做一个草莓蛋糕", dimension_id="E32",
                           fact=fact("共同创作", "蛋糕故事", "提出做蛋糕", "event"),
                           structure=structure("shared_work", "story", "蛋糕故事", [dict(kind="contribution", text=s["turns"][0]["user_text"], evidence_id=s["turns"][0]["evidence_id"], speaker="user")]))
        skill = self.h.registry.dimensions["E32"]["owner_skill_id"]
        c = validate_candidate(raw, skill, self.h.registry, s, "w")
        self.assertEqual(c["retention"], "event")
        broken = copy.deepcopy(raw); broken["structure"]["type"] = "real_event"
        with self.assertRaises(ValueError):
            validate_candidate(broken, skill, self.h.registry, s, "w")


if __name__ == "__main__":
    unittest.main()
