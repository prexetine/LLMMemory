"""图片提取、视觉审核、证据入库及失败续跑的离线测试。"""
import copy
import gzip
import io
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_source_agent_v3 import BASE, Fake, verdict
from agent_data import Registry, digest
from agent_photos import options, safe_url, load_photo_sessions, bind_image, image_content, normalize_image
from agent_store import MemoryStore
from agent_store_v3 import V3Store
from agent_v3_contracts import verify_source
from agent_llm import JsonClient
from legacy_memory_agent import CentralAgent, run_one
from export_memory_html import export_user


def config():
    return json.loads((BASE / "python_app/config/memory_agent_seed_top5_0907_photos.json").read_text(encoding="utf-8"))


def fixture(root):
    (root / "users.csv").write_text("device_id\nu1\n", encoding="utf-8")
    folder = root / "candidate_data/u1"
    folder.mkdir(parents=True)
    photo = dict(disposition="eligible", occurred_at="2026-09-01T00:00:00+08:00", source_file="fixture20260907.xlsx", sheet="photo", row=2,
                 data=dict(device_id="u1", picture_id="1", picture_url="https://hipanda-cdn-ws.toycloud.com/test.jpg", del_flag="0"))
    with gzip.open(folder / "photo.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps(photo) + "\n")
    with gzip.open(folder / "dialogue.jsonl.gz", "wt", encoding="utf-8") as f:
        pass
    return load_photo_sessions(root, "u1", config())[0][0]


def pixels():
    from PIL import Image
    output = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(output, format="PNG")
    return output.getvalue()


class PhotoFake(Fake):
    def ask(self, role, prompt, data):
        r = super().ask(role, prompt, data)
        if role.startswith("source_extractor:"):
            for c in r["candidates"]:
                c.update(kind="observation", subject="照片画面", subject_key=data["session"]["turns"][0]["evidence_id"],
                         claim="画面是红色", scope="仅此照片", attribute_key="画面颜色")
        return r


class PhotoTests(unittest.TestCase):
    def test_url_and_config_restrictions(self):
        s = options(config())
        for url in ("http://127.0.0.1/x", "https://hipanda-cdn-ws.toycloud.com.evil/x", "file:///a", "https://user@hipanda-cdn-ws.toycloud.com/x", "https://hipanda-cdn-ws.toycloud.com:8000/x"):
            with self.assertRaises(ValueError):
                safe_url(url, s)
        self.assertTrue(safe_url("http://hipanda-cdn-ws.toycloud.com/x", s).startswith("https://"))
        c = config(); c["provider"] = "Deepseek"
        with self.assertRaises(ValueError):
            options(c)

    def test_pixels_limits(self):
        s = options(config())
        self.assertTrue(normalize_image(pixels(), s).startswith(b"\xff\xd8"))
        with self.assertRaises(ValueError):
            normalize_image(pixels(), dict(s, max_bytes=1))
        with self.assertRaises(ValueError):
            normalize_image(pixels(), dict(s, max_pixels=1))
        with self.assertRaises(Exception):
            normalize_image(b"<html>not image</html>", s)

    def test_binding_review_store_html_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s = fixture(root)
            cfg = config()
            fake = PhotoFake()
            opener = SimpleNamespace(open=lambda *a, **kw: io.BytesIO(pixels()))
            with patch("agent_photos.urllib.request.build_opener", return_value=opener):
                bind_image(s, root, cfg, fake)
            images = image_content(fake, {"session": s})
            self.assertEqual(images[1]["type"], "image_url")
            with patch("agent_photos.urllib.request.build_opener", side_effect=AssertionError("禁止重复下载")):
                bind_image(s, root, cfg, fake)
            store = MemoryStore(root / "memory.sqlite3")
            try:
                registry = Registry(BASE / cfg["rules_file"])
                a = CentralAgent(fake, registry, store, cfg, root / "trace.jsonl")
                result = a.run_session(s)
                self.assertEqual(len(result["approved"]), 1)
                reviews = [data for role, data in fake.calls if role == "source_reviewer"]
                self.assertEqual(reviews[0]["evidence_pack"][0]["image_sha256"], s["turns"][0]["image_sha256"])
                self.assertEqual(len(image_content(fake, reviews[0])), 2)
                before = len(fake.calls)
                a.run_session(s)
                self.assertEqual(len(fake.calls), before)
                evidence = json.loads(store.db.execute("SELECT data FROM evidence").fetchone()[0])
                self.assertEqual(evidence["source_kind"], "photo")
                self.assertNotIn("base64", json.dumps(evidence))
                proof = json.loads(store.db.execute("SELECT data FROM v3_source_reviews").fetchone()[0])
                altered = copy.deepcopy(s); altered["turns"][0]["image_sha256"] = "bad"
                with self.assertRaises(ValueError):
                    verify_source(proof, result["candidates"][0], altered)
                (root / "report.json").write_text(json.dumps(dict(device_id="u1", status="completed", results=[result], selected_sessions=1, source_scope="dialogue_and_photo")), encoding="utf-8")
                (root / "manifest.json").write_text(json.dumps(dict(review_version="source-maintenance-v3")), encoding="utf-8")
                export_user(root, rules=BASE / cfg["rules_file"])
                self.assertIn("dialogue_and_photo", (root / "用户记忆维度展示.html").read_text(encoding="utf-8"))
            finally:
                store.close()

    def test_actual_request_contains_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); s = fixture(root); cfg = config()
            # 假凭据只写测试目录；HTTP由内存响应替代。
            path = root / "credentials.yaml"
            path.write_text("Doubao:\n  base_url: https://ark.cn-beijing.volces.com/api/v3\n  api_key: fixture\n  model_map:\n    " + cfg["model_alias"] + ": fixture-model\n", encoding="utf-8")
            client = JsonClient(cfg, path, root / "calls.jsonl")
            with patch("agent_photos.urllib.request.build_opener", return_value=SimpleNamespace(open=lambda *a, **k: io.BytesIO(pixels()))):
                bind_image(s, root, cfg, client)
            captured = []
            def respond(request, **kwargs):
                captured.append(json.loads(request.data))
                return io.BytesIO(json.dumps(dict(choices=[dict(message=dict(content='{"skills":[],"signals":[],"reason":"empty"}'), finish_reason="stop")], usage=dict(total_tokens=123))).encode())
            client.opener = SimpleNamespace(open=respond)
            client.ask("source_router", "测试", {"session": s})
            content = captured[0]["messages"][1]["content"]
            self.assertTrue(any(x.get("type") == "image_url" for x in content))
            log = (root / "calls.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("base64", log)
            self.assertEqual(json.loads(log)["usage"]["total_tokens"], 123)

    def test_download_failure_is_retryable_and_html_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            cfg = config(); cfg.update(data_root=str(root), device_id="u1")
            out = root / "out"
            with patch("legacy_memory_agent.JsonClient", return_value=PhotoFake()), patch("legacy_memory_agent.bind_image", side_effect=RuntimeError("图片获取失败")):
                self.assertEqual(run_one(cfg, "live", str(out)), 2)
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["photo_completed"], 0)
            self.assertEqual(report["results"][0]["status"], "failed")
            export_user(out, rules=BASE / cfg["rules_file"])


if __name__ == "__main__":
    unittest.main()
