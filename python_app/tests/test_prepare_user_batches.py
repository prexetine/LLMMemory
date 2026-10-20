"""分批导出回归测试，不调用模型或读取用户真实原文。"""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_user_batches import BatchIndex, export, identity, dump, validate, peak_memory, filtered_rows, normalize_existing, write_csv, sha


class BatchTests(unittest.TestCase):
    def test_identity(self):
        old = {"device_id": "A", "dialog_id": "1", "text": "你好"}
        self.assertEqual(identity("dialogue", old), identity("dialogue", dict(old, user_id="99")))
        self.assertNotEqual(identity("dialogue", old)[1], identity("dialogue", dict(old, text="再见"))[1])
        self.assertEqual(identity("dialogue", {"text": "你好"})[2], "content_hash")

    def test_export_numeric_precision(self):
        a = dict(device_id="A", dialog_id="1", text="你好", create_time="46206.649571759255")
        self.assertEqual(identity("dialogue", a), identity("dialogue", dict(a, create_time="46206.6495717593", user_id="new")))
        self.assertNotEqual(identity("dialogue", a), identity("dialogue", dict(a, text="再见")))
        a = dict(id="1", accuracy="19.867999999999999", lat="25.000000000000004")
        self.assertEqual(identity("location", a), identity("location", dict(a, accuracy="19.868", lat="25")))
        self.assertNotEqual(identity("location", a), identity("location", dict(a, lat="25.000001")))

    def test_reclassify_generated_v1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            out.mkdir()
            users = [dict(device_id="A", is_seed=1, is_existing_candidate=0, is_core_50=0)]
            index = BatchIndex(root / "index.db")
            try:
                # 故意以时间之外的差异造出v1修订，然后将源字段换回真实浮点尾差。
                for batch, text in (("0907", "原文"), ("0929", "变化")):
                    index.add(batch, "dialogue", "A", dict(data=dict(device_id="A", dialog_id="1", text=text,
                        create_time="46206.649571759255"), disposition="eligible", device_ids=["A"],
                        source_relative_path=batch + "/fixture.xlsx", sheet="sheet1", row=2))
                counts, totals = export(index, out, users)
            finally:
                index.db.close()
            for stage in ("baseline_0907", "revisions_0929"):
                path = out / stage / "candidate_data/A/dialogue.jsonl.gz"
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    p = json.loads(next(f))
                p["data"]["text"] = "原文"
                if stage == "revisions_0929":
                    p["data"]["create_time"] = "46206.6495717593"
                p["content_fingerprint"] = sha(p["data"])
                with gzip.open(path, "wt", encoding="utf-8") as f:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            write_csv(out / "user_source_counts.csv", counts)
            dump(out / "manifest.json", dict(schema_version="user-data-batches-v1", users=users,
                legacy_root=str(root / "legacy"), totals=totals, policy={}, seconds=0, process_peak_working_set_bytes=0))
            normalize_existing(out)
            m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(m["totals"]["revised"], 0)
            self.assertEqual(m["totals"]["overlap"], 1)
            self.assertEqual(validate(out)["status"], "passed")
            with gzip.open(out / "baseline_0907/candidate_data/A/dialogue.jsonl.gz", "rt", encoding="utf-8") as f:
                self.assertEqual(json.loads(next(f))["data"]["create_time"], "46206.649571759255")

    def test_partition_and_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = BatchIndex(root / "index.db")
            users = [dict(device_id=d, is_seed=1, is_existing_candidate=0, is_core_50=0) for d in ("A", "B")]
            def add(batch, key, text, device="A", **extra):
                data = dict(device_id=device, dialog_id=key, chat_id="chat-1", text=text, **extra)
                index.add(batch, "dialogue", device, dict(data=data, source_relative_path=batch + "/fixture.xlsx",
                    disposition="eligible", device_ids=[device], sheet="sheet1", row=2))
            try:
                add("0907", "1", "旧记录")
                add("0907", "1", "旧记录")
                add("0907", "2", "修订前")
                index.remember("dialogue", "A", dict(device_id="A", dialog_id="3", chat_id="chat-1", text="旧包独有"), "legacy")
                add("0929", "1", "旧记录", user_id="new_column")
                add("0929", "2", "修订后")
                add("0929", "3", "旧包独有")
                add("0929", "4", "新增")
                add("0929", "4", "新增")
                add("0929", "5", "冲突甲")
                add("0929", "5", "冲突乙")
                add("0929", "1", "其他设备", device="B")
                out = root / "out"
                out.mkdir()
                counts, totals = export(index, out, users)
                self.assertEqual(totals["baseline_rows"], 3)
                self.assertEqual(totals["baseline_eligible"], 2)
                self.assertEqual(totals["new_rows"], 2)
                self.assertEqual(totals["overlap"], 2)
                self.assertEqual(totals["batch_duplicate"], 1)
                self.assertEqual(totals["revised"], 1)
                self.assertEqual(totals["within_batch_conflict"], 2)
                self.assertEqual(totals["extends_existing_session_rows"], 1)
                dump(out / "manifest.json", dict(users=users, legacy_root=str(root / "legacy"), totals=totals))
                self.assertEqual(validate(out)["status"], "passed")
                with gzip.open(out / "increment_0929/candidate_data/A/dialogue.jsonl.gz", "rt", encoding="utf-8") as stream:
                    p = json.loads(next(stream))
                self.assertEqual(p["session_relation"], "extends_existing_session")
                self.assertTrue(all(r["source_0929"] == "not_provided" for r in counts if r["source_kind"] == "photo"))
            finally:
                index.db.close()

    def test_peak(self):
        value = peak_memory()
        self.assertTrue(value is None or value > 0)

    def test_filtered_reader_all_sheets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.zip"
            xml = '''<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
            <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
            <row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>11</v></c></row>
            <row r="3"><c r="A3" t="s"><v>3</v></c><c r="B3"><v>12</v></c></row>
            </sheetData></worksheet>'''
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("one.xml", xml.encode("utf-8"))
                z.writestr("two.xml", xml.encode("utf-8"))
            with zipfile.ZipFile(path) as z:
                strings = ["device_id", "dialog_id", "A", "B"]
                book = SimpleNamespace(path=path, zip=z, string=lambda i: strings[i], sheets=lambda: [("one", "one.xml"), ("two", "two.xml")])
                rows = list(filtered_rows(book, "dialogue", {"A"}))
            self.assertEqual(len(rows), 4)
            self.assertEqual(sum(r[2] is not None for r in rows), 2)
            self.assertEqual(rows[0][2], {"device_id": "A", "dialog_id": "11"})


if __name__ == "__main__":
    unittest.main()
