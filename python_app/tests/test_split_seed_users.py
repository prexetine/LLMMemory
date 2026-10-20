"""种子和候选目录隔离测试。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_user_batches import BatchIndex, export, dump, write_csv
from split_seed_users import split


class SplitTests(unittest.TestCase):
    def test_separate_and_preserve_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            users = [dict(device_id="A", is_seed=1, is_existing_candidate=0, is_core_50=0),
                     dict(device_id="B", is_seed=0, is_existing_candidate=1, is_core_50=1)]
            index = BatchIndex(root / "index.sqlite3")
            try:
                for device in ("A", "B"):
                    for batch in ("0907", "0929"):
                        index.add(batch, "dialogue", device, dict(data=dict(device_id=device, dialog_id="1", text="原话"),
                            disposition="eligible", device_ids=[device], source_relative_path=batch+"/fixture.xlsx", sheet="sheet1", row=2))
                counts, totals = export(index, source, users)
            finally:
                index.db.close()
            dump(source / "manifest.json", dict(users=users, totals=totals, legacy_root=str(root / "legacy"), policy={}))
            write_csv(source / "users.csv", users)
            write_csv(source / "user_source_counts.csv", counts)
            result = split(source, root / "seed")
            self.assertEqual(result["seed_only"]["totals"]["overlap"], 1)
            self.assertEqual(result["existing_candidates_only"]["totals"]["baseline_rows"], 1)
            self.assertFalse((source / "baseline_0907/candidate_data/A").exists())
            self.assertTrue((root / "seed/baseline_0907/candidate_data/A").exists())
            self.assertEqual(json.loads((source / "manifest.json").read_text(encoding="utf-8"))["users"][0]["is_core_50"], 1)
            with self.assertRaises(ValueError):
                split(source, root / "seed")

    def test_no_existing_target_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                split(Path(tmp) / "source", Path(tmp))


if __name__ == "__main__":
    unittest.main()
