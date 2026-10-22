"""Dependency-free checks; also collected by pytest in the normal suite."""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_demo_data import ROOT, build_data, serialize


class DemoExportTests(unittest.TestCase):
    def setUp(self):
        self.data = build_data()

    def test_export_is_deterministic_and_current(self):
        self.assertEqual(serialize(self.data), serialize(build_data()))
        self.assertEqual((ROOT / "demo/data/epa.json").read_text(encoding="utf-8"), serialize(self.data))
        for path, digest in self.data["input_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)

    def test_counts_spans_and_attachments(self):
        d = self.data
        self.assertEqual(len(d["annotation"]["steps"]), 14)
        self.assertEqual(len(d["constraints"]), 15)
        steps = {s["id"] for s in d["annotation"]["steps"]}
        for item in d["annotation"]["steps"] + d["constraints"]:
            p = item["provenance"]
            a, b = p["char_start"] - d["source"]["char_start"], p["char_end"] - d["source"]["char_start"]
            self.assertTrue(0 <= a < b <= len(d["source"]["text"]))
            self.assertTrue(d["source"]["text"][a:b].strip())
            if "text" in item:
                self.assertEqual(d["source"]["text"][a:b], item["text"])
                self.assertTrue(set(item.get("attached_to", item.get("applies_to", []))) <= steps)
        self.assertEqual(d["summary"]["omitted_constraint_ids"], [c["id"] for c in d["constraints"]])

    def test_status_and_complete_packet_are_preserved(self):
        d = self.data
        self.assertFalse(d["human_verified"])
        self.assertEqual(d["status"]["status"], "candidate")
        self.assertEqual(d["review_packet"]["status"], "prepared_not_human_evidence")
        self.assertEqual(len(d["review_packet"]["human_decisions_required"]), 8)
        self.assertEqual(len(d["corpus"]), 8)
        self.assertEqual(sum(not c["include_for_evaluation"] for c in d["corpus"]), 3)
        self.assertFalse(d["replay"]["available"])
        packet = json.loads((ROOT / f"datasets/paper/review_packets/{d['status']['doc_id']}.json").read_text())
        self.assertEqual(d["review_packet"], packet)

    def test_rejects_invalid_spans_attachments_relations_and_status(self):
        from scripts.export_demo_data import read_json
        mutations = [
            ("review_candidates/", lambda a: a["steps"][0]["provenance"].update(char_start=0)),
            ("review_candidates/", lambda a: a["steps"][0]["constraints"][0].update(text="invented")),
            ("review_candidates/", lambda a: a["steps"][0]["constraints"][0].update(attached_to=["S999"])),
            ("review_candidates/", lambda a: a["relations"][0].update(target="S999")),
            ("corpus_manifest", lambda a: a["documents"][0].update(status="human_verified")),
        ]
        for path_part, mutate in mutations:
            def changed(root, path):
                obj = copy.deepcopy(read_json(root, path))
                if path_part in path:
                    mutate(obj)
                return obj
            with self.subTest(path=path_part), patch("scripts.export_demo_data.read_json", side_effect=changed):
                with self.assertRaises(ValueError):
                    build_data()

    def test_packet_hash_mismatch_fails_closed(self):
        from scripts.export_demo_data import read_json
        def changed(root, path):
            obj = read_json(root, path)
            if "review_packets/" in path:
                obj["source"]["sha256"] = "0" * 64
            return obj
        with patch("scripts.export_demo_data.read_json", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build_data()


if __name__ == "__main__":
    unittest.main()
