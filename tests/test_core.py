import json
import unittest
from pathlib import Path

from recon.data import load_months
from recon.gen import self_check
from recon.score import score_month
from recon.shortlist import build_shortlist, parse_answer, target_json
from verify_submission import exact_two_sided_sign_test


ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.month = load_months(ROOT / "data" / "phase0_hard.jsonl")[0]

    def test_committed_month_has_exact_gold(self):
        self.assertEqual(self_check(self.month), [])

    def test_gold_scores_perfectly(self):
        score = score_month(self.month, self.month.gold)
        self.assertEqual(score["tp"], score["n_gold"])
        self.assertEqual(score["line_ok"], score["n_lines"])

    def test_shortlist_target_round_trips(self):
        for bank in self.month.bank:
            shortlist = build_shortlist(self.month, bank)
            parsed = parse_answer(shortlist, target_json(shortlist))
            expected = sorted(
                ledger_id
                for ledger_id in shortlist.gold
                if ledger_id in {candidate.id for candidate in shortlist.candidates}
            )
            self.assertEqual(parsed, expected)

    def test_primary_result_artifacts_are_complete(self):
        whole_path = ROOT / "results" / "lines" / "fireworks_gpt_oss_120b_whole_month.jsonl"
        harness_path = ROOT / "results" / "lines" / "fireworks_gpt_oss_120b_harness.jsonl"
        with whole_path.open(encoding="utf-8") as handle:
            whole = [json.loads(line) for line in handle if line.strip()]
        with harness_path.open(encoding="utf-8") as handle:
            harness = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(whole), 60)
        self.assertEqual(len(harness), 10_000)
        self.assertEqual(len({row["month"] for row in whole}), 60)
        self.assertEqual(len({(row["month"], row["bank"]) for row in harness}), 10_000)

    def test_exact_sign_test(self):
        self.assertEqual(exact_two_sided_sign_test(0, 0), 1.0)
        self.assertEqual(exact_two_sided_sign_test(1, 1), 1.0)
        self.assertAlmostEqual(exact_two_sided_sign_test(56, 1), 8.049116928532385e-16)


if __name__ == "__main__":
    unittest.main()
