import unittest

from util.schema import parse_q1_region_ids, parse_q2_position_aligned


class Q1SchemaTests(unittest.TestCase):
    def test_parses_ordered_triplet(self):
        self.assertEqual(parse_q1_region_ids("[11, 14, 15]"), [11, 14, 15])

    def test_rejects_non_json_and_extra_ids(self):
        for response in ("regions: 11, 14, 15", "```json\n[11, 14, 15]\n```", "[11, 14, 15, 16]"):
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_q1_region_ids(response)

    def test_rejects_duplicate_or_out_of_range_ids(self):
        for response in ("[1, 1, 2]", "[0, 2, 3]", "[14, 15]"):
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_q1_region_ids(response)


class Q2SchemaTests(unittest.TestCase):
    def test_parses_indexed_tuples_and_binds_q1_ids_by_position(self):
        response = (
            '(T1=speech, F1=mid, P1=vowel, En1="mid-frequency vowel artifact"); '
            '(T2=non-speech, F2=high, P2=none, En2="high-frequency pause artifact"); '
            '(T3=speech, F3=low, P3=consonant, En3="low-frequency consonant artifact")'
        )
        regions = parse_q2_position_aligned(response, [11, 14, 15])
        self.assertEqual([region["region_id"] for region in regions], [11, 14, 15])
        self.assertEqual([region["rank"] for region in regions], [1, 2, 3])
        self.assertEqual(regions[1]["P"], "none")

    def test_rejects_legacy_cn_schema(self):
        response = (
            '(Cn=11, T=speech, F=mid, P=vowel, En="artifact"); '
            '(Cn=14, T=speech, F=low, P=vowel, En="artifact"); '
            '(Cn=15, T=speech, F=low, P=vowel, En="artifact")'
        )
        with self.assertRaises(ValueError):
            parse_q2_position_aligned(response, [11, 14, 15])

    def test_rejects_legacy_unvoiced_label(self):
        response = (
            '(T1=speech, F1=mid, P1=unvoiced, En1="artifact"); '
            '(T2=speech, F2=low, P2=vowel, En2="artifact"); '
            '(T3=speech, F3=low, P3=vowel, En3="artifact")'
        )
        with self.assertRaises(ValueError):
            parse_q2_position_aligned(response, [11, 14, 15])


if __name__ == "__main__":
    unittest.main()
