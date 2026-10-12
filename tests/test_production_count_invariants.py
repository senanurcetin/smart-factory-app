"""Check that simulated OEE never claims more rejects than produced units."""

import random
import unittest

from demo_state import advance_production_counts


class ProductionCounterTests(unittest.TestCase):
    def test_zero_production_cannot_create_scrap(self):
        self.assertEqual(advance_production_counts(0, 0, 0, True), (0, 0))
        self.assertEqual(advance_production_counts(10, 4, 0, True), (10, 4))

    def test_scrap_only_if_units_were_completed(self):
        self.assertEqual(advance_production_counts(0, 0, 1, True), (1, 1))
        self.assertEqual(advance_production_counts(1, 1, 2, False), (3, 1))
        self.assertEqual(advance_production_counts(3, 1, 2, True), (5, 2))

    def test_invariants_across_repeatable_random_batches(self):
        rng = random.Random(42)
        actual = scrap = 0
        for _ in range(10_000):
            batch = rng.randrange(4)
            bad_unit = rng.random() < 0.02
            actual, scrap = advance_production_counts(actual, scrap, batch, bad_unit)
            self.assertGreaterEqual(scrap, 0)
            self.assertLessEqual(scrap, actual)

    def test_invalid_counter_state_is_rejected(self):
        for values in ((0, 0, -1, True), (0, 1, 1, True), (1, -1, 1, False), (1, 0, True, True)):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    advance_production_counts(*values)


if __name__ == "__main__":
    unittest.main()
