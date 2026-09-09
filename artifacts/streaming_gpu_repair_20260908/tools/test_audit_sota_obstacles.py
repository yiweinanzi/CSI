"""Synthetic parser tests only; these records are not experimental evidence."""

import unittest

from audit_sota_obstacles import ARMS, BUDGETS, CITIES, meter_grid


def rows():
    return [
        {"city_id": city, "budget": str(budget), "arm": arm, "mean_bank_median_error_m": str(index + 1)}
        for city in CITIES for budget in BUDGETS for index, arm in enumerate(ARMS)
    ]


class GridAuditTests(unittest.TestCase):
    def test_complete_grid_and_direction(self):
        grid = meter_grid(rows())
        self.assertEqual(len(grid), 8)
        self.assertTrue(all(cell["full_minus_response_m"] == 1 for cell in grid))
        self.assertFalse(any(cell["full_strictly_first_among_four_arms"] for cell in grid))

    def test_duplicate_is_rejected(self):
        data = rows()
        with self.assertRaises(ValueError):
            meter_grid(data + data[:1])

    def test_missing_cell_is_rejected(self):
        with self.assertRaises(ValueError):
            meter_grid(rows()[:-1])

    def test_nonfinite_and_nonpositive_are_rejected(self):
        for invalid in ("nan", "inf", "-inf", "0", "-1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                data = rows()
                data[0]["mean_bank_median_error_m"] = invalid
                meter_grid(data)

    def test_tie_is_not_superiority(self):
        data = rows()
        for row in data:
            row["mean_bank_median_error_m"] = "1"
        self.assertFalse(any(cell["full_strictly_first_among_four_arms"] for cell in meter_grid(data)))

    def test_unexpected_budget_is_rejected(self):
        data = rows()
        data[0]["budget"] = "256"
        with self.assertRaises(ValueError):
            meter_grid(data)


if __name__ == "__main__":
    unittest.main()
