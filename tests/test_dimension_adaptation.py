import unittest

from dimension_adaptation import map_slot

class TestMapSlot(unittest.TestCase):
    def test_maps_four_slots_to_eight_slots(self):
        mapped_indices = []

        for source_index in range(4):
            target_index = map_slot(
                source_size=4,
                target_size=8,
                source_index=source_index,
            )

            mapped_indices.append(target_index)

        expected_indices = [0, 2, 4, 6]

        print()
        print(f"Actual mapping:   {mapped_indices}")
        print(f"Expected mapping: {expected_indices}")

        self.assertEqual(
            mapped_indices,
            expected_indices,
        )

    def test_expand_three_slots_to_five_slots(self):
        actual = [
            map_slot(
                source_size=3,
                target_size=5,
                source_index=i,
            )
            for i in range(3)
        ]
        expected = [0, 1, 3]

        print("3 -> 5")
        print("actual:  ", actual)
        print("expected:", expected)

        self.assertEqual(actual, expected)


    def test_rejects_negative_source_index(self):
        with self.assertRaises(IndexError):
            map_slot(
                source_size=4,
                target_size=8,
                source_index=-1,
            )

    def test_rejects_source_index_out_of_range(self):
        with self.assertRaises(IndexError):
            map_slot(
                source_size=4,
                target_size=8,
                source_index=4,
            )

    def test_rejects_zero_source_size(self):
        with self.assertRaises(ValueError):
            map_slot(
                source_size=0,
                target_size=8,
                source_index=0,
            )

    def test_rejects_smaller_target(self):
        with self.assertRaises(ValueError):
            map_slot(
                source_size=8,
                target_size=4,
                source_index=0,
            )

    if __name__ == "__main__":
        unittest.main()
