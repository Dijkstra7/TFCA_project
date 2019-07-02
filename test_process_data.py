from unittest import TestCase
from main import load_data, process_data


class TestProcess_data(TestCase):
    def setUp(self) -> None:
        self.data = load_data()
        self.processed_data = process_data(self.data)


class Test_Size(TestProcess_data):
    def test_data_not_empty(self):
        self.assertGreater(len(self.processed_data), 0)

    def test_correct_size(self):
        self.assertEqual(len(self.processed_data), 3 * len(
            self.data.student_id.unique()))

    def test_not_all_answers_correct(self):
        self.assertNotEqual(len(self.processed_data),
                            self.processed_data.percentage_correct.sum())
