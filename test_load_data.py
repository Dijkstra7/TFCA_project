from unittest import TestCase
from main import load_data

class TestLoad_data(TestCase):
    def setUp(self) -> None:
        self.data = load_data()


class Test_Loading(TestLoad_data):
    def test_data_not_empty(self):
        self.assertGreater(len(self.data), 0)

    def test_info_is_four_columns(self):
        self.assertEqual(len(self.data.columns), 4)
