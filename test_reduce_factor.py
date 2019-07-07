from unittest import TestCase
from main import reduce_factor


class TestReduce_factor(TestCase):
    def setUp(self) -> None:
        self.factor1 = ['F1', [[['A'], 0.25], [['~A'], .75]]]
        self.factor2 = ['F2', [[['A', 'B'], 0.35], [['A', '~B'], .75]]]
        self.factor3 = ['F3', [[['~A', 'C'], 0.25], [['~A', '~C'], .75]]]


class Test_amounts(TestReduce_factor):

    def test_more_than_two_factors(self):
        reduce_factor([self.factor1, self.factor2, self.factor3], 'A')

    def test_one_factor(self):
        reduce_factor([self.factor2], 'A')

