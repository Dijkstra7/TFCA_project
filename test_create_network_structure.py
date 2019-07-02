from unittest import TestCase

from main import create_network_structure


class TestCreate_network_structure(TestCase):
    def setUp(self) -> None:
        self.kc_ids = [1, 2, 3]
        self.network = create_network_structure(self.kc_ids, 0)


class TestInit(TestCreate_network_structure):
    def test_size(self):
        self.assertEqual(len(self.network), 5)

    def test_no_children_hyp_node(self):
        self.assertEqual(len(self.network["master_1"].children), 0)

    def test_no_parents_other_master_nodes(self):
        total_parents = 0
        for kc in self.kc_ids[1:]:  # Skip the hypothesis kc
            total_parents += len(self.network[f"master_{kc}"].parents)
        self.assertEqual(total_parents, 0)

    def test_master_matching_corrects(self):
        for kc in self.kc_ids[1:]:  # Skip the hypothesis kc
            self.assertEqual(self.network[f"correct_{kc}"].parents[0],
                             self.network[f"master_{kc}"])
            self.assertEqual(self.network[f"master_{kc}"].children[0],
                             self.network[f"correct_{kc}"])
