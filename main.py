import numpy as np
import pandas as pd


class Node:

    def __init__(self, kc_id, poss_values=None):
        self.poss_values = ["T", "F"]
        if poss_values is not None:
            self.poss_values = poss_values
        self.parents = []
        self.children = []
        self.cpt = {}
        self.kc_id = kc_id


def create_network_structure(kcs: list, hypothesis_node_id: int) -> dict:
    """
    Create a network structure based on the available knowledge components
    Parameters
    ----------
    kcs: list of str
        list of knowledge components
    hypothesis_node_id: int
        id of which of the knowledge components is the hypothesis node

    Returns
    -------
    dict of Node
        The description of the bayesian network
    """
    # Create the network starting with the hypothesis node
    hyp_name = f"master_{kcs[hypothesis_node_id]}"
    network = {hyp_name: Node(kcs[hypothesis_node_id])}
    for kc_id, kc in enumerate(kcs):
        if kc_id != hypothesis_node_id:  # Exclude the hypothesis node
            # Create nodes
            network[f"master_{kc}"] = Node(kc)  # master_kc node
            network[f"correct_{kc}"] = Node(kc)  # correct_percentage node

            # Create connections
            network[f"master_{kc}"].children.append(network[f"correct_{kc}"])
            network[f"correct_{kc}"].parents.append(network[f"master_{kc}"])

            network[f"master_{kc}"].children.append(network[hyp_name])
            network[hyp_name].parents.append(network[f"master_{kc}"])

    return network


def create_probabilities_network(my_network, data):
    """
    Create the conditional probability tables for the network.

    Parameters
    ----------
    my_network: dict of Nodes
        The bayesian network
    data: pandas.DataFrame
        The data to calculate the probabilities

    Returns
    -------
    None
    """
    for node_key in my_network.keys():
        if "master" in node_key:
            node = my_network[node_key]
            if len(node.children) > 0:
                select = data.loc[data.kc == node.kc_id]
                total = len(select)
                part = len(select.loc[select.mastering_grade >= .5])
                node.cpt = {'T': round(part / total, 4),
                            'F': 1-round(part / total, 4)}

    for node_key in my_network.keys():
        if "correct" in node_key:
            node = my_network[node_key]
            select = data.loc[data.kc == node.kc_id]
            true = len(select.loc[select.mastering_grade >= .5])
            true_high = len(select.loc[
                                (select.mastering_grade >= .5) &
                                (select.percentage_correct >= .7)
                            ]) / true
            false = len(select.loc[select.mastering_grade < .5])
            false_high = len(select.loc[
                                (select.mastering_grade < .5) &
                                (select.percentage_correct < .7)
                            ]) / false
            node.cpt = {f'T{node.kc_id}, High': true_high,
                        f'T{node.kc_id}, Low': 1 - true_high,
                        f'F{node.kc_id}, High': false_high,
                        f'F{node.kc_id}, Low': 1 - false_high}

    for node_key in my_network.keys():
        if "master" in node_key:
            node = my_network[node_key]
            if len(node.children) == 0:
                # When having more than two parents, this should be 
                # optimized using a recurrent calculation
                parent1 = node.parents[0]
                parent2 = node.parents[1]
                p1t_p2t = data.loc[((data.kc == parent1.kc_id) &
                                    (data.mastering_grade >= .5)) |
                                   ((data.kc == parent2.kc_id) &
                                    (data.mastering_grade >= .5))
                                   ]
                p1t_p2t_high = len(p1t_p2t.loc[
                                       p1t_p2t.percentage_correct >= .7]) / \
                               len(p1t_p2t) 
                node.cpt[f'T{parent1.kc_id}, T{parent2.kc_id}, High'] = \
                    p1t_p2t_high
                
                p1f_p2t = data.loc[((data.kc == parent1.kc_id) &
                                    (data.mastering_grade < .5)) |
                                   ((data.kc == parent2.kc_id) &
                                    (data.mastering_grade >= .5))
                                   ]
                p1f_p2t_high = len(p1f_p2t.loc[
                                       p1f_p2t.percentage_correct >= .7]) / \
                               len(p1f_p2t) 
                node.cpt[f'F{parent1.kc_id}, T{parent2.kc_id}, High'] = \
                    p1f_p2t_high

                p1t_p2f = data.loc[((data.kc == parent1.kc_id) &
                                    (data.mastering_grade >= .5)) |
                                   ((data.kc == parent2.kc_id) &
                                    (data.mastering_grade < .5))
                                   ]
                p1t_p2f_high = len(p1t_p2f.loc[
                                       p1t_p2f.percentage_correct >= .7]) / \
                               len(p1t_p2f)
                node.cpt[f'T{parent1.kc_id}, F{parent2.kc_id}, High'] = \
                    p1t_p2f_high

                p1f_p2f = data.loc[((data.kc == parent1.kc_id) &
                                    (data.mastering_grade < .5)) |
                                   ((data.kc == parent2.kc_id) &
                                    (data.mastering_grade < .5))
                                   ]
                p1f_p2f_high = len(p1f_p2f.loc[
                                       p1f_p2f.percentage_correct >= .7]) / \
                               len(p1f_p2f)
                node.cpt[f'F{parent1.kc_id}, F{parent2.kc_id}, High'] = \
                    p1f_p2f_high

                node.cpt[f'T{parent1.kc_id}, T{parent2.kc_id}, Low'] = \
                    1 - node.cpt[f'T{parent1.kc_id}, T{parent2.kc_id}, High']
                node.cpt[f'T{parent1.kc_id}, F{parent2.kc_id}, Low'] = \
                    1 - node.cpt[f'T{parent1.kc_id}, F{parent2.kc_id}, High']
                node.cpt[f'F{parent1.kc_id}, T{parent2.kc_id}, Low'] = \
                    1 - node.cpt[f'F{parent1.kc_id}, T{parent2.kc_id}, High']
                node.cpt[f'F{parent1.kc_id}, F{parent2.kc_id}, Low'] = \
                    1 - node.cpt[f'F{parent1.kc_id}, F{parent2.kc_id}, High']






def load_data(f_name="./res/data.csv"):
    """
    Load and pre-process the data.

    Parameters
    ----------
    f_name: str
        Path to the data

    Returns
    -------
    Pandas.DataFrame
        a DataFrame containing the data
    """
    raw_data = pd.read_csv(f_name)
    data = raw_data[["EloRatingcontext", "ExerciseResponseisCorrect",
                     "EloRatingstudentAbility", "ExerciseResponsestudent"]]
    data.columns = ["kc", "correct", "mastering", "student_id"]
    return data


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process the data retrieved from the datafile by calculating stats per
    student.

    Parameters
    ----------
    data: Pandas.DataFrame
        DataFrame containing the raw data

    Returns
    -------
    Pandas.DataFrame
        a DataFrame containing the processed data per student.
    """
    processed_data = {"student_id": [], "kc": [], "mastering_grade": [],
                      "percentage_correct": []}
    for student in data.student_id.unique():
        for kc in [8232, 8234, 8240]:
            select = data.loc[(data.student_id == student) &
                              (data.kc == kc)]
            # Get last ability score to find out the current mastering grade
            mastering = select.mastering.iloc[-1]

            # Calculate percentage of correct answers
            print(select.correct.values)
            print("======")
            percentage_correct = \
                len(select.loc[(select['correct'] == True)]) / len(select)

            # Add new row of data
            processed_data["student_id"].append(student)
            processed_data["kc"].append(kc)
            processed_data["mastering_grade"].append(mastering)
            processed_data["percentage_correct"].append(percentage_correct)

    return pd.DataFrame(processed_data)


if __name__ == "__main__":
    data = load_data()
    my_network = create_network_structure([8232, 8234, 8240], 0)
    processed_data = process_data(data)
    tables = create_probabilities_network(my_network, processed_data)

