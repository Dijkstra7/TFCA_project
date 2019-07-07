import pandas as pd
from random import random, choice


eff = 120
mg = .3

class Node:

    current_letter = 0
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    current_factor = 1

    def __init__(self, kc_id, poss_values=None, type_ = "master"):
        """
        Initialize the node.

        Parameters
        ----------
        kc_id: str
            Identifier of the node
        poss_values:
        """
        self.poss_values = ["T", "F"]
        if poss_values is not None:
            self.poss_values = poss_values
        self.parents = []
        self.children = []
        self.cpt = []
        self.kc_id = kc_id
        self.id_ = f"{type_}_{kc_id}"
        self.value = None
        self.lid = Node.letters[Node.current_letter]
        Node.current_letter += 1

    def set_value(self, value):
        self.value = value
        # self.cpt = {self.cpt[value]}

    def create_factor(self):
        factors = self.cpt
        if self.value is not None:
            if self.value is True:
                factors = [fac for fac in factors if '~' not in fac[0][0]]
            elif self.value is False:
                factors = [fac for fac in factors if '~' in fac[0][0]]
        for p_id, parent in enumerate(self.parents):
            parent = self.parents[p_id]
            if parent.value is not None:
                if parent.value is True:
                    factors = [fac for fac in factors
                               if '~' not in fac[0][p_id + 1]]
                else:
                    factors = [fac for fac in factors
                               if '~' in fac[0][p_id + 1]]
        Node.current_factor += 1
        return [f'F{Node.current_factor -1}', factors]


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
    network = {hyp_name: Node(kcs[hypothesis_node_id], poss_values=["T", "F"])}
    for kc_id, kc in enumerate(kcs):
        if kc_id != hypothesis_node_id:  # Exclude the hypothesis node
            # Create nodes
            network[f"master_{kc}"] = Node(kc, poss_values=["T", "F"])
            network[f"effort_{kc}"] = Node(kc, poss_values=["High", "Low"],
                                           type_="effort")

            # Create connections
            network[f"master_{kc}"].parents.append(network[f"effort_{kc}"])
            network[f"effort_{kc}"].children.append(network[f"master_{kc}"])

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
    # print(data.tail(len(data)))
    for node_key in my_network.keys():
        if "effort" in node_key:
            node = my_network[node_key]
            total = len(data)
            part = len(data.loc[data[f"effort_{node.kc_id}"] >= eff])
            probability = round(part / total, 4)
            node.cpt = [[[node.lid], probability],
                        [[f'~{node.lid}'], 1-probability]]

    for node_key in my_network.keys():
        if "master" in node_key:
            node = my_network[node_key]
            if len(node.children) != 0:  # Skip the hypothesis node
                high = len(data.loc[data[f"effort_{node.kc_id}"] >= eff])
                high_true = len(data.loc[
                                    (data[f"mastering_{node.kc_id}"] >= mg) &
                                    (data[f"effort_{node.kc_id}"] >= eff)
                                    ]) / high
                low = len(data.loc[data[f"effort_{node.kc_id}"] < eff])
                low_true = len(data.loc[
                                     (data[f"mastering_{node.kc_id}"] >= mg) &
                                     (data[f"effort_{node.kc_id}"] < eff)
                                     ]) / low
                node.cpt = [[[f'{node.lid}', f'{node.parents[0].lid}'],
                             high_true],
                            [[f'~{node.lid}', f'{node.parents[0].lid}'],
                             1 - high_true],
                            [[f'{node.lid}', f'~{node.parents[0].lid}'],
                              low_true],
                            [[f'~{node.lid}', f'~{node.parents[0].lid}'],
                              1 - low_true]]

    for node_key in my_network.keys():
        if "master" in node_key:
            node = my_network[node_key]
            if len(node.children) == 0:  # Only hypothesis node
                # When having more than two parents, this should be 
                # optimized using a recurrent calculation
                parent1 = node.parents[0]
                parent2 = node.parents[1]
                set_ = data.loc[(data[f"mastering_{parent1.kc_id}"] >= mg) &
                                (data[f"mastering_{parent2.kc_id}"] >= mg)
                                ]
                p1t_p2t_high = len(set_.loc[
                                       set_[f"mastering_{node.kc_id}"] >= mg
                                   ]) / (len(set_) + 1e-12)
                node.cpt.append([[f'{node.lid}', f'{parent1.lid}',
                                   f'{parent2.lid}'], p1t_p2t_high])
                
                set_ = data.loc[(data[f"mastering_{parent1.kc_id}"] < mg) &
                                (data[f"mastering_{parent2.kc_id}"] >= mg)
                                ]
                p1f_p2t_high = len(set_.loc[
                                       set_[f"mastering_{node.kc_id}"] >= mg
                                   ]) / (len(set_) + 1e-12)
                node.cpt.append([[f'{node.lid}', f'~{parent1.lid}',
                                  f'{parent2.lid}'], p1f_p2t_high])

                set_ = data.loc[(data[f"mastering_{parent1.kc_id}"] >= mg) &
                                (data[f"mastering_{parent2.kc_id}"] < mg)
                                ]
                p1t_p2f_high = len(set_.loc[
                                       set_[f"mastering_{node.kc_id}"] >= mg
                                   ]) / (len(set_) + 1e-12)
                node.cpt.append([[f'{node.lid}', f'{parent1.lid}',
                                  f'~{parent2.lid}'], p1t_p2f_high])

                set_ = data.loc[(data[f"mastering_{parent1.kc_id}"] < mg) &
                                (data[f"mastering_{parent2.kc_id}"] < mg)
                                ]
                p1f_p2f_high = len(set_.loc[
                                       set_[f"mastering_{node.kc_id}"] >= mg
                                   ]) / (len(set_) + 1e-12)
                node.cpt.append([[f'{node.lid}', f'~{parent1.lid}',
                                  f'~{parent2.lid}'], p1f_p2f_high])

                node.cpt.append([[f'~{node.lid}', f'{parent1.lid}',
                                 f'{parent2.lid}'], 1 - node.cpt[0][1]])
                node.cpt.append([[f'~{node.lid}', f'{parent1.lid}',
                                 f'~{parent2.lid}'], 1 - node.cpt[1][1]])
                node.cpt.append([[f'~{node.lid}', f'~{parent1.lid}',
                                 f'{parent2.lid}'], 1 - node.cpt[2][1]])
                node.cpt.append([[f'~{node.lid}', f'~{parent1.lid}',
                                 f'~{parent2.lid}'], 1 - node.cpt[3][1]])


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
    data.columns = ["kc", "effort", "mastering", "student_id"]
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
    processed_data = {"student_id": []}
    for kc in [8232, 8234, 8240]:
        processed_data[f"effort_{kc}"] = []
        processed_data[f"mastering_{kc}"] = []
        for student in data.student_id.unique():
            select = data.loc[(data.student_id == student) &
                              (data.kc == kc)]
            # Get last ability score to find out the current mastering grade
            mastering = select.mastering.iloc[-1]

            # Calculate percentage of effort answers
            effort = len(select)

            # Add new row of data
            if kc == 8232:  # Only once per all kc's
                processed_data["student_id"].append(student)
            processed_data[f"mastering_{kc}"].append(mastering)
            processed_data[f"effort_{kc}"].append(effort)

    return pd.DataFrame(processed_data)


def reduce_factor(factors, letter):
    """
    Reduce the factors given to one factor marginalized over a given letter.
    Parameters
    ----------
    factors: list of lists
        The factors to be reduced
    letter: str
        letter to be marginalized

    Returns
    -------
    The reduced factor
    """
    if len(factors) > 2:
        raise NotImplementedError

    if len(factors) == 2:
        # Multiply factors with same letters
        new_options = []
        fac1 = factors[0]
        for option in fac1[1]:
            for other_option in factors[1][1]:
                option_letters = [let for let in option[0] if letter not in let]
                select_letter = [let for let in option[0] if letter in let][0]
                if select_letter in other_option[0]:
                    for other_letter in other_option[0]:
                        if other_letter not in option_letters and \
                                letter not in other_letter:
                            option_letters.append(other_letter)
                new_options.append([option_letters,
                                    option[1] * other_option[1]])
        mult_factor = [f'F{Node.current_factor}', new_options]
        # print('multi_factor = ', mult_factor)
        # Marginalize over letters
        new_factor = [mult_factor[0], []]
        new_letters = []
        for new_option in new_options:
            option_letters = new_option[0]
            if option_letters == []:
                continue
            if option_letters in new_letters:
                new_factor[1][new_letters.index(option_letters)][1] += \
                    new_option[1]
            else:
                new_factor[1].append(new_option)
                new_letters.append(option_letters)
    elif len(factors) == 1:
        print(factors)
        new_factor = [[ltr for ltr in factors[0][0] if letter not in ltr] ,
                      factors[0][1]]

    else:
        new_factor = factors
    Node.current_factor += 1
    # print("New factor:", new_factor)
    return(new_factor)


def find_most_frugal(my_network, hyp_node_key, N):
    """
    Find the most frugal explanation by sampling random nodes to be relevant.

    Return the nodes that are seen to be relevant the most in the most
    frugal explanation.

    Parameters
    ----------
    my_network: Dict of Nodes
        The Bayesian network
    hyp_node_key: str
        The id of the hypothesis node
    N: int
        The amount of samples we use.

    Returns
    -------
    list of Nodes:
        The intermediate nodes that are relevant
    """
    # Define type of nodes
    intermediate_node_keys = [key for key in my_network.keys()
                              if "master" in key and key != hyp_node_key]
    evidence_node_keys = [key for key in my_network.keys() if "effort" in key]

    selected_relevant = {}
    for n in range(N):
        # set initial values
        Node.current_factor = 0
        for node_key in my_network.keys():
            my_network[node_key].value = None

        # Create random irrelevant or relevant nodes
        relevant_node_keys = []
        irrelevant_node_keys = []
        for node_key in intermediate_node_keys:
            if random() > .5:
                relevant_node_keys.append(node_key)
            else:
                irrelevant_node_keys.append(node_key)
                i_node = my_network[node_key]
                # Set random value to irrelevant node
                i_node.set_value(choice([True, False]))
                # print(f"set node {i_node.id_} ({i_node.lid}) to "
                #       f"{i_node.value}")

        # Randomly determine the evidence variables
        for node_key in evidence_node_keys:
            e_node = my_network[node_key]
            e_node.set_value(choice([True, False]))
            # print(f"set node {e_node.id_} ({e_node.lid}) to "
            #       f"{e_node.value}")

        # factorize network and do variable elimination
        factors = []
        for node_key in my_network.keys():
            factors.append(my_network[node_key].create_factor())
        # for factor in factors:
        #     print(factor)
        for reduce_letter in "ECBD":
            red_factor = []
            for factor in factors:
                if reduce_letter in factor[1][0][0] or '~'+reduce_letter in \
                        factor[1][0][0]:
                    red_factor.append(factor)
            # print(f"Reducing factors: {[f[0] for f in red_factor]} with "
            #       f"letter {reduce_letter}")
            reduced = reduce_factor(red_factor, reduce_letter)
            factors.append(reduced)
            # Assign value if only one variable is left in the factor
            if len(reduced[1][0][0]) == 1 and len(reduced[1]) == 2:
                # Find the node with the correct letter
                for node_key in my_network.keys():
                    set_node = my_network[node_key]
                    if set_node.lid == reduced[1][0][0][0]:
                        # Assign value of highest probability
                        set_node.set_value(reduced[1][0][1] >
                                           reduced[1][1][1])
                        # print(f"set node {set_node.id_} ({set_node.lid}) to "
                        #       f"{set_node.value}")



            for factor in red_factor:
                factors.remove(factor)
            # print(f"factors left: {[f[0] for f in factors]}")

        if len(factors) > 1:  # Should have only one factor left.
            raise ValueError

        # Determine Hmax
        last_factor = factors[0]
        h_values = [option[1] for option in last_factor[1]]
        # print("h_values = ", h_values)
        h_max = max(h_values)

        # Collate truth assignment
        jva = ""
        for letter in "ABCDE":
            for node_key in my_network.keys():
                if my_network[node_key].lid == letter:
                    if my_network[node_key].value is True:
                        jva += 'T'
                    else:
                        jva += 'F'

        if jva not in selected_relevant:
            selected_relevant[jva] = [relevant_node_keys]
            selected_relevant[f'{jva}_max'] = h_max
        else:
            selected_relevant[jva].append(relevant_node_keys)
            selected_relevant[f'{jva}_max'] = \
                max(h_max, selected_relevant[f'{jva}_max'])

    most_selected_relevant = None
    h_max = 0
    for key in selected_relevant.keys():
        if 'max' in key:
            continue
        if most_selected_relevant is None or \
                len(most_selected_relevant) < len(selected_relevant[key]):
            most_selected_relevant = selected_relevant[key]
            h_max = selected_relevant[f'{key}_max']
        elif len(most_selected_relevant) == len(selected_relevant[key]):
            if h_max < selected_relevant[f'{key}_max']:
                h_max = selected_relevant[f'{key}_max']
                most_selected_relevant = selected_relevant[key]

    return most_selected_relevant


def find_relevancy_level_nodes(relevant_combinations):
    """
    Calculate the relevancy percentage of kc's being in the relevant
    combination
    Parameters
    ----------
    relevant_combinations: list of list of str
        The relevant nodes per mfe result.

    Returns
    -------
    dict
        The relevancy percentage per kc.
    """
    kc_totals = {}
    for combination in relevant_combinations:
        for kc in combination:
            if kc not in kc_totals:
                kc_totals[kc] = 1
            else:
                kc_totals[kc] += 1

    relevant_levels = []
    for key in kc_totals.keys():
        relevant_levels.append([key,
                                kc_totals[key]/len(relevant_combinations)])
    return relevant_levels

if __name__ == "__main__":
    # Load the data
    quick_load = False
    if quick_load is False:
        data = load_data()
        processed_data = process_data(data)
        processed_data.to_csv("./res/processed_data.csv")
    else:
        processed_data = pd.read_csv("./res/processed_data.csv")

    # Set up variables
    kcs = [8232, 8234, 8240]
    for hyp_id in [2, 1, 0]:
        Node.current_letter = 0
        N = 8000
        hyp_key = f"master_{kcs[hyp_id]}"

        # Set up the network
        my_network = create_network_structure(kcs, hyp_id)
        create_probabilities_network(my_network, processed_data)

        # Find relevant nodes
        most_relevant = find_most_frugal(my_network, hyp_key, N)
        relevance_levels = find_relevancy_level_nodes(most_relevant)
        print(f"relevance levels for intermediate nodes for {hyp_key} are:",
              relevance_levels)
