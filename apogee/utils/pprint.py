import numpy as np


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def pprint_marginals(network, marginals, n=4, percent=True):
    for variable, marginal in marginals.items():
        print("=" * 50)
        if len(network[variable].parents) > 0:
            print(variable, "|", ", ".join(network[variable].parents))
        else:
            print(variable)
        print("-" * 50)

        probs = np.asarray(list(marginal.values()))

        if percent:
            statement = lambda x, y: ("{0}: {1:." + str(n) + "}%").format(x, y * 100)
        else:
            statement = lambda x, y: ("{0}: {1:." + str(n) + "}").format(x, y)

        if np.max(probs) == 1.0:
            idx = np.argmax(probs)
            for i, (state, prob) in enumerate(marginal.items()):
                if i == idx:
                    print(
                        bcolors.BOLD
                        + bcolors.OKGREEN
                        + statement(state, prob)
                        + bcolors.ENDC
                    )
                else:
                    print(bcolors.FAIL + statement(state, prob) + bcolors.ENDC)

        elif np.allclose(*probs):
            for i, (state, prob) in enumerate(marginal.items()):
                print(bcolors.OKBLUE + statement(state, prob) + bcolors.ENDC)
        else:
            idx = np.argmax(probs)
            for i, (state, prob) in enumerate(marginal.items()):
                if i == idx:
                    print(bcolors.OKGREEN + statement(state, prob) + bcolors.ENDC)
                else:
                    print(bcolors.OKBLUE + statement(state, prob) + bcolors.ENDC)
    print("-" * 50)
