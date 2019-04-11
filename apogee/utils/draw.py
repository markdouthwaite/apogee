import networkx as nx
import matplotlib.pyplot as plt


def draw_network(network, **kwargs):
    graph = network.graph()
    nx.spring_layout(graph, **kwargs)
    nx.draw(
        graph,
        labels=nx.get_node_attributes(graph, "name"),
        width=3,
        edge_color="black",
        node_size=1500,
        node_color="gray",
    )
    plt.show()
