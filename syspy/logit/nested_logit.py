from multinomial_logit import nest_probabilities, nest_utility
import networkx as nx

def plot_nests(nests):
    g = nx.DiGraph(nests)
    root = [n for n in g.nodes if g.in_degree(n) == 0][0]
    lengths = nx.single_source_shortest_path_length(g, root)
    pos = {}
    levels = [0] * (max(lengths.values()) + 1)
    for key, x in lengths.items():
        pos[key] = [levels[x],  - x]
        levels[x] += 1
    plot = nx.draw(
        g, 
        pos=pos, 
        node_color='white', 
        alpha=1, 
        node_size=1000,
        arrows=False,
        edge_color='green',
        font_size=15,
        font_weight='normal',
        labels={k: k for k in g.nodes}
    )
    return plot