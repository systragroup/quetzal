def build_neighbors(edges):
    neighbors = {}
    for a, b, w in edges:
        try:
            neighbors[a].add(b)
        except KeyError:
            neighbors[a] = {b}
        try:
            neighbors[b].add(a)
        except KeyError:
            neighbors[b] = {a}
    return neighbors

def vertex_indexed_weights(edges):
    ab_dict = {a: dict() for a, b, w in edges}
    ba_dict = {b: dict() for a, b, w in edges}

    for a, b, w in edges:
        try:
            ab_dict[a][b] = w
        except KeyError:
            ab_dict[a] = {b: w}
        try:
            ba_dict[b][a] = w
        except KeyError:
            ba_dict[b] = {a: w}
    return ab_dict, ba_dict

def combine_edges(edges, keep=set()):

    neighbors = build_neighbors(edges)

    # find nodes of degree two
    degree_two_or_less = {k for k, n in neighbors.items() if len(n) <= 2}
    to_combine = degree_two_or_less - keep

    ab_dict, ba_dict = vertex_indexed_weights(edges)
    new_edges = set()
    shortcuts = {}

    to_combine = {
        n for n in to_combine 
        if n in ab_dict and n in ba_dict
    } # dead_ends can not be combined

    def pop_node(node, ab_dict, ba_dict, new_edges, shortcuts):

        
        ab = ab_dict.pop(node)
        ba = ba_dict.pop(node)

        for p, w_left in ba.items():  # predecessors
            ab_dict[p].pop(node)

        for s, w_right in ab.items():  # successors
            ba_dict[s].pop(node)

            for p, w_left in ba.items():

                if p != s:
                    time = w_left + w_right
                    try:
                        former_time = ab_dict[p][s]
                        if time < former_time:
                            ab_dict[p][s] = time
                            ba_dict[s][p] = time
                            new_edges.add((p, s))
                            left = shortcuts.get((p, node), [p, node])
                            right = shortcuts.get((node, s), [node, s])
                            shortcuts[(p, s)] = left + right[1:]

                    except KeyError:  # the edge does not exist
                        ab_dict[p][s] = time
                        ba_dict[s][p] = time
                        new_edges.add((p, s))
                        left = shortcuts.get((p, node), [p, node])
                        right = shortcuts.get((node, s), [node, s])
                        shortcuts[(p, s)] = left + right[1:]

    for node in to_combine:
        pop_node(node, ab_dict, ba_dict, new_edges, shortcuts)

    e = []
    for a, ss in ab_dict.items():
        for b, w in ss.items():
            e.append([a, b, w])

    combined_edges = {(a, b) for a, b, w in e}
    shortcuts = {k: v for k, v in shortcuts.items() if k in combined_edges}
    new_edges = new_edges.intersection(combined_edges)

    return e, shortcuts

def expand_path(p, shortcuts):
    path = [p[0]]
    for e in list(zip(p[:-1], p[1:])):  # edge_path
        path += shortcuts.get(e, e)[1:]
    return path
