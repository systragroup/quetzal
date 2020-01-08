def find_optimal_strategy(edges, destination, inf=1e9):
    
    zero= 1 / inf
    
    # set of all nodes
    nodes = set.union(*[{i, j} for ix, i, j, f, c in edges])
    
    # create index in order to access edges by j
    j_edges = {node: [] for node in nodes} 
    for e in edges:
        ix, i, j, f, c = e
        j_edges[j].append(e)
        
    # initialization
    j = r = destination
    u = {r: 0} # node distance
    f = {node: zero for node in nodes}
    strategy = set()
    
    edge_data = {ix: (i, j, fa, ca) for ix, i, j, fa, ca in edges}
    distance = {ix : u[j] + ca for ix, i, j, fa, ca in j_edges[destination]}
    while(len(distance)):
        # Get next link
        ix = min(distance, key=distance.get)
        a = i, j, fa, ca = edge_data[ix]
        
        # Update node label
        if u.get(i, inf) >= u[j] + ca:
            u[i] = (f[i] * u.get(i, inf) + fa * (u[j] + ca)) / (f[i] + fa)
            f[i] = f[i] + fa
            strategy.add(ix)
            for ixa, i, j, fa, ca in j_edges[i]:
                distance[ixa] = u[j] + ca    
        distance.pop(ix)
              
    return strategy, u, f

def assign_optimal_strategy(sources, edges, u, f):
    
    distance = {
        (ix, i, j, fa, ca) : u[j] + ca
        for  ix, i, j, fa, ca in edges
    }
    
    nodes = set.union(*[{i, j} for ix, i, j, f, c in edges])
    
    node_v = {node: 0 for node in nodes}
    node_v.update(sources)
    edge_v = {}

    # do for every link a in A, in decreasing order of u[j] + ca
    relevant = list(distance.items())
    relevant.sort(key=lambda x: x[1])

    while len(relevant):
        a = ix, i, j, fa, ca =  relevant.pop()[0]
        if node_v[i] > 0:
            edge_v[ix] = (fa / f[i]) * node_v[i]
            node_v[j] = node_v[j] + edge_v[ix]
            
    return node_v, edge_v