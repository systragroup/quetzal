import networkx as nx
from tqdm import tqdm


def pairwise_function(single_source_function, sources, targets, **kwargs):
    resp_dict = {}
    for source in tqdm(sources):
        try:
            to_all_targets = single_source_function(source=source, **kwargs)
            filtered = {
                key: value for key, value in to_all_targets.items()
                if key in targets
            }
            resp_dict[source] = filtered
        except KeyError:
            print(source)
    return resp_dict
