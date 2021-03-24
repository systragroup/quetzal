import networkx as nx


def io_from_doc(doc):
    try:
        if 'deprecated!' in doc:
            return [], []
    except TypeError:  # argument of type 'NoneType' is not iterable
        return [], []

    doc = doc.replace(' ', ' ')  # espace insécable

    try:
        requirements = doc.split('* requires: ')[1].split('\n')[0]
        requirements = [r.strip() for r in requirements.split(',')]
    except IndexError:
        requirements = []
    try:
        products = doc.split('* builds: ')[1].split('\n')[0]
        products = [r.strip() for r in products.split(',')]
    except IndexError:
        products = []
    return requirements, products


def contain_pattern(s, patterns):
    for p in patterns:
        if p in s:
            return True
    return False


class DocModel:
    def __init__(self):
        pass

    def io_from_method(self, name):
        method = self.__getattribute__(name)
        doc = method.__doc__
        return io_from_doc(doc)

    def edges_from_method(self, name):
        inputs, outputs = self.io_from_method(name)
        return [(i, name) for i in inputs] + [(name, o) for o in outputs]

    def dot(self, patterns, header=None):

        header = """
        ratio = fill;
        node [style=filled, fontname = "calibri", fontsize=24, color="#C8D2B3"];
        edge[ fontname = "calibri", fontsize=24];
        ranksep = "0.5";
        rankdir="HR";
        """ if header is None else header

        methods = [m for m in dir(self) if contain_pattern(m, patterns)]
        edges = []
        for method in methods:
            edges += self.edges_from_method(method)

        g = nx.DiGraph()
        g.add_edges_from(edges)

        # colors
        color = "#AACDDA"
        input_color = "#EEC880"
        output_color = "#C8D2B3"

        reversed_g = g.reverse()

        for node in list(g.nodes):
            if not bool(list(g.predecessors(node))):
                g.nodes[node]['color'] = input_color
            elif not bool(list(reversed_g.predecessors(node))):
                g.nodes[node]['color'] = output_color
            else:
                g.nodes[node]['color'] = color

        for method in methods:
            try:
                g.nodes[method]['color'] = '#E89196'
                g.nodes[method]['shape'] = 'rectangle'
            except KeyError:
                pass

        dot = nx.nx_pydot.to_pydot(g)

        return dot.to_string().replace('{', '{' + header)
