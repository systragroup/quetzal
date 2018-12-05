# -*- coding: utf-8 -*-

__author__ = 'qchasserieau'

import networkx as nx


class DijkstraMonkey:

    """
    nx.DiGraph based object. Runs Dijkstra algorithm to infer the extensive sequence of nodes from a partial
    sequence. Add intermediate nodes to a .LIN to fit a detailed network.

    example:
    ::
        df_edges =links[['a', 'b', 'length']]
        edges = [tuple(df_edges.ix[i]) for i in df_edges.index]  # edges is a list of tuples (a, b, length)
        edsger = pycube.lin.DijkstraMonkey(edges)
        edsger.make_lin(data_path + 'monterrey_2045_edited.lin', data_path + 'monterrey_2045_edited_dijkstra.lin')
    """

    def __init__(self, edges):
        self.g = nx.DiGraph()
        self.g.add_weighted_edges_from(edges)

    def _to_insert(path):
        """ returns a string containing all the intermediates nodes of a path in .lin format """
        return ''.join(['N=-' + str(int(path[i])) + ', ' for i in range(1, len(path)-1)])
    _to_insert = staticmethod(_to_insert)

    def _path_from_pair(self, pair):
        _pair = [int(i.replace('-','')) for i in pair]
        """ tries to return the shortest path between to points of a pair"""
        try:
            path = nx.dijkstra_path(self.g, _pair[0], _pair[1])
        except:
            print('fail at ' + str(pair))
            path = []

        return path

    def _line_with_intermediate_nodes(self, line):
        """ returns a string string with all the intermediates nodes from a .lin line string"""

        sequence = line.split('N=')
        pairs = [[sequence[i].split(',')[0], sequence[i+1].split(',')[0]] for i in range(1,len(sequence)-1)]
        paths = [self._path_from_pair(pair) for pair in pairs]

        k = 2
        for i in range(0,len(pairs)):
            if len(self._to_insert(paths[i])):
                sequence.insert(i + k, self._to_insert(paths[i]))
                k +=1
        return 'N='.join(sequence).replace('N=N','N')

    def make_lin(self, from_lin_file, to_lin_file, sep=None):
        """
        make a .lin file from another using dijkstra algorithm to add intermediates nodes

        :param from_lin_file: the lin to process
        :param to_lin_file: the path to the lin to save
        :param sep: the string that separates the lines in the file ex: 'LINE NAME'
        :return: None
        """

        # lecture du .LIN d'origine

        with open(from_lin_file, 'r') as f:
            lines = f.readlines()

        if sep:
            with open(from_lin_file, 'r') as f:
                lines = _to_n_equal(f.read()).split(sep)


        # construction de la liste des lignes du nouveau .LIN à partir de celle de l'ancien
        lines_with_intermediate_nodes = [self._line_with_intermediate_nodes(line) for line in lines]


        # écriture du nouveau .LIN
        with open(to_lin_file, 'w') as f:
            f.write(''.join(lines_with_intermediate_nodes))
        if sep:
            with open(to_lin_file, 'w') as f:
                f.write(sep.join(lines_with_intermediate_nodes))


def _to_n_equal(l):
    return ','.join([_chunk_to_n_equal(c) for c in l.split(',')])


def _chunk_to_n_equal(c):
    return 'N=' + c if _represents_int(c) else c


def _represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
