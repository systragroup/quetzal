"""
**The Voyager class takes advantage from the executable file voyager.exe to launch CUBE scripts using command lines.**

Launching a CUBE script with a command line requires the voyager folder that contains voyager.exe to belong to the
system(windows) path. In order to be able to launch CUBE scripts using command lines, you must add the voyager folder
to the system path.

If CUBE has been properly installed, this folder must be in the program files, maybe at:
*'C:\Program Files (x86)\Citilabs\CubeVoyager'*.

It may be added to one's system path with the following method:

    * Panneau de configuration\Tous les Panneaux de configuration\Système
    * Paramètres système avancés
    * Variable d'environnement
    * Variables utilisateur
    * PATH (create PATH or add your voyager folder to its list)

example:
::

    import pycube
    voyager = pycube.voyager.Voyager(r'N:/python/voyager')
    voyager.build_net(node_dataframe, link_dataframe , r"Q:\cube_network.net")

"""

import os
import pandas as pd
import shapely

from syspy.io.pandasdbf import pandasdbf
from syspy.io.pandasshp import pandasshp


class Voyager:
    def __init__(self, environment):
        self.environment = environment

    def mat_to_dbf(
        self,
        input_matrix,
        output_dbf,
        fields=None,
        n_tab=1,
        debug=False
    ):

        """
        creates a dbf from a cube matrix,
        requires fields OR n_tab = len(fields)

        :param input_matrix: path to a cube matrix (.mat)
            dont forget .mat in the name !
        :type input_matrix: str
        :param output_dbf: path to the dbf to create
        :type output_dbf: str
        :param fields: list of the fields of the input matrix
        :type fields: list
        :param n_tab: number of tabs of the matrix
            (required if the fields are not provided)
        :type n_tab: int
        :param debug: switch to manual control of the script launcher if True
        :type debug: bool
        :return: None
        """

        script_text = r"""
            RUN PGM=MATRIX PRNFILE="format_env\mat_to_dbf.prn" MSG='mat_to_dbf'

            FILEI MATI[1] = filei_mati
            FILEO RECO[1] = fileo_reco,
                FIELDS = I, J, field_names

            JLOOP

                RO.I=I
                RO.J=J
                rec_in_jloop
                WRITE RECO = 1

            ENDJLOOP

            ENDRUN
        """

        if not fields:
            tabs = ['tab_%i' % (i + 1) for i in range(n_tab)]
            fields = tabs
        else:
            n_tab = len(fields)
        field_names = ', '.join(fields)

        filei_mati = '"%s"' % input_matrix
        fileo_reco = '"%s"' % output_dbf
        rec_in_jloop = '        '.join(['RO.%s = MI.1.%s \n' % (fields[i], i + 1) for i in range(n_tab)])

        # creating a cube script
        script = open(self.environment + r'\mat_to_dbf.s', 'w', encoding='latin')

        script.write(script_text.replace(
            'format_env', self.environment).replace(
            'filei_mati', filei_mati).replace(
            'fileo_reco', fileo_reco).replace(
            'field_names', field_names).replace(
            'rec_in_jloop', rec_in_jloop))

        script.close()

        # runs the script with voyager.exe
        options = """/Start /CloseWhenDone /Minimize /NoSplash""" if not debug else ""
        os.system('voyager.exe "' + self.environment + r'\mat_to_dbf.s" ' + options)

    def mat_to_csv(
        self,
        input_matrix,
        output_csv,
        fields=None,
        n_tab=1,
        debug=False,
        i='origin',
        j='destination'
    ):
        """
        creates a csv from a cube matrix, requires fields OR n_tab = len(fields)

        :param input_matrix: path to a cube matrix (.mat)
        :type input_matrix: str
        :param output_csv: path to the csv to create ()
        :type output_csv: str
        :param fields: list of the fields of the input matrix
        :type fields: list
        :param n_tab: number of tabs of the matrix (required if the fields are not provided)
        :type n_tab: int
        :param debug: switch to manual control of the script launcher if True
        :type debug: bool
        :return: None
        """
        script_text = r"""
            RUN PGM=MATRIX PRNFILE="format_env\mat_to_csv.prn" MSG='mat_to_csv'

            FILEI MATI[1] = filei_mati
            FILEO PRINTO[1] = fileo_printo

            print_headers
            JLOOP
                print_in_jloop
            ENDJLOOP

            ENDRUN
        """
        if fields is None:
            tabs = ['tab_%i' % (i + 1) for i in range(n_tab)]
            fields = tabs
        else:
            n_tab = len(fields)
        field_names = ', '.join(fields)

        filei_mati = '"%s"' % input_matrix
        fileo_printo = '"%s"' % output_csv

        print_headers = 'IF (I = 1) \n PRINT LIST ="' + '" ,";" ,"'.join([i, j] + fields) + '" PRINTO = 1 \n ENDIF'
        print_assignation = '        '.join(['%s = MI.1.%s \n' % (fields[i].replace(' ', '_'), i + 1) for i in range(n_tab)])
        print_statement = 'PRINT LIST = I, ";", J, ";", ' + ',";",'.join([f.replace(' ', '_') for f in fields]) + ' PRINTO = 1'
        print_in_jloop = print_assignation + ' \n' + print_statement

        # creating a cube script
        script = open(self.environment + r'\mat_to_csv.s', 'w', encoding='latin')
        script.write(script_text.replace(
            'format_env', self.environment).replace(
            'filei_mati', filei_mati).replace(
            'fileo_printo', fileo_printo).replace(
            'field_names', field_names).replace(
            'print_in_jloop', print_in_jloop).replace('print_headers', print_headers))
        script.close()

        # runs the script with voyager.exe
        options = """/Start /CloseWhenDone /Minimize /NoSplash""" if not debug else ""
        os.system('voyager.exe "' + self.environment + r'\mat_to_csv.s" ' + options)

    def net_to_dbf(self, input_network, output_links, output_nodes, debug=False):
        """
        creates a dbf from a cube network

        :param input_network: path to a cube network (.net)
        :type input_network: str
        :param output_links: path to the linkfile dbf create ()
        :type output_links: str
        :param output_nodes: path to the nodefile dbf to create ()
        :type output_nodes: str
        :param debug: switch to manual control of the script launcher if True
        :type debug: bool
        :return: None
        """
        script_text = r"""
            RUN PGM=NETWORK PRNFILE="%s\net_to_dbf.prn"
            FILEI LINKI[1] = "%s"
            FILEO LINKO = "%s"
            FILEO NODEO = "%s"
            ENDRUN
        """ % (self.environment, input_network, output_links, output_nodes)

        # creating a cube script
        script = open(self.environment + r'\net_to_dbf.s', 'w', encoding='latin')
        script.write(script_text)
        script.close()

        # runs the script with voyager.exe
        options = """/Start /CloseWhenDone /Minimize /NoSplash""" if not debug else ""
        os.system('voyager.exe "' + self.environment + r'\net_to_dbf.s" ' + options)

    def build_net_from_links_shape(
        self, links, output_network, first_node=0, length=False, debug=False,
        add_symmetric=False, write_shp=False, shp_kwargs={}
    ):
        name = output_network.replace('.net', '').replace('.NET', '')
        links_to_shp = name + '_links.shp'
        nodes_to_shp = name + '_nodes.shp'

        links['coordinates_a'] = links['geometry'].apply(lambda c: c.coords[-1])
        links['coordinates_b'] = links['geometry'].apply(lambda c: c.coords[0])

        coordinate_list = list(set(list(links['coordinates_a'])).union(list(links['coordinates_b'])))
        coordinate_dict = {first_node + i: coordinate_list[i] for i in range(len(coordinate_list))}

        nodes = pd.DataFrame(pd.Series(coordinate_dict)).reset_index()
        nodes.columns = ['n', 'coordinates']

        links = pd.merge(links, nodes.rename(columns={'coordinates': 'coordinates_a'}), on='coordinates_a', how='left')
        links = pd.merge(links, nodes.rename(columns={'coordinates': 'coordinates_b'}), on='coordinates_b', how='left',
                         suffixes=['_a', '_b'])

        links.drop(['a', 'b', 'A', 'B', 'coordinates_a', 'coordinates_b'], axis=1, errors='ignore', inplace=True)
        links.rename(columns={'n_a': 'a', 'n_b': 'b'}, inplace=True)
        links = links.groupby(['a', 'b'], as_index=False).first()

        links = pandasdbf.convert_stringy_things_to_string(links)

        if length:
            links[length] = links['geometry'].apply(lambda g: g.length)

        if add_symmetric:
            sym = links.copy()
            sym['a'], sym['b'] = links['b'], links['a']
            sym = sym[sym['a'] != sym['b']]
            links = pd.concat([links, sym])

        nodes['geometry'] = nodes['coordinates'].apply(shapely.geometry.point.Point)
        if write_shp:
            pandasshp.write_shp(nodes_to_shp, nodes, **shp_kwargs)
            pandasshp.write_shp(links_to_shp, links, **shp_kwargs)

        links.drop(['geometry'], axis=1, errors='ignore', inplace=True)

        self.build_net(nodes[['n', 'geometry']], links.fillna(0), output_network, debug=debug)

    def build_net(self, nodes, links, output_network, from_geometry=True, debug=False):
        """
        creates a Cube .NET from links and nodes geoDataFrames

        :param output_network: path to a cube network (.net)
        :type output_network: str
        :param nodes:
        :type nodes: pd.DataFrame with geometry field
        :param links:
        :type links: pd.DataFrame
        :param from_geometry: calculate x and y fields from a shapely geometry if True
        :type debug: bool
        :param debug: switch to manual control of the script launcher if True
        :type debug: bool
        :return: None
        """
        _nodes = nodes.copy()
        _links = links.copy()

        if from_geometry:
            _nodes[['x', 'y']] = _nodes['geometry'].apply(lambda g: pd.Series([g.coords[0][0], g.coords[0][1]]))
            _nodes.drop(['geometry'], axis=1, errors='ignore', inplace=True)

        pandasdbf.write_dbf(_nodes, self.environment + r'\temp_nodes_to_dbf.dbf', pre_process=False)
        pandasdbf.write_dbf(_links, self.environment + r'\temp_links_to_dbf.dbf', pre_process=False)

        script_text = r"""

        RUN PGM=NETWORK PRNFILE="%s\temp_net.prn"
        FILEO NETO = "%s"
        FILEI LINKI[1] = "%s"
        FILEI NODEI[1] = "%s"
        ENDRUN

        """ % (
            self.environment,
            output_network,
            self.environment + r'\temp_links_to_dbf.dbf',
            self.environment + r'\temp_nodes_to_dbf.dbf'
        )

        # creating a cube script
        script = open(self.environment + r'\build_net.s', 'w', encoding='latin')
        script.write(script_text)
        script.close()

        # runs the script with voyager.exe
        options = """/Start /CloseWhenDone /Minimize /NoSplash""" if not debug else ""
        cmd = 'voyager.exe "' + self.environment + r'\build_net.s" ' + options
        print(cmd)
        os.system(cmd)

    def dbf_to_mat(self, lookupi, mato, fields, zones, debug=False):
        statements = ''
        lookups = ''
        for i in range(len(fields)):
            statements += "MW[%i] = L(%i,I*1000000+J) \n" % (i + 1, i + 1)
            lookups += "LOOKUP[%i]=lookup, RESULT=%s, \n" % (i + 1, fields[i])

        script_text = r"""RUN PGM=MATRIX PRNFILE="%s\dbf_to_mat.prn"
        FILEI LOOKUPI[1] = "%s"
        FILEO MATO[1] = "%s",
        MO="%s", name = "%s"

        zones = "%i"

        LOOKUP LOOKUPI=1, LIST=Y, NAME=L,
        %s

        JLOOP
        %s
        ENDJLOOP

        ENDRUN""" % (
            self.environment,
            lookupi,
            mato,
            ', '.join([str(i) for i in list(range(1, len(fields) + 1))]),
            ', '.join(fields),
            zones,
            lookups[:-3],
            statements
        )

        # creating a cube script
        script = open(self.environment + r'\dbf_to_mat.s', 'w', encoding='latin')
        script.write(script_text)
        script.close()

        # runs the script with voyager.exeS
        options = """/Start /CloseWhenDone /Minimize /NoSplash""" if not debug else ""
        cmd = 'voyager.exe "' + self.environment + r'\dbf_to_mat.s" ' + options
        print(cmd)
        os.system(cmd)

    def _write_lookup_database(df, i, j, path):
        _df = df.copy()
        _df['lookup'] = _df[i] * 1e6 + _df[j]
        print('test')
        print(len(df))
        pandasdbf.write_dbf(_df, path)
    _write_lookup_database = staticmethod(_write_lookup_database)

    def write_mat(self, df, i, j, mato, fields=None, zones=None, debug=False, remove_dbf=True):
        """
        creates a cube .mat from a pd.DataFrame

        :param df: pd.DataFrame to convert
        :type df: pd.DataFrame
        :param i: name of the column of df to use as origin
        :type i: str
        :param j: name of the column of df to use as destination
        :type i: str
        :param mato: path to the .mat to create
        :type mato: str
        :param fields: list of column names to write
        :type fields: list
        :param zones: number of zones
        :type zones: int
        :param debug: switch to manual control of the script launcher if True
        :type debug: bool
        :return: None
        """
        if not zones:
            zones = df[[i, j]].max().max()
        if not fields:
            fields = list(set(df.columns) - {i, j})
        _path = self.environment + r'\dbf_to_mat_from_df.dbf'
        self._write_lookup_database(df, i, j, _path)
        self.dbf_to_mat(_path, mato, fields, zones, debug=debug)
        if remove_dbf:
            os.remove(_path)

    def read_mat(
        self,
        input_matrix,
        output_csv,
        fields=None,
        n_tab=1,
        debug=False,
        i='origin',
        j='destination',
        remove_csv=True
    ):
        """
        creates a csv from a cube matrix, requires fields OR n_tab = len(fields), returns a matrix

        :param input_matrix: path to a cube matrix (.mat)
        :type input_matrix: str
        :param output_csv: path to the csv to create ()
        :type output_csv: str
        :param fields: list of the fields of the input matrix
        :type fields: list
        :param n_tab: number of tabs of the matrix (required if the fields are not provided)
        :type n_tab: int
        :param debug: switch to manual control of the script launcher if True
        :type debug: bool
        :return: the pd.DataFrame corresponding to the matrix
        """
        n_tab = len(fields) if fields is not None else n_tab
        self.mat_to_csv(input_matrix, output_csv, n_tab=n_tab, debug=debug, i=i, j=j)
        df = pd.read_csv(output_csv, sep=';', encoding='latin')
        df.set_index([i, j], inplace=True)
        if fields is not None:
            df.columns = fields
        if (remove_csv):
            os.remove(output_csv)
        return df.reset_index()
