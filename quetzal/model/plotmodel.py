import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from IPython.core import display
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from quetzal.io import export_utils
from quetzal.model import summarymodel
from shapely import geometry
from syspy.syspy_utils import data_visualization
from syspy.syspy_utils.data_visualization import trim_axs
from tqdm import tqdm
import subprocess


styles = {
    'zones': {'color': 'grey', 'alpha': 0},
    'nodes': {'color': 'grey', 'alpha': 1, 'width': 2, 'markersize': 10},
    'links': {'color': 'red', 'alpha': 1, 'width': 3},
    'road_nodes': {'color': 'blue', 'width': 1, 'markersize': 1},
    'road_links': {'color': 'blue', 'width': 1}
}


def plot_one_path(path, styles=styles, ax=None):
    df = styles.loc[[k for k in path if k in styles.index]]
    geometries = list(df['geometry'])
    coord_list = list(geometries[0].centroid.coords)  # origin
    for g in list(geometries[1:-1]):
        try:
            coord_list += list(g.coords)
        except NotImplementedError:  # In case a zone is in the path
            coord_list += list(g.centroid.coords)
    coord_list += list(geometries[-1].centroid.coords)  # destination
    full_path = geometry.LineString(coord_list)

    ax = gpd.GeoSeries(full_path).plot(color='black', linewidth=2, ax=ax, aspect='equal')
    grouped = df.groupby(['color', 'width', 'alpha', 'markersize'], as_index=False)['geometry'].agg(list)
    for color, width, alpha, geometries, markersize in grouped[
        ['color', 'width', 'alpha', 'geometry', 'markersize']
    ].values.tolist():
        s = gpd.GeoSeries(geometries)
        ax = s.plot(color=color, linewidth=width, ax=ax, alpha=alpha, markersize=markersize, aspect='equal')
    return ax


class PlotModel(summarymodel.SummaryModel):
    def get_geometries(self, styles=styles):
        to_concat = []
        for key, style in styles.items():
            df = self.__getattribute__(key).copy()
            style_columns = []
            for column, value in style.items():
                style_columns.append(column)
                df[column] = df.get(column, value)
                df[column].fillna(value, inplace=True)

            to_concat.append(df[style_columns + ['geometry']])
        geometries = pd.concat(to_concat)
        geometries['alpha'].fillna(1, inplace=True)
        geometries['width'].fillna(1, inplace=True)
        geometries['color'].fillna('black', inplace=True)
        geometries['markersize'].fillna(1, inplace=True)
        return geometries

    def od_basemap(self, origin, destination, alpha=0.5, color='grey', squared=False, *args, **kwargs):
        s = gpd.GeoSeries(self.zones.loc[[origin, destination]]['geometry'])
        ax = s.plot(alpha=alpha, color=color, *args, **kwargs)
        s.centroid.plot(ax=ax, color='black')

        if squared:
            xw = abs(ax.get_xlim()[0] - ax.get_xlim()[1])
            yw = abs(ax.get_ylim()[0] - ax.get_ylim()[1])
            if xw > yw:
                mid = (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2
                ax.set_ylim([mid - xw / 2, mid + xw / 2])
            else:
                mid = (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2
                ax.set_xlim([mid - yw / 2, mid + yw / 2])

        return ax

    def plot_paths(
        self,
        origin,
        destination,
        ax=None,
        separated=False,
        basemap_url=None,
        zoom=9,
        *args,
        **kwargs
    ):

        styles = self.get_geometries()
        ax = self.od_basemap(origin, destination, *args, **kwargs)
        paths = self.pt_los.set_index(['origin', 'destination']).loc[origin, destination]
        if paths.ndim == 1:  # their is only one path
            paths = pd.DataFrame(data=paths).T

        # the path is added to the ax
        for p in tqdm(list(paths['path'])):
            ax = plot_one_path(p, styles, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])

        if basemap_url is not None:
            assert self.epsg == 3857
            data_visualization.add_basemap(ax, url=basemap_url, zoom=zoom)
        return ax

    def plot_car_paths(
        self,
        origin,
        destination,
        ax=None,
        title=None,
        titlesize=12,
        separated=False,
        basemap_url=None,
        basemap_raster=None,
        north_arrow=None,
        scalebar=None,
        zoom=9,
        styles=None,
        squared=False,
        *args,
        **kwargs
    ):

        if styles is None:
            styles = self.get_geometries()
            styles = styles[~styles.index.duplicated(keep='first')]
        ax = self.od_basemap(origin, destination, squared=squared, *args, **kwargs)
        paths = self.car_los.set_index(['origin', 'destination']).loc[origin, destination]
        paths['title'] = '' if title is None else paths[title]
        if paths.ndim == 1:
            paths = pd.DataFrame(data=paths).T

        def build_road_path(p, lp):
            if len(p):
                rp = p[:2]
                rp += lp
                rp += p[-2:]
                return rp
            else:
                return lp
        paths['road_path'] = paths[['path', 'link_path']].apply(
            lambda x: build_road_path(x['path'], x['link_path']), 1
        )
        # the path is added to the ax
        for p, t in tqdm(list(paths[['road_path', 'title']].values.tolist())):
            ax = plot_one_path(p, styles, ax=ax)
            if len(t):
                ax.set_title(t, fontsize=titlesize)
            ax.set_xticks([])
            ax.set_yticks([])

        if basemap_url is not None:
            assert self.epsg == 3857
            data_visualization.add_basemap(ax, url=basemap_url, zoom=zoom)
        if north_arrow is not None:
            data_visualization.add_north(ax)
        if scalebar is not None:
            data_visualization.add_scalebar(ax)
        if basemap_raster is not None:
            data_visualization.add_raster(ax, raster=basemap_raster)
        return ax

    def plot_separated_paths(
        self,
        origin,
        destination,
        ax=None,
        rows=1,
        title=None,
        titlesize=12,
        basemap_url=None,
        basemap_raster=None,
        north_arrow=None,
        scalebar=None,
        zoom=9,
        resize=False,
        styles=None,
        squared=False,
        *args,
        **kwargs,
    ):
        if styles is None:
            styles = self.get_geometries()
            styles = styles[~styles.index.duplicated(keep='first')]
        paths = self.pt_los.set_index(['origin', 'destination']).loc[origin, destination]
        if paths.ndim == 1:  # their is only one path
            paths = pd.DataFrame(data=paths).T
        paths['title'] = '' if title is None else paths[title]
        g_id_set = set.union(*[set(p) for p in paths['path']])

        columns = len(paths) // rows + bool(len(paths) % rows)
        fig, ax_array = plt.subplots(rows, columns, *args, **kwargs)
        if len(paths) > 1:
            ax_array = trim_axs(ax_array, len(paths))
        axes = fig.get_axes()

        if paths.ndim == 1:  # their is only one path
            paths = pd.DataFrame(data=paths).T

        i = 0
        # the path is added to the ax
        for p, t in tqdm(list(paths[['path', 'title']].values.tolist())):
            ax = self.od_basemap(origin, destination, ax=axes[i], squared=squared)
            ax = plot_one_path(p, styles, ax=ax)
            gpd.GeoDataFrame(
                styles.reindex(g_id_set).dropna(subset=['color'])
            ).plot(alpha=0, ax=ax)
            if len(t):
                ax.set_title(t, fontsize=titlesize)
            ax.set_xticks([])
            ax.set_yticks([])
            i += 1

        if basemap_url is not None:
            assert self.epsg == 3857
            for ax in axes:
                data_visualization.add_basemap(ax, url=basemap_url, zoom=zoom)

        if north_arrow is not None:
            for ax in axes:
                data_visualization.add_north(ax)
        if scalebar is not None:
            for ax in axes:
                data_visualization.add_scalebar(ax)

        if basemap_raster is not None:
            for ax in axes:
                data_visualization.add_raster(ax, raster=basemap_raster,adjust=None)

        if resize:
            ax = fig.get_axes()[0]
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.set_size_inches(bbox.width * columns, rows * (bbox.height))
            fig.constrained_layout = True
        return fig, axes

    def plot_load_ba_graph(
        self, route, by='route_id', stop_name_column=None, graph_direction='both',
        max_value=None, yticks=None, reverse_direction=False,
        title=''
    ):
        """
        Plot line graph with load, boardings and alightings

        :param route: name of line to plot. Must be a value of the param 'by' column
        :param by (route_id): name of column to search route from:
        :param stop_name_column (None): column of nodes dataframe containing stop names
        :param graph_direction:
            *both (default): try to plot the two directions in the same graphe
            *single: one graph per direction
        :param reverse_direction (False): plot directions in reversed order

        Returns fig, axes
        """
        df = self.loaded_links.loc[
            self.loaded_links[by] == route,
            ['a', 'b', 'direction_id', 'link_sequence', 'boardings', 'alightings', 'load']
        ]

        directions = df['direction_id'].unique()
        if reverse_direction:
            directions = directions[::-1]

        if max_value is None:
            max_value = df['load'].max()

        if stop_name_column is not None:
            df = df.merge(self.nodes[[stop_name_column]], left_on='a', right_index=True)
            df = df.merge(self.nodes[[stop_name_column]], left_on='b', right_index=True, suffixes=('_a', '_b'))
            df = df.drop(['a', 'b'], 1).sort_values(['direction_id', 'link_sequence']).rename(
                columns={
                    'stop_name_a': 'a',
                    'stop_name_b': 'b'
                }
            )
        if graph_direction == 'both':
            if not _both_directions_graph_possible(df):
                print('Cannot plot bidirectional graph')
                graph_direction = 'single'
            else:
                both = export_utils.directional_loads_to_station_bidirection_load(
                    df.loc[df['direction_id'] == directions[0]],
                    df.loc[df['direction_id'] == directions[1]]
                )
                fig, axes = export_utils.create_two_directions_load_b_a_graph(both)
                fig.suptitle(title)
                plt.setp(axes, ylim=[-max_value, max_value])
                if yticks is not None:
                    yticks = sorted(set(yticks).union(-yticks))
                    plt.setp(axes, yticks=yticks, yticklabels=map(abs, yticks))

        if graph_direction == 'single':
            fig, ax_array = plt.subplots(2, 1)
            axes = fig.get_axes()
            export_utils.plot_load_b_a_for_loadedlinks(df.loc[df['direction_id'] == directions[0]], ax=axes[0])
            axes[0].set_title('direction {}'.format(directions[0]))
            export_utils.plot_load_b_a_for_loadedlinks(df.loc[df['direction_id'] == directions[0]], ax=axes[1])
            axes[1].set_title('direction {}'.format(directions[1]))
            fig.suptitle(title)
            plt.setp(axes, ylim=[0, max_value])
            if yticks is not None:
                plt.setp(axes, yticks=yticks)
        return fig, axes

    def display_aggregated_edges(self, origin, destination, ranksep=0.1, rankdir='LR', *args, **kwargs):
        from graphviz import Source
        a = self.get_aggregated_edges(origin, destination, *args, **kwargs)
        a = a.groupby(['i', 'j'], as_index=False)[['p']].sum()  # for clusters
        a['l'] = 'p=' + np.round(a['p'], 2).astype(str)  # + '\nh:' + a['h'].astype(str)
        a.loc[a['p'] == 1, 'l'] = ''

        odg = nx.DiGraph()
        for e in a.to_dict(orient='records'):
            odg.add_edge(e['i'], e['j'], label=e['l'])
        name = 'test'

        header = """
        ratio = fill;
        node [style="filled,rounded" ,shape="record", fontname = "calibri", fontsize=24,];
        edge[ fontname = "calibri", fontsize=24];
        ranksep = "%s";
        rankdir="%s";
        """ % (str(ranksep), rankdir)
        dot_string = nx.nx_pydot.to_pydot(odg).to_string().replace('{', '{' + header)
        src = Source(dot_string, format='png')
        return display.Image(filename=src.render(name))

    def plot_strategy(
        self, origin, destination, road=False,
        color='red', cmap='Reds', legend='right',
        legend_kwds=None, walk_on_road=False,
        basemap_raster=None,
        north_arrow=None,
        scalebar=None,
        *args, **kwargs
    ):
        try:
            volumes = self.volumes.copy()
            restore = True
        except AttributeError:
            restore = False
            pass
        self.volumes = pd.DataFrame(
            data=[[origin, destination, 1]],
            columns=['origin', 'destination', 'dummy'])
        self.step_strategy_assignment('dummy', road=road, od_set={(origin, destination)})

        self.volumes.drop('dummy', inplace=True, axis=1)

        try:
            loc = set(self.loaded_edges.loc[self.loaded_edges['dummy'] > 0].index)
            access = self.road_links
            for attr in ['zone_to_road', 'road_to_transit']:
                if hasattr(self, attr):
                    access = pd.concat([access, getattr(self, attr)])
            loc = loc.intersection(access.index)
            access = access.loc[loc]
            assert len(access) > 0
        except AssertionError:
            loc = set(self.loaded_edges.loc[self.loaded_edges['dummy'] > 0].index)
            access = self.zone_to_transit
            loc = loc.intersection(access.index)
            access = access.loc[loc]

        links = self.road_links if road else self.links
        links = links.dropna(subset=['dummy'])
        links = links.loc[links['dummy'] > 1e-9]
        links = gpd.GeoDataFrame(links)

        norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
        ax = self.od_basemap(origin, destination, *args, **kwargs)

        # access['dummy'] = 1
        access.plot(ax=ax, alpha=1, color='black', linewidth=2)
        links.plot(ax=ax, alpha=1, color='white', linewidth=7, zorder=3)
        # links.plot(ax=ax, alpha=1, color='black', linewidth=6, zorder=4)
        if road:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(legend, size="2%", pad=0.05)
            links.plot(
                zorder=5,
                ax=ax, alpha=1, column='dummy', linewidth=5, cmap=cmap, norm=norm,
                legend=bool(legend), cax=cax, legend_kwds=legend_kwds)
        else:
            links.plot(ax=ax, alpha=1, color='white', linewidth=5)
            for alpha in set(links['dummy']):
                links.loc[links['dummy'] == alpha].plot(color=color, ax=ax, alpha=alpha, linewidth=5, zorder=5)
        ax.set_yticks([])
        ax.set_xticks([])

        mask = (self.nodes['boardings'] + self.nodes['alightings']) > 1e-9
        nodes = gpd.GeoDataFrame(self.nodes[mask])
        nodes.plot(ax=ax, marker=10, markersize=200, zorder=10, column='boardings', cmap=cmap, norm=norm, linewidth=0)
        nodes.plot(ax=ax, marker=11, markersize=200, zorder=10, column='alightings', cmap=cmap, norm=norm, linewidth=0)

        if north_arrow is not None:
            data_visualization.add_north(ax)
        if scalebar is not None:
            data_visualization.add_scalebar(ax)
        if basemap_raster is not None:
            data_visualization.add_raster(ax, raster=basemap_raster)

        if restore:
            self.volumes = volumes.copy()
        return ax

    def plot_line_arc_diagram(self, line, stop_label_col=None, graph_direction='both', **kwargs):
        line_od_vol = export_utils.compute_line_od_vol(self, line=line, **kwargs)
        if stop_label_col is not None:
            stop_names = self.nodes[stop_label_col].to_dict()
            line_od_vol.rename(columns=stop_names, index=stop_names, inplace=True)
            assert line_od_vol.index.duplicated().sum() == 0

        return export_utils.arc_diagram_from_dataframe(line_od_vol)
    

    def data_loom_plot(self, filtering={'agency_id': None, 'route_type': None, 'route_id': None}, utm_epsg=2154):

        sm_map = self.copy()
        for col, l_val in filtering.items():
            if l_val is not None:
                sm_map.links = sm_map.links[sm_map.links[col].isin(l_val)]

        # Passage en système métrique pour clusterisation des arrêts
        sm_map.links = sm_map.links.to_crs(utm_epsg)
        sm_map.nodes = sm_map.nodes.to_crs(utm_epsg)
        nodes = sm_map.nodes.copy()     # save copy to add stop names as station labels

        nodeset = set(sm_map.links["a"].values).union(set(sm_map.links["b"].values))
        sm_map.nodes = sm_map.nodes.loc[list(nodeset)]

        sm_map.preparation_clusterize_nodes(distance_threshold=20)
        sm_map.nodes = gpd.GeoDataFrame(sm_map.nodes, crs=utm_epsg)
        sm_map.nodes["id"] = sm_map.nodes.index

        def add_stop_name(nodes):

            sm_map.nodes = sm_map.nodes.reset_index()
            sm_map.nodes['cluster'] = sm_map.nodes['cluster'].astype(int)
            sm_map.nodes = sm_map.nodes.merge(sm_map.node_parenthood.reset_index()[['index', 'cluster']], on='cluster', how='left')

            sm_map.nodes = sm_map.nodes.merge(nodes[['stop_name']], left_on='index', right_on=nodes.index, how='left')

            sm_map.nodes = sm_map.nodes.drop(columns='index')

            def unique_stop_name(df):
                def select_row(group):
                    stop_names = group['stop_name']
                    if stop_names.isnull().all() or stop_names.notnull().all():
                        return group.iloc[0]
                    else:
                        return group[stop_names.notnull()].sample(n=1).iloc[0]

                return df.groupby('id').apply(select_row).reset_index(drop=True)
            
            return unique_stop_name(sm_map.nodes)
        
        sm_map.nodes = add_stop_name(nodes=nodes)

        # Retransformation en (lat, lon) parce que loom ne fonctionne pas sinon
        sm_map = sm_map.change_epsg(epsg=4326, coordinates_unit="degree")

        # Création du dataframe admissible par loom à partir des inputs links et nodes de sm
        sm_map.links['route_color'] = sm_map.links['route_color'].apply(lambda x: x.lower())
        gdata = sm_map.links[["a", "b", "route_id", "route_color", "route_short_name", "geometry"]].rename(
            columns={"a": "from", "b": "to", "route_id": "id", "route_color": "color", "route_short_name": "label"})
        gdata["lines"] = gdata[["color", "id", "label"]].apply(dict, 1)
        gdata["ab"] = gdata[["from", "to"]].apply(lambda x: str(set(x)), 1)

        line_data = gdata.groupby(["ab"],as_index=False).agg({"from": "first", "to": "first", "geometry": "first", "lines": list})
        
        node_data = sm_map.nodes[["id", "cluster", "stop_name", "geometry"]]
        node_data["station_id"] = node_data["id"].map(int).map(str)
        node_data['station_label'] = node_data.apply(lambda row: row['stop_name'] if row['stop_name']!=None else row["cluster"].map(int).map(str), axis=1)
        node_data = node_data.drop(columns='stop_name')
        node_data["deg"] = 2

        data = pd.concat([node_data, line_data])
        data["excluded_conn"] = None
        data = data.drop(["cluster", "ab"], axis=1, errors="ignore")

        return data
    
    def plot_subway_map(self, filtering={'agency_id': None, 'route_type': None, 'route_id': None}, utm_epsg=2154,
                    plot_type='octilinear',
                    plot_kwgs={'-l': True,
                               '--line-width': 200,
                               '--outline-width': 20,
                               '--line-spacing': 20,
                               '--line-label-textsize': 2000,
                               '--station-label-textsize': 1200
                               }
                    ):

        """
        Draws a schematic view of the TC network using loom.
        Requires loom docker installation to run tool with subprocess.
        Takes a StepModel object as input with existing links and nodes edited in Quetzal Network Editor.

        :param filtering: choose which part of the TC network to draw, various filter levels
        :param utm_epsg: metric coordinate system at the model network location
        :param plot_type: graph style expected in the end - choose in ['realistic', 'octilinear', 'orthoradial']
        :param plot_kwgs: input args to adapt to the network size (the more compact the TC network, the lower the values should be)
            default parameters settled for a regional network (150 km x 50 km)
            :param -l: write lines and stations labels
        """

        data = PlotModel.data_loom_plot(self, filtering=filtering, utm_epsg=utm_epsg)
        
        # Steps 1 and 2 : define json graph
        topo = subprocess.run("docker run -i loom topo", input=data.to_json(na="drop").encode(), capture_output=True)
        loom = subprocess.run("docker run -i loom loom --ilp-num-threads 16 -m hillc", input=topo.stdout, capture_output=True)
        
        graph_style = ' '.join(f"{key} {value}" if not isinstance(value, bool) else key for key, value in plot_kwgs.items())

        if plot_type == 'realistic':
            transit = subprocess.run("docker run -i loom transitmap " + graph_style, input=loom.stdout, capture_output=True)
            return loom.stdout.decode(), transit.stdout.decode()

        elif plot_type == 'octilinear':
            octo = subprocess.run("docker run -i loom octi", input=loom.stdout, capture_output=True)
            transit = subprocess.run("docker run -i loom transitmap " + graph_style, input=octo.stdout, capture_output=True)
            return loom.stdout.decode(), transit.stdout.decode()
            
        elif plot_type == 'orthoradial':
            ortho = subprocess.run("docker run -i loom octi -b orthoradial", input=loom.stdout, capture_output=True)
            transit = subprocess.run("docker run -i loom transitmap " + graph_style, input=ortho.stdout, capture_output=True)
            return loom.stdout.decode(), transit.stdout.decode()

        else:
            print("plot_type not available, choose within ['realistic', 'octilinear', 'orthoradial']")
            return   


def _both_directions_graph_possible(df):
    try:
        directions = df['direction_id'].value_counts()
        assert (len(directions) == 2), 'Cannot plot both directions'
        assert (directions.iloc[0] == directions.iloc[1]), 'Cannot plot both directions'
        d0 = df[df['direction_id'] == directions.keys()[0]][['a', 'b']].apply(lambda x: tuple(sorted(x)), 1).values
        d1 = df[df['direction_id'] == directions.keys()[1]][['b', 'a']].apply(lambda x: tuple(sorted(x)), 1).values
        assert (set(d0) == set(d1)), 'Cannot plot both directions'
        return True
    except AssertionError as e:
        if str(e) == 'Cannot plot both directions':
            return False
        else:
            raise e
