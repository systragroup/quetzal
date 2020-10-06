from shapely import geometry
from tqdm import tqdm
from quetzal.model import model, transportmodel, summarymodel
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from syspy.syspy_utils.data_visualization import trim_axs
from syspy.syspy_utils import data_visualization
from quetzal.io import export_utils
import matplotlib.pyplot as plt

styles = {
    'zones': {'color': 'grey', 'alpha': 0},
    'nodes': {'color': 'grey', 'alpha': 1, 'width':2, 'markersize':10},
    'links': {'color': 'red', 'alpha':1 ,'width': 3},
    'road_nodes': {'color': 'blue', 'width': 1, 'markersize':1},
    'road_links': {'color': 'blue', 'width': 1}
}

def plot_one_path(path, styles=styles, ax=None):
    
    df = styles.loc[[k for k in path if k in styles.index]]
    geometries = list(df['geometry'])
    coord_list = list(geometries[0].centroid.coords) # origin
    for g in list(geometries[1:-1]): 
        coord_list += list(g.coords)
    coord_list += list(geometries[-1].centroid.coords) # destination
    full_path = geometry.LineString(coord_list)
    
    ax = gpd.GeoSeries(full_path).plot(color='black', linestyle='dotted', ax=ax)
    grouped = df.groupby(['color', 'width', 'alpha', 'markersize'], as_index=False)['geometry'].agg(list)
    for color, width, alpha, geometries, markersize in grouped[
        ['color', 'width','alpha', 'geometry', 'markersize']].values.tolist():
        s = gpd.GeoSeries(geometries)
        ax = s.plot(color=color, linewidth=width, ax=ax, alpha=alpha, markersize=markersize)
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
                
            to_concat.append(df[style_columns+ ['geometry']] )
        geometries = pd.concat(to_concat)
        geometries['alpha'].fillna(1, inplace=True)
        geometries['width'].fillna(1, inplace=True)
        geometries['color'].fillna('black', inplace=True)
        geometries['markersize'].fillna(1, inplace=True)
        return geometries

    def od_basemap(self, origin, destination, alpha=0.5, color='grey', *args, **kwargs):
        s = gpd.GeoSeries(self.zones.loc[[origin, destination]]['geometry'])
        ax=s.plot(alpha=alpha, color=color, *args, **kwargs)
        s.centroid.plot(ax=ax, color='black')   
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
        ax = self.od_basemap(origin,  destination, *args, **kwargs)
        paths = self.pt_los.set_index(['origin', 'destination']).loc[origin, destination]
        if paths.ndim == 1: # their is only one path 
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

    def plot_separated_paths(
            self, 
            origin, 
            destination, 
            ax=None,
            rows=1,
            title=None,
            basemap_url=None,
            zoom=9,
            resize=False,
            *args,
            **kwargs,
    ):
        styles = self.get_geometries()
        paths = self.pt_los.set_index(['origin', 'destination']).loc[origin, destination]
        if paths.ndim == 1: # their is only one path 
            paths = pd.DataFrame(data=paths).T 
        paths['title'] = '' if title is None else paths[title] 
        g_id_set = set.union(*[set(p) for p in paths['path']])
        

        columns = len(paths)//rows + bool(len(paths)%rows)
        fig, ax_array = plt.subplots(rows, columns,*args, **kwargs) 
        if len(paths) > 1:
            ax_array = trim_axs(ax_array, len(paths))
        axes = fig.get_axes()
                
        if paths.ndim == 1: # their is only one path 
            paths = pd.DataFrame(data=paths).T 
            
        i = 0
        # the path is added to the ax
        for p, t in tqdm(list(paths[['path', 'title']].values.tolist())):
            ax = self.od_basemap(origin,  destination, ax=axes[i])
            ax = plot_one_path(p, styles, ax=ax)
            gpd.GeoDataFrame(styles.loc[g_id_set]).plot(alpha=0, ax=ax)
            if len(t):
                ax.set_title(t)
            ax.set_xticks([])
            ax.set_yticks([])
            i +=1 

        if basemap_url is not None:
            assert self.epsg == 3857
            for ax in axes:
                data_visualization.add_basemap(ax, url=basemap_url, zoom=zoom)

        if resize:

            ax = fig.get_axes()[0]
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.set_size_inches(bbox.width*columns, rows*(bbox.height))
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
            self.loaded_links[by]==route,
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
                    df.loc[df['direction_id']==directions[0]],
                    df.loc[df['direction_id']==directions[1]]
                )
                fig, axes = export_utils.create_two_directions_load_b_a_graph(both)
                fig.suptitle(title)
                plt.setp(axes, ylim=[-max_value, max_value])
                if yticks is not None:
                    yticks = sorted(set(yticks).union(-yticks))
                    plt.setp(axes, yticks=yticks, yticklabels=map(abs,yticks))

        if graph_direction == 'single':
            fig, ax_array = plt.subplots(2,1)
            axes = fig.get_axes()
            export_utils.plot_load_b_a_for_loadedlinks(df.loc[df['direction_id']==directions[0]], ax=axes[0])
            axes[0].set_title('direction {}'.format(directions[0]))
            export_utils.plot_load_b_a_for_loadedlinks(df.loc[df['direction_id']==directions[0]], ax=axes[1])
            axes[1].set_title('direction {}'.format(directions[1]))
            fig.suptitle(title)
            plt.setp(axes, ylim=[0,max_value])
            if yticks is not None:
                plt.setp(axes, yticks=yticks)

        return fig, axes


def _both_directions_graph_possible(df):
    try:
        directions = df['direction_id'].value_counts()
        assert(len(directions)==2), 'Cannot plot both directions'
        assert(directions.iloc[0] == directions.iloc[1]), 'Cannot plot both directions'
        d0 = df[df['direction_id']==directions.keys()[0]][['a','b']].apply(lambda x: tuple(sorted(x)), 1).values
        d1 = df[df['direction_id']==directions.keys()[1]][['b','a']].apply(lambda x: tuple(sorted(x)), 1).values
        assert(set(d0)==set(d1)), 'Cannot plot both directions'
        return True
    except AssertionError as e:
        if str(e) == 'Cannot plot both directions':
            return False
        else:
            raise(e)
