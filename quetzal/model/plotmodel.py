from shapely import geometry
from tqdm import tqdm
from quetzal.model import model, transportmodel, summarymodel
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from syspy.syspy_utils.data_visualization import trim_axs
from syspy.syspy_utils import data_visualization

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