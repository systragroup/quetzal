# -*- coding: utf-8 -*-

import json
import os

import geopandas as gpd
import pandas as pd

from copy import deepcopy
from functools import wraps
from tqdm import tqdm
import shutil
import uuid
import ntpath

from syspy.syspy_utils.data_visualization import add_basemap

from quetzal.model.integritymodel import IntegrityModel


def read_hdf(filepath):
    m = Model()
    m.read_hdf(filepath)
    return m


def log(text, debug=False):
    if debug:
        print(text)

def authorized_column(
    df, 
    column, 
    authorized_types=(str, int, float)
):
    type_set = set(df[column].dropna().apply(type))
    delta = type_set - set(authorized_types)
    return delta == set()

def track_args(method):

    # wraps changes decorated attributes for method attributes
    # decorated.__name__ = method.__name__ etc...
    @wraps(method)
    def decorated(self,  *args, **kwargs):

        """
        All the parameters are stored
        if use_tracked_args=True is passed to a method, the last parameters
        passed to this method are used
        """
        try:
            use_tracked_args = kwargs.pop('use_tracked_args')
        except KeyError:
            use_tracked_args = False

        name = method.__name__
        debug = self.debug if 'debug' in self.__dict__ else True

        if use_tracked_args:
            args = self.parameters[name]['args']
            kwargs = self.parameters[name]['kwargs']
            log('using parameters from self.parameters:', debug)
            log('args:' + str(args), debug)
            log('kwargs:' + str(kwargs), debug)
        else:
            self.parameters[name] = {}
            self.parameters[name]['args'] = args
            self.parameters[name]['kwargs'] = kwargs
        return method(self, *args, **kwargs)

    return decorated


def merge_links_and_nodes(
    left_links,
    left_nodes,
    right_links,
    right_nodes,
    suffixes=['_left', '_right'],
    join_nodes='inner',
    join_links='inner',
    reindex=True
):
    left_links = left_links.copy()
    right_links = right_links.copy()
    left_nodes = left_nodes.copy()
    right_nodes = right_nodes.copy()

    # suffixes
    sx_left = suffixes[0]
    sx_right = suffixes[1]

    # we can not use suffixes_on non string indexes
    left_links[['a', 'b']] = left_links[['a', 'b']].astype(str) + sx_left
    right_links[['a', 'b']] = right_links[['a', 'b']].astype(str) + sx_right
    left_nodes.index = pd.Series(left_nodes.index).astype(str) + sx_left
    right_nodes.index = pd.Series(right_nodes.index).astype(str) + sx_right

    # join = 'inner' means that we only keep columns that are shared
    nodes = pd.concat([left_nodes, right_nodes], join=join_nodes)
    links = pd.concat([left_links, right_links], join=join_links)

    if reindex:

        reindex_series = pd.Series(range(len(nodes)), index=nodes.index)
        reindex_dict = reindex_series.to_dict()
        reindex_function = lambda n: reindex_dict[n]
        links['a'] = links['a'].apply(reindex_function).astype(str)
        links['b'] = links['b'].apply(reindex_function).astype(str)
        nodes.index = pd.Series(nodes.index).apply(reindex_function).astype(str)
    
    return links, nodes


def merge(
    left,
    right,
    suffixes=['_left', '_right'],
    how='inner',
    reindex=True,
    clear=True
):
    assert left.epsg == right.epsg
    # we want to return an object with the same class as left
    model = left.__class__(epsg=left.epsg, coordinates_unit=left.coordinates_unit) if clear else left.copy()
    

    model.links, model.nodes = merge_links_and_nodes(
        left_links=left.links,
        left_nodes=left.nodes,
        right_links=right.links,
        right_nodes=right.nodes,
        suffixes=suffixes,
        join_nodes=how,
        join_links=how,
        reindex=reindex
    )

    return model


class Model(IntegrityModel):

    def __init__(
        self,
        json_database=None,
        json_folder=None,
        hdf_database=None,
        omitted_attributes=(),
        only_attributes=None,
        *args,
        **kwargs
    ):

        """
        Initialization function, either from a json folder or a json_database representation.
        Args:
            json_database (json): a json_database representation of the model. Default None.
            json_folder (json): a json folder representation of the model. Default None.
        Examples:
        >>> sm = stepmodel.Model(json_database=json_database_object)
        >>> sm = stepmodel.Model(json_folder=folder_path)
        """
        super().__init__(*args, **kwargs)

        if json_database and json_folder:
            raise Exception('Only one argument should be given to the init function.')
        elif json_database:
            self.read_json_database(json_database)
        elif json_folder:
            self.read_json(json_folder)
        elif hdf_database:
            self.read_hdf(
                hdf_database, 
                omitted_attributes=omitted_attributes,
                only_attributes=only_attributes
            )

        self.debug = True

        # Add default coordinates unit and epsg
        if self.epsg is None:
            print('Model epsg not defined: setting epsg to default one: 4326')
            self.epsg = 4326
        if self.coordinates_unit is None:
            print('Model coordinates_unit not defined: setting coordinates_unit to default one: degree')
            self.coordinates_unit = 'degree'

    def plot(
        self, attribute, ticks=False, 
        basemap_url=None, zoom=12, 
        title=None, fontsize=24,
        fname=None,  
        *args, **kwargs
    ):
        gdf = gpd.GeoDataFrame(self.__dict__[attribute])
        if self.epsg != 3857 and basemap_url is not None:
            gdf.crs = {'init': 'epsg:{}'.format(self.epsg)}
            gdf = gdf.to_crs(epsg=3857)
        
        plot = gdf.plot(*args, **kwargs)
        if ticks == False:
            plot.set_xticks([])
            plot.set_yticks([])

        if title: 
            plot.set_title(title, fontsize=fontsize)

        if basemap_url is not None:
            add_basemap(plot, zoom=zoom, url=basemap_url)
        if fname:
            fig = plot.get_figure()
            fig.savefig(fname, bbox_inches='tight')

        return plot

    def read_hdf(self, filepath, omitted_attributes=(), only_attributes=None):
        with pd.HDFStore(filepath) as hdf:
            keys = [k.split('/')[1] for k in hdf.keys()]

        iterator = tqdm(keys, desc='read_hdf: ')
        for key in iterator:
            if key in omitted_attributes:
                continue
            if only_attributes is not None and key not in only_attributes:
                continue
            value = pd.read_hdf(filepath, key)
            if isinstance(value, pd.DataFrame) and 'geometry' in value.columns:
                value = gpd.GeoDataFrame(value)
            self.__setattr__(key, value)

        # some attributes may have been store in the json_series
        try:
            json_dict = self.jsons.to_dict()
            for key, value in json_dict.items():
                self.__setattr__(key, json.loads(value))
        except AttributeError:
            print('coul not read json attributes')

    @track_args
    def to_hdf(self, filepath, omitted_attributes=(), only_attributes=None):
        """
        export the full model to a hdf database
        """
        try:
            os.remove(filepath) 
            status = 'overwriting'
        except FileNotFoundError:
            status = 'new file'
        jsons = {}
        iterator = tqdm(self.__dict__.items(), desc='to_hdf(%s)' % status)

        attributeerrors = []
        for key, attribute in iterator:
            if key in omitted_attributes:
                continue
            if only_attributes is not None and key not in only_attributes:
                continue

            if isinstance(attribute, gpd.GeoDataFrame):
                df=pd.DataFrame(attribute)
                df['geometry'] = df['geometry'].astype(object)
                df.to_hdf(filepath, key=key, mode='a')
            elif isinstance(attribute, gpd.GeoSeries):
                pd.Series(attribute).astype(object).to_hdf(filepath, key=key, mode='a')
            elif isinstance(attribute, (pd.DataFrame, pd.Series)):
                df = attribute
                try:
                    df['geometry'] = df['geometry'].astype(object)
                except:
                    pass
                df.to_hdf(filepath, key=key, mode='a')
            else:
                try:
                    jsons[key] = json.dumps(attribute)
                except TypeError:
                    attributeerrors.append(key)

        for key in attributeerrors:
            log('could not save attribute: ' + key, self.debug)

        json_series = pd.Series(jsons)
        # mode=a : we do not want to overwrite the file !
        json_series.to_hdf(filepath, 'jsons', mode='a')

    def to_zip(self, filepath, *args, **kwargs):
        filedir = ntpath.dirname(filepath)
        tempdir = filedir + '/quetzal_temp' + '-' + str(uuid.uuid4())
        os.mkdir(tempdir)
        hdf_path = tempdir+ r'/model.hdf'
        self.to_hdf(hdf_path, *args, **kwargs)
        shutil.make_archive(filepath.split('.zip')[0], 'zip', tempdir)
        shutil.rmtree(tempdir)

    @track_args
    def to_json(self, folder, omitted_attributes=(), only_attributes=None, verbose=False):
        
        """
        export the full model to a hdf database
        """
        try:
            os.makedirs(folder, exist_ok=True) 
            status = 'overwriting'
        except FileNotFoundError:
            status = 'new file'
            
        jsons = {}
        iterator = tqdm(self.__dict__.items(), desc='to_hdf(%s)' % status)

        attributeerrors = []
        for key, attribute in iterator:
            if key in omitted_attributes:
                continue
            if only_attributes is not None and key not in only_attributes:
                continue

            root_name = folder + '/' + key
            geojson_file = root_name + '.geojson'
            json_file = root_name + '.json'
            
            for filename in (geojson_file, json_file):
                try: 
                    os.remove(filename)
                except OSError:
                    pass

            if isinstance(attribute, (pd.DataFrame, pd.Series)):
                
                msg = 'datframe attributes must have unique index:' + key
                assert attribute.index.is_unique, msg
                attribute = pd.DataFrame(attribute)  # copy and type conversion
                attribute.drop('index', axis=1, errors='ignore', inplace=True)
                attribute.index.name = 'index'
                attribute = attribute.reset_index()  # loss of index name
                attribute.rename(columns={x: str(x) for x in attribute.columns}, inplace=True)

                df = attribute
                geojson_columns = [c for c in df.columns 
                    if authorized_column(df, c) 
                    or c in ('index', 'geometry')
                ] 
                json_columns = [c for c in df.columns if c not in geojson_columns]
                try:
                    gpd.GeoDataFrame(attribute[geojson_columns]).to_file(
                        geojson_file, 
                        driver='GeoJSON'
                    )
                    if len(json_columns):
                        attribute[json_columns + ['index']].to_json(root_name + '_quetzaldata.json')
                except (AttributeError, KeyError, ) as e: # "['geometry'] not in index"
                    if verbose:
                        print(e)
                    df = pd.DataFrame(attribute).drop('geometry', axis=1, errors='ignore')
                    df.to_json(root_name + '.json')       
                except ValueError as e: 
                    if verbose:
                        print(e)
                    # Geometry column cannot contain mutiple geometry types when writing to file.
                    print('could not save geometry from table ' + key)
                    df = pd.DataFrame(attribute).drop('geometry', axis=1, errors='ignore')
                    df.to_json(root_name + '.json')     
                    
            else:
                try:
                    jsons[key] = json.dumps(attribute)
                except TypeError:
                    attributeerrors.append(key)

        for key in attributeerrors:
            print('could not save attribute: ' + key)
            
        json_series = pd.Series(jsons)
        json_series.name = 'json'
        json_series = json_series.reset_index()
        json_series.to_json(folder + '/' + 'jsons.json')
    
    def read_json(self, folder):
        files = os.listdir(folder)
        geojson_attributes = [file.split('.geojson')[0] for file in files if '.geojson' in file]
        json_attributes = [file.split('.json')[0] for file in files if '.json' in file]
        
        for key in json_attributes:

            value = pd.read_json('%s/%s.json' % (folder, key))
            value.set_index('index', inplace=True)
            self.__setattr__(key, value)
            
        for key in geojson_attributes:
            
            value = gpd.read_file('%s/%s.geojson' % (folder, key))
            value.set_index('index', inplace=True)
            self.__setattr__(key, pd.DataFrame(value))
            
        # some attributes may have been store in the json_series
            try:
                json_dict = self.jsons['json'].to_dict()
                for key, value in json_dict.items():
                    self.__setattr__(key, json.loads(value))
            except AttributeError:
                print('error')

        # some attributes have been stored separately in quetzaldata.json files
        to_delete = []
        for key, data in self.__dict__.items():
            if '_quetzaldata' in key:
                key_to_set = key.split('_quetzaldata')[0]
                left = self.__getattribute__(key_to_set)
                merged = pd.merge(left, data, left_index=True, right_index=True)
                self.__setattr__(key_to_set, merged)
                to_delete.append(key)
        for key in to_delete:
            self.__delattr__(key)

    @track_args
    def to_json_database(self):
        """
        Dumps the model into a single json organized as follow:
        json_database = {
            'geojson': {    # Contains all GeoDataFrame objects
                key: value
            },
            'pd_json': {    # Contains all DataFrame objects but GeoDataFrame
                key: value
            },
            'json': {       # Contains all other objects (model parameters)
                key: value
            }
        }
        Args:
            stepmodel
        Returns:
            json_database (json): the single json representation of the model
        """

        iterator = tqdm(self.__dict__.items(), desc='to_json_database')

        # Create json_database object
        json_database = {
            'geojson': {},
            'pd_json':{},
            'json': {}
        }

        attributeerrors = []
        for key, attribute in iterator:
            # If DataFrame attribute
            if isinstance(attribute, (pd.DataFrame, pd.Series)):
                # Check index
                assert attribute.index.is_unique, 'DataFrame attributes must have unique index'
                attribute = pd.DataFrame(attribute) # copy and type conversion
                attribute.drop('index', axis=1, errors='ignore', inplace=True)
                attribute.index.name = 'index'
                attribute = attribute.reset_index() # loss of index name

                try:
                    geojson_to_store = gpd.GeoDataFrame(attribute).to_json()
                    json_database['geojson'].update({key: geojson_to_store})

                except KeyError:
                    df = pd.DataFrame(attribute).drop('geometry', axis=1, errors='ignore')
                    json_to_store = df.to_json()
                    json_database['pd_json'].update({key: json_to_store})

            # Else parameter attribute
            else:
                try:
                    json_database['json'][key] = json.dumps(attribute)
                except TypeError:
                    attributeerrors.append(key)

            for key in attributeerrors:
                print('could not save attribute: ' + key)

        return(json.dumps(json_database))

    def read_json_database(self, json_database):
        """
        Load model from its json_database representation.
        Args:
            stepmodel
            json_database (json): the json_database model representation
        
        Returns:
            None
        """
        # Load json
        json_database = json.loads(json_database)

        # Geojson objects
        for key, attr in json_database['geojson'].items():
            value = gpd.GeoDataFrame.from_features(json.loads(attr))
            value.set_index('index', inplace=True)  
            self.__setattr__(key, pd.DataFrame(value))

        # Dataframe objects
        for key, attr in json_database['pd_json'].items():
            value = pd.read_json(attr)
            value.set_index('index', inplace=True)  
            self.__setattr__(key, pd.DataFrame(value))

        # Parameters
        for key, attr in json_database['json'].items():
            self.__setattr__(key, json.loads(attr))

    def copy(self):
        copy = deepcopy(self)
        return copy

    def merge(self, *args, **kwargs):
        return merge(left=self, *args, **kwargs)


    @track_args
    def change_epsg(self, epsg, coordinates_unit):

        projected_model = self.copy()
        iterator = tqdm(
            projected_model.__dict__.items(),
            desc='Reprojecting model from epsg {} to epsg {}'.format(self.epsg, epsg)
        )
        failed = []

        for key, attribute in iterator:
            if isinstance(attribute, (gpd.GeoDataFrame, gpd.GeoSeries)):
                try:
                    attribute.crs = {'init': 'epsg:{}'.format(self.epsg)}
                    attribute = attribute.to_crs(epsg=epsg)
                    projected_model.__setattr__(key, attribute)
                except RuntimeError: 
                #b'tolerance condition error', b'latitude or longitude exceeded limits'
                    failed.append(key)
            elif isinstance(attribute, pd.DataFrame):
                if 'geometry' in attribute.columns:
                    try:
                        # print('Converting {}'.format(key))
                        temp = gpd.GeoDataFrame(attribute)
                        temp.crs =  {'init': 'epsg:{}'.format(self.epsg)}
                        attribute = pd.DataFrame(temp.to_crs(epsg=epsg))
                        projected_model.__setattr__(key, attribute)
                    except RuntimeError: 
                    #b'tolerance condition error', b'latitude or longitude exceeded limits'
                        failed.append(key)


            elif isinstance(attribute, pd.Series):
                try:
                    temp = gpd.GeoSeries(attribute)
                    temp.crs =  {'init': 'epsg:{}'.format(self.epsg)}
                    attribute = pd.Series(temp.to_crs(epsg=epsg))
                    projected_model.__setattr__(key, attribute)
                except:
                    if attribute.name == 'geometry':
                        failed.append(key)
        projected_model.epsg = epsg
        projected_model.coordinates_unit = coordinates_unit

        if len(failed) > 0:
            print('could not change epsg for the following attributes: ')
            print(failed)

        return projected_model