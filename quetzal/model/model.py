import json
import ntpath
import os
import pickle
import shutil
import sys
import uuid
import zlib
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import wraps

import geopandas as gpd
import pandas as pd
from quetzal.io import hdf_io
from quetzal.model.integritymodel import IntegrityModel
from shapely.geometry import Point
from syspy.syspy_utils.data_visualization import add_basemap, add_north, add_raster, add_scalebar
from tqdm import tqdm


def read_hdf(filepath):
    m = Model()
    m.read_hdf(filepath)
    return m


def list_dir(folder):
    if folder.startswith('s3://'):
        import s3fs

        s3 = s3fs.S3FileSystem()
        prefix = folder.split('s3://')[1]
        files = s3.ls(prefix)
        return [f.split('/')[-1] for f in files]
    else:
        return os.listdir(folder)


def get_file(folder, key):
    filname = '%s/%s.zippedpickle' % (folder, key)
    if folder.startswith('s3://'):
        import s3fs

        s3 = s3fs.S3FileSystem()
        return s3.open(filname, 'rb')
    else:
        return open(filname, 'rb')


def log(text, debug=False):
    if debug:
        print(text)


def authorized_column(df, column, authorized_types=(str, int, float)):
    type_set = set(df[column].dropna().apply(type))
    delta = type_set - set(authorized_types)
    return delta == set()


def track_args(method):
    # wraps changes decorated attributes for method attributes
    # decorated.__name__ = method.__name__ etc...
    @wraps(method)
    def decorated(self, *args, **kwargs):
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
    reindex=True,
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

        def reindex_function(n):
            return reindex_dict[n]

        links['a'] = links['a'].apply(reindex_function).astype(str)
        links['b'] = links['b'].apply(reindex_function).astype(str)
        nodes.index = pd.Series(nodes.index).apply(reindex_function).astype(str)
    return links, nodes


def merge(left, right, suffixes=['_left', '_right'], how='inner', reindex=True, clear=True):
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
        reindex=reindex,
    )
    return model


def obj_size_fmt(num):
    if num < 10**3:
        return '{:.2f}{}'.format(num, 'B')
    elif (num >= 10**3) & (num < 10**6):
        return '{:.2f}{}'.format(num / (1.024 * 10**3), 'KB')
    elif (num >= 10**6) & (num < 10**9):
        return '{:.2f}{}'.format(num / (1.024 * 10**6), 'MB')
    else:
        return '{:.2f}{}'.format(num / (1.024 * 10**9), 'GB')


class Model(IntegrityModel):
    def __init__(
        self,
        json_database=None,
        json_folder=None,
        hdf_database=None,
        zip_database=None,
        zippedpickles_folder=None,
        omitted_attributes=(),
        only_attributes=None,
        *args,
        **kwargs,
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
            self.read_hdf(hdf_database, omitted_attributes=omitted_attributes, only_attributes=only_attributes)
        elif zip_database:
            self.read_zip(zip_database, omitted_attributes=omitted_attributes, only_attributes=only_attributes)
        elif zippedpickles_folder:
            self.read_zippedpickles(
                zippedpickles_folder, omitted_attributes=omitted_attributes, only_attributes=only_attributes
            )

        self.debug = True

        # Add default coordinates unit and epsg
        if self.epsg is None:
            # print('Model epsg not defined: setting epsg to default one: 4326')
            self.epsg = 4326
        if self.coordinates_unit is None:
            # print('Model coordinates_unit not defined: setting coordinates_unit to default one: degree')
            self.coordinates_unit = 'degree'

    def memory_usage(model, head=10):
        memory_usage_by_variable = pd.DataFrame(
            {k: sys.getsizeof(v) for (k, v) in model.__dict__.items()}, index=['Size']
        )
        memory_usage_by_variable = memory_usage_by_variable.T
        memory_usage_by_variable = memory_usage_by_variable.sort_values(by='Size', ascending=False)
        memory_usage_by_variable['Size'] = memory_usage_by_variable['Size'].apply(lambda x: obj_size_fmt(x))
        return memory_usage_by_variable

    def plot(
        self,
        attribute,
        ticks=False,
        basemap_url=None,
        zoom=12,
        title=None,
        fontsize=24,
        fname=None,
        basemap_raster=None,
        keep_ax_limits=True,
        north_arrow=None,
        scalebar=None,
        *args,
        **kwargs,
    ):
        gdf = gpd.GeoDataFrame(self.__dict__[attribute])
        if self.epsg != 3857 and basemap_url is not None:
            gdf.crs = {'init': 'epsg:{}'.format(self.epsg)}
            gdf = gdf.to_crs(epsg=3857)

        plot = gdf.plot(*args, **kwargs)
        if ticks is False:
            plot.set_xticks([])
            plot.set_yticks([])

        if title:
            plot.set_title(title, fontsize=fontsize)

        if basemap_url is not None:
            add_basemap(plot, zoom=zoom, url=basemap_url)
        if basemap_raster is not None:
            add_raster(plot, basemap_raster, keep_ax_limits=keep_ax_limits)
        if north_arrow is not None:
            add_north(plot)
        if scalebar is not None:
            add_scalebar(plot)
        if fname:
            fig = plot.get_figure()
            fig.savefig(fname, bbox_inches='tight')
        return plot

    def split_attribute(self, attr, by=None, nchunks=None, drop=True):
        # attr = pt_los
        # split self.pt_los in several attributes self.pt_los_1, self.pt_los_2 etc
        if by is not None:
            pool = self.__getattribute__(attr).set_index(by, append=True, drop=drop).swaplevel()
            self.__delattr__(attr)
            keys = set(pool.index.levels[0])
            for k in keys:
                self.__setattr__('%s_%s' % (attr, str(k)), pool.loc[k])

        elif nchunks is not None:
            pool = self.__getattribute__(attr)
            self.__delattr__(attr)
            length = len(pool)
            chunk_size = length // nchunks + 1
            pd.Series(range(length)) // chunk_size
            for i in range(nchunks):
                self.__setattr__('%s_%s' % (attr, str(i)), pool.iloc[i * chunk_size : (i + 1) * chunk_size])

    def merge_attribute(self, attr, keys=None):
        # attr = pt_los
        # merge several attributes self.pt_los_1, self.pt_los_2 etc in self.pt_los
        if keys is None:
            keys = [s.split(attr + '_')[1] for s in self.__dict__.keys() if s.startswith(attr + '_')]
        to_concat = []
        for key in keys:
            to_concat.append(self.__getattribute__(attr + '_' + key))
            self.__delattr__(attr + '_' + key)
        self.__setattr__(attr, pd.concat(to_concat))

    def to_zippedpickles(
        self, folder, omitted_attributes=(), only_attributes=None, max_workers=1, complevel=-1, remove_first=True
    ):
        if remove_first:
            shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        if max_workers == 1:
            iterator = tqdm(self.__dict__.items())
            for key, value in iterator:
                iterator.desc = key
                if key in omitted_attributes:
                    continue
                if only_attributes is not None and key not in only_attributes:
                    continue
                hdf_io.to_zippedpickle(value, '%s/%s.zippedpickle' % (folder, key), complevel=complevel)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for key, value in self.__dict__.items():
                    if key in omitted_attributes:
                        continue
                    if only_attributes is not None and key not in only_attributes:
                        continue
                    executor.submit(
                        hdf_io.to_zippedpickle, value, r'%s/%s.zippedpickle' % (folder, key), complevel=complevel
                    )

    def read_zippedpickles(self, folder, omitted_attributes=(), only_attributes=None):
        files = list_dir(folder)
        keys = [file.split('.zippedpickle')[0] for file in files if '.zippedpickle' in file]
        iterator = tqdm(keys)
        for key in iterator:
            if key in omitted_attributes:
                continue
            if only_attributes is not None and key not in only_attributes:
                continue

            iterator.desc = key
            with get_file(folder, key) as file:
                buffer = file.read()
                bigbuffer = zlib.decompress(buffer)
                self.__setattr__(key, pickle.loads(bigbuffer))

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

    def read_zip(self, filepath, omitted_attributes=(), only_attributes=None):
        if only_attributes is not None:
            only_attributes = {'/' + a for a in only_attributes}.union(only_attributes)
        omitted_attributes = {'/' + a for a in omitted_attributes}.union(omitted_attributes)

        # read the zip in a buffer
        with open(filepath, 'rb') as file:
            data = file.read()
            bigbyte = zlib.decompress(data)

        # build a store from the buffer
        with pd.HDFStore(
            'quetzal-%s.h5' % str(uuid.uuid4()),
            mode='r',
            driver='H5FD_CORE',
            driver_core_backing_store=0,
            driver_core_image=bigbyte,
        ) as store:
            iterator = tqdm(store.keys())
            for key in iterator:
                skey = key.split(r'/')[-1]
                iterator.desc = skey

                if key in omitted_attributes:
                    continue
                if only_attributes is not None and key not in only_attributes:
                    continue

                value = store.select(key)
                if isinstance(value, pd.DataFrame) and 'geometry' in value.columns:
                    value = gpd.GeoDataFrame(value)

                self.__setattr__(skey, value)
                value = None

            # some attributes may have been store in the json_series
            try:
                json_dict = self.jsons.to_dict()
                for key, value in json_dict.items():
                    self.__setattr__(key, json.loads(value))
            except AttributeError:
                print('could not read json attributes')

    def to_excel(self, filepath, prefix='stack'):
        length = len(prefix)
        stacks = {name[length + 1 :]: attr for name, attr in self.__dict__.items() if name[:length] == prefix}
        with pd.ExcelWriter(filepath) as writer:
            for name, stack in stacks.items():
                stack.reset_index().to_excel(writer, sheet_name=name, index=False)

    def to_frames(self, omitted_attributes=(), only_attributes=None):
        """
        export the full model to a dataframe dict
        """
        jsons = {}
        attributeerrors = []
        frames = {}
        for key, attribute in self.__dict__.items():
            if key in omitted_attributes:
                continue
            if only_attributes is not None and key not in only_attributes:
                continue

            elif isinstance(attribute, gpd.GeoSeries):
                frames[key] = pd.Series(attribute).astype(object)
            elif isinstance(attribute, (pd.DataFrame, pd.Series, gpd.GeoDataFrame)):
                df = pd.DataFrame(attribute)
                try:
                    df['geometry'] = df['geometry'].astype(object)
                except Exception:
                    pass
                frames[key] = df
            else:
                try:
                    jsons[key] = json.dumps(attribute)
                except TypeError:
                    attributeerrors.append(key)
        frames['jsons'] = pd.Series(jsons)
        return frames

    def to_zip(self, filepath, complevel=None, *args, **kwargs):
        frames = self.to_frames(*args, **kwargs)
        buffer = hdf_io.write_hdf_to_buffer(frames, complevel=complevel)
        smallbuffer = zlib.compress(buffer)
        with open(filepath, 'wb') as file:
            file.write(smallbuffer)

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
                df = pd.DataFrame(attribute)
                df['geometry'] = df['geometry'].astype(object)
                df.to_hdf(filepath, key=key, mode='a')
            elif isinstance(attribute, gpd.GeoSeries):
                pd.Series(attribute).astype(object).to_hdf(filepath, key=key, mode='a')
            elif isinstance(attribute, (pd.DataFrame, pd.Series)):
                df = attribute
                try:
                    df['geometry'] = df['geometry'].astype(object)
                except Exception:
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

    def to_zipped_hdf(self, filepath, *args, **kwargs):
        filedir = ntpath.dirname(filepath)
        tempdir = filedir + '/quetzal_temp' + '-' + str(uuid.uuid4())
        os.mkdir(tempdir)
        hdf_path = tempdir + r'/model.hdf'
        self.to_hdf(hdf_path, *args, **kwargs)
        shutil.make_archive(filepath.split('.zip')[0], 'zip', tempdir)
        shutil.rmtree(tempdir)

    @track_args
    def to_json(
        self, folder, omitted_attributes=(), only_attributes=None, verbose=False, encoding='utf-8', use_fiona=True
    ):
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
                geojson_columns = [c for c in df.columns if authorized_column(df, c) or c in ('index', 'geometry')]
                json_columns = [c for c in df.columns if c not in geojson_columns]
                try:
                    if use_fiona:
                        gpd.GeoDataFrame(attribute[geojson_columns]).to_file(
                            geojson_file, driver='GeoJSON', encoding=encoding
                        )
                    else:
                        gdf = gpd.GeoDataFrame(attribute[geojson_columns])
                        geojson = gdf.to_json()
                        crs_json = json.dumps(
                            {'crs': {'type': 'name', 'properties': {'name': 'EPSG:{e}'.format(e=gdf.crs.to_epsg())}}}
                        )[1:-1]
                        splitted = geojson.split(',', 1)
                        geojson = splitted[0] + ',' + crs_json + ',' + splitted[1]
                        with open(geojson_file, 'w') as outfile:
                            outfile.write(geojson)

                    if len(json_columns):
                        attribute[json_columns + ['index']].to_json(root_name + '_quetzaldata.json')
                # "['geometry'] not in index"
                except (AttributeError, KeyError) as e:
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

    def read_json(self, folder, encoding='utf-8'):
        files = os.listdir(folder)
        geojson_attributes = [file.split('.geojson')[0] for file in files if '.geojson' in file]
        json_attributes = [file.split('.json')[0] for file in files if '.json' in file]

        for key in json_attributes:
            value = pd.read_json('%s/%s.json' % (folder, key))
            value.set_index('index', inplace=True)
            self.__setattr__(key, value)

        for key in geojson_attributes:
            value = gpd.read_file('%s/%s.geojson' % (folder, key), encoding=encoding)
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
        json_database = {'geojson': {}, 'pd_json': {}, 'json': {}}

        attributeerrors = []
        for key, attribute in iterator:
            # If DataFrame attribute
            if isinstance(attribute, (pd.DataFrame, pd.Series)):
                # Check index
                assert attribute.index.is_unique, 'DataFrame attributes must have unique index'
                attribute = pd.DataFrame(attribute)  # copy and type conversion
                attribute.drop('index', axis=1, errors='ignore', inplace=True)
                attribute.index.name = 'index'
                attribute = attribute.reset_index()  # loss of index name

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
        return json.dumps(json_database)

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
            projected_model.__dict__.items(), desc='Reprojecting model from epsg {} to epsg {}'.format(self.epsg, epsg)
        )
        failed = []

        for key, attribute in iterator:
            if isinstance(attribute, (gpd.GeoDataFrame, gpd.GeoSeries)):
                try:
                    attribute.crs = {'init': 'epsg:{}'.format(self.epsg)}
                    attribute = attribute.to_crs(epsg=epsg)
                    projected_model.__setattr__(key, attribute)
                except RuntimeError:
                    # b'tolerance condition error', b'latitude or longitude exceeded limits'
                    failed.append(key)
            elif isinstance(attribute, pd.DataFrame):
                if 'geometry' in attribute.columns:
                    try:
                        # print('Converting {}'.format(key))
                        temp = gpd.GeoDataFrame(attribute)
                        temp.crs = {'init': 'epsg:{}'.format(self.epsg)}
                        attribute = pd.DataFrame(temp.to_crs(epsg=epsg))
                        projected_model.__setattr__(key, attribute)
                    except RuntimeError:
                        # b'tolerance condition error', b'latitude or longitude exceeded limits'
                        failed.append(key)

            elif isinstance(attribute, pd.Series):
                try:
                    temp = gpd.GeoSeries(attribute)
                    temp.crs = {'init': 'epsg:{}'.format(self.epsg)}
                    attribute = pd.Series(temp.to_crs(epsg=epsg))
                    projected_model.__setattr__(key, attribute)
                except Exception:
                    if attribute.name == 'geometry':
                        failed.append(key)
        projected_model.epsg = epsg
        projected_model.coordinates_unit = coordinates_unit

        if len(failed) > 0:
            print('could not change epsg for the following attributes: ')
            print(failed)
        return projected_model

    def describe(self):
        data = []
        # general case
        for attr, value in self.__dict__.items():
            if isinstance(value, bool):
                desc = value
            elif isinstance(value, str):
                desc = value
            elif isinstance(value, int):
                desc = value
            elif isinstance(value, gpd.GeoDataFrame) and value.centroid.equals(gpd.GeoSeries({0: Point(0, 0)})):
                continue
            elif isinstance(value, pd.DataFrame) and value.empty:
                continue
            elif isinstance(value, list) and len(value) < 4:
                desc = value
            elif isinstance(value, dict) and len(value) < 4:
                desc = value
            else:
                try:
                    desc = len(value)
                except TypeError:
                    desc = None
            data.append([attr, type(value), desc])
        # specific cases
        return pd.DataFrame(data=data, columns=['name', 'type', 'desc']).set_index('name').sort_index()
