"""
To facilitate handling of OSM data that cannot be done through OSMNX
For graph-related purposes - refer to quetzal.io.osm
"""

import geopandas as gpd
import glob
import osmium
import pandas as pd
import requests
import shapely.wkb as wkblib


def _split_bbox(bbox, n):
        long_min,lat_min,long_max,lat_max = bbox
        longs = [long_min + ((long_max - long_min)/n)*x for x in range(0, n+1)]
        bboxes = [[x,lat_min,y,lat_max] for x,y in zip(longs[:-1], longs[1:])]
        return bboxes


def _add_suffixe(osm_filename, suffix):
        f = osm_filename.split('.osm')
        return '{}{}.osm'.format(f[0], suffix)


def _split_and_download(bbox, export_filename, n=1):
    # split
    bboxes = _split_bbox(bbox, n)

    for i in range(len(bboxes)):
        # download
        r = requests.get(
            "https://api.openstreetmap.org/api/0.6/map?bbox={}".format(','.join([str(s) for s in bboxes[i]])),
            verify=False
        )
        if r.status_code==400 and 'You requested too many nodes' in str(r.content):
            return 'You requested too many nodes'

        # write
        osm_data = r.content
        suffix = '' if n == 1 else '_{}'.format(i)
        filename = _add_suffixe(export_filename, suffix)
        print(r)
        with open(filename, 'wb') as f: 
            f.write(osm_data)

def download_osm_data(bbox, export_filename):
    """
    Download data from OSM through overpass api.
    Automatically split download perimeter 
    if it exceeds OSM api nodes limit (50,000).
    """
    n = 1
    status = _split_and_download(bbox, export_filename, n)
    while status == 'You requested too many nodes':
        print(status, n)
        n += 1
        status = _split_and_download(bbox, export_filename, n)

wkbfab = osmium.geom.WKBFactory()

class OSMHandler(osmium.SimpleHandler):
    def __init__(self):
        osmium.SimpleHandler.__init__(self)
        self.osm_data = []

    def tag_inventory(self, elem, elem_type):
        # get tags
        deep_copy_tags = {}
        for tag in elem.tags:
            deep_copy_tags.update({tag.k: tag.v})
        # get geometry
        if elem_type == 'way':
            wkb = wkbfab.create_linestring(elem)
            geom = wkblib.loads(wkb, hex=True)
        elif elem_type == 'node':
            wkb = wkbfab.create_point(elem)
            geom = wkblib.loads(wkb, hex=True)
        else:
            geom = None

        self.osm_data.append(       
            [
                elem_type, 
                elem.id, 
                elem.version,
                elem.visible,
                pd.Timestamp(elem.timestamp),
                elem.uid,
                elem.user,
                elem.changeset,
                len(elem.tags),
                deep_copy_tags,
                geom
            ]
        )

    def node(self, n):
        # wkb = wkbfab.create_point(n)
        self.tag_inventory(n, "node")

    def way(self, w):
        self.tag_inventory(w, "way")

    def relation(self, r):
        self.tag_inventory(r, "relation")


def _osmdata_files(export_filename):
    prefix = export_filename.split('.osm')[0]
    search = glob.glob(prefix + '*.osm')
    return search


def osmdata_to_df(osm_filenames):
    if len(osm_filenames[0]) == 1:
        osm_filenames = [osm_filenames]
    osmhandlers = {f: OSMHandler() for f in osm_filenames}

    data_colnames = [
        'type', 'id', 'version', 'visible', 'ts', 'uid',
        'user', 'chgset', 'ntags', 'tags', 'geometry'
    ]

    df_osm = pd.DataFrame()
    for k, v in osmhandlers.items():
        v.apply_file(k, locations=True, idx='flex_mem')
        temp = pd.DataFrame(v.osm_data, columns=data_colnames)
        df_osm = pd.concat([df_osm, temp])

    df_osm = df_osm.drop_duplicates(subset=['id'])

    return df_osm

class OSMImporter():
    def __init__(self, bbox=[]):
        self.bbox = bbox
    
    def download(self, export_filename):
        download_osm_data(self.bbox, export_filename)
        self.osm_filenames = _osmdata_files(export_filename)
    
    def to_dataframe(self):
        self.osm_df = osmdata_to_df(self.osm_filenames)

    def extract_road_links_with_tags(self, tags=[]):
        # filter
        self.road_links = self.osm_df[
            (self.osm_df['type']=='way')&
            (self.osm_df['tags'].apply(lambda d: 'highway' in d.keys()))
        ]

        self.road_links = gpd.GeoDataFrame(self.road_links).reset_index(drop=True)

        # add tags as attributes
        columns_to_keep = ['id','user','geometry', 'tags']
        for tag in tags:
            self.road_links[tag] = self.road_links['tags'].apply(lambda d: d.get(tag, None))

        self.road_links = self.road_links[columns_to_keep + tags]
