# -*- coding: utf-8 -*-

import os
import itertools
import zipfile
import kml2geojson
from shapely import geometry
import shapely
import json
import pandas as pd
from tqdm import tqdm

from syspy.io.pandasshp import pandasshp

def list_files(path, patterns):
    files = [
        os.path.join(path, file)
        for file in os.listdir(path)
        if file.split('.')[-1].lower() in patterns
    ]
    subdirectories = [
        os.path.join(path, dir)
        for dir in os.listdir(path)
        if os.path.isdir(os.path.join(path, dir))
    ]
    files += list(itertools.chain(
            *[
                list_files(subdirectory, patterns)
                for subdirectory in subdirectories
            ]
        )
    )
    return files


def read_kmz_folder(folder):
    geometries = []
    files = list_files(folder, ['kmz'])

    for filename in files:

        # ValueError: Unknown geometry type: geometrycollection

        kmz = zipfile.ZipFile(filename, 'r')
        kml = kmz.open('doc.kml', 'r')
        to_write = kml.read().decode()
        kmlname = filename.replace('.kmz', '.kml').split(folder)[1]
        kmlfilename = folder + 'temp.kml'
        with open(kmlfilename, 'w') as test:
            test.write(to_write)

        kml2geojson.convert(
            kmlfilename,
            folder + 'temp'
        )

        with open(folder + 'temp/temp.geojson', 'r') as file:
            d = json.load(file)

        to_add = []
        for g in d['features']:
            try:
                to_add.append(
                    (
                        g['properties']['name'],
                        shapely.geometry.shape(g['geometry']),
                        kmlname
                    )
                )
            except:
                print('test')

        geometries += to_add

    return pd.DataFrame(geometries, columns=['name', 'geometry', 'kml'])


def read_kmz(folder, kmzname):
    
    kmzfilename = (folder + kmzname + '.kmz').replace('.kmz.kmz', '.kmz')
    geometries = []
    # ValueError: Unknown geometry type: geometrycollection
    
    kmz = zipfile.ZipFile(kmzfilename, 'r')

    with kmz.open('doc.kml', 'r') as kml:
        to_write = kml.read().decode()

    to_format = to_write.split('<Folder>')[0] + '%s'+ to_write.split('</Folder>')[-1]
    insert_strings = [s.split('</Folder>')[0] for s in to_write.split('<Folder>')[1:]]
    
    kmlfilename = folder + 'temp.kml'

    for insert in tqdm(insert_strings):
        to_add = []
        name = insert.split('<name>')[1].split('</name>')[0]
        to_write = to_format % insert

        with open(kmlfilename, 'w') as file:
            file.write(to_write)

        kml2geojson.convert(
            kmlfilename,
            folder + 'temp'
        )

        with open(folder + 'temp/temp.geojson', 'r') as file:
            d = json.load(file)

        print(len(d['features']))
        for g in d['features']:
            try:
                desc = g['properties']['description']
            except:
                desc = ''
            try:
                to_add.append(
                    (
                        g['properties']['name'],
                        desc,
                        shapely.geometry.shape(g['geometry']),
                        kmzname,
                        name
                    )
                )
            except ValueError: # Unknown geometry type: geometrycollection
                pass

        geometries += to_add
        
    layers = pd.DataFrame(
        geometries, 
        columns=['name', 'desc','geometry', 'kmz', 'folder']
    )
    
    
    return layers

def write_shp_by_folder(layers, shapefile_folder, **kwargs) :
            
        for folder in tqdm(set(layers['folder'])):
            
            try:
                layer = layers.loc[layers['folder'] == folder]
                pandasshp.write_shp(
                    shapefile_folder +'//'+ folder + '.shp',
                    layer, 
                    **kwargs
                )
            except KeyError:
                print(folder)