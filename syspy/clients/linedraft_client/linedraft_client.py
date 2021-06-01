# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np

import requests
import shapely
from shapely.geometry import shape

from syspy.skims import skims
from syspy.io.pandasshp import shapefile
from syspy.spatial import spatial


class LinedraftClient:

    """
    A client for requesting Linedraft's API. Handles location and zonal_data posts. The url to the API root should be
    provided as an initialization parameter.
    """

    def __init__(self, api_root="http://swsdev.systra.info/test/transim/api/"):
        self.api_root = api_root
        self.api_locations = api_root + "locations"
        self.api_zonaldatasummaries = api_root + "zonal_data_summaries"
        self.api_zonaldata = api_root + "zonal_data"
        self.api_projects = api_root + "projects"
        self.api_projectsummaries = api_root + "project_summaries"

    def post_location(self, location_id, location_name, coordinates, comments=None):
        """
        posts a location to Linedraft's API

        :param location_id: id of the location to be posted, all lowercase, with underscores instead of spaces
        :type location_id:  str
        :param location_name: name of the location te be posted, capitalized
        :type location_name:  str
        :param coordinates: coordinates of the location in epsg 4326 {'latitude':y, 'longitude':x}
        :type coordinates: dict
        :return: None
        :rtype: NoneType
        """

        locations = requests.get(self.api_locations)
        df_locations = pd.read_json(locations.text, orient='records')
        if location_id in set(df_locations['id']):
            print("Location already exists")

        else:
            location_dict = {
                "id": location_id,
                "name": location_name,
                "latitude": coordinates['latitude'],
                "longitude": coordinates['longitude'],
                "comments": comments
            }
            location_json = json.dumps(location_dict)
            requests.post(self.api_locations, data=location_json)
            print("New location created - id : " + location_id)

    # TODO: ajouter les locations et zonal_data existants à self rename to
    # post_zona_data_from_shp
    def post_zonal_data(
        self,
        pop_emp_shp,
        location_id,
        zonal_data_name,
        zonal_data_year,
        population_column='pop_dens',
        employment_column='emp_dens',
        comments=''
    ):
        """
        posts a zonal_data to Linedraft's API

        :param pop_emp_shp: path to the ESRI shapefile to be converted and posted (saved in epsg 4326)
        :type pop_emp_shp:  str
        :param location_id: id of the parent location
        :type location_id:  str
        :param zonal_data_name: name of the zonal_data to be posted (containing the year of the survey)
        :type zonal_data_name:  str
        :param zonal_data_year: year of the survey
        :type zonal_data_year:  int
        :param population_column: name of the ESRI shapefile field containing population density
        :type population_column:  str
        :param employment_column: name of the ESRI shapefile field containing employment density
        :type employment_column:  str
        :param comments: comments to be added to the database
        :type comments: str
        :return: None
        :rtype: NoneType
        """

        zonal_data_summaries = requests.get(self.api_zonaldatasummaries)
        df_zonaldatasummaries = pd.read_json(
            zonal_data_summaries.text, orient='records')

        if zonal_data_name in set(df_zonaldatasummaries['name']):
            print("Zonal data with name " + zonal_data_name + " already exists")
        else:
            reader = shapefile.Reader(pop_emp_shp)
            fields = reader.fields[1:]
            field_names = [field[0] for field in fields]

            zone_list = []
            for sr in reader.shapeRecords():
                attributes_dict = dict(zip(field_names, sr.record))
                properties_dict = dict({"pop_dens": attributes_dict.get(population_column),
                                        "emp_dens": attributes_dict.get(employment_column)})
                geom = sr.shape.__geo_interface__
                zone_list.append(
                    dict(type="Feature", geometry=geom, properties=properties_dict))

            zoning_geodict = {
                "type": "FeatureCollection",
                "features": zone_list
            }

            zoning_dict = {
                "location_id": location_id,
                "name": zonal_data_name,
                "year": zonal_data_year,
                "comments": comments,
                "geojson": zoning_geodict
            }

            requests.post(self.api_zonaldata, json.dumps(zoning_dict))

            print("Zonal data has been created")

    def get_project_id(self, name):
        url = self.api_projectsummaries
        res = requests.get(url)
        id_list = [d['id'] for d in json.loads(res.text) if d['name'] == name]
        assert len(id_list) == 1
        return int(id_list[0])

    def get_links_nodes(self, project_id, include_items=False, add_symmetric=True):

        url = self.api_projects + '/' + str(project_id)
        res = json.loads(requests.get(url).text)

        network = res['networks'][0]
        return self.graph_from_network(network, include_items, add_symmetric)

    def graph_from_network(self, network, include_items=False, add_symmetric=True):

        transport_lines = network['transport_lines']
        to_concat = []
        for line in transport_lines:
            df = pd.DataFrame(line['links'])
            df['line'] = line['name']
            df['color'] = line['color']
            df['headway'] = line['headway']
            df['capacity'] = line['capacity']
            df['modeId'] = line['modeId']
            df['maxTraffic'] = line['maxTraffic']

            df['defaultLinearItem'] = line['defaultLinearItem']
            df['defaultNonlinearItem'] = line['defaultNonlinearItem']
            non_item_columns = ['a', 'b', 'line', 'color', 'headway', 'speed',
                                'capacity', 'modeId', 'maxTraffic']

            df['speed'] = line['speed'] if 'speed' in line.keys() else np.nan

            if len(df):
                to_concat.append(df if include_items else df[non_item_columns])

        links = pd.concat(to_concat)
        nodes = pd.DataFrame(network['nodes'])
        links[['headway', 'speed']] = links[['headway', 'speed']].astype(float)
        nodes['id'] = nodes['id'].astype(int)
        nodes.set_index('id', inplace=True)

        if not include_items:
            nodes.drop('nonlinear_items', inplace=True,
                       axis=1, errors='ignore')

        nodes['geometry'] = nodes.apply(
            lambda r: shapely.geometry.point.Point(r['longitude'], r['latitude']), axis=1)

        def line_neighbors(line):
            return links.loc[links['line'] == line].groupby('a')['b'].agg(lambda s: set(s)).to_dict()

        def line_ends(line):
            return list(links.loc[links['line'] == line]['a'].value_counts().index[-2:])

        def line_sequence(line):
            ends = line_ends(line)
            neighbors = line_neighbors(line)

            station = ends[0]
            sequence = [station]

            while station != ends[1]:

                station = list(neighbors[station] - set(sequence))[0]
                sequence.append(station)

            return sequence

        def sequence_dataframe(line):
            ls = line_sequence(line)
            df = pd.DataFrame([(i, ls[i], ls[i + 1], 1, line) for i in range(len(ls) - 1)],
                              columns=['sequence', 'a', 'b', 'direction', 'line'])
            df_sym = pd.DataFrame([(len(ls) - 1 - i, ls[i + 1], ls[i], 0, line) for i in range(len(ls) - 1)],
                                  columns=['sequence', 'a', 'b', 'direction', 'line'])
            return pd.concat([df, df_sym])

        if add_symmetric:
            sym = links.copy()
            sym['a'], sym['b'] = links['b'], links['a']
            sym = sym[sym['a'] != sym['b']]
            links = pd.concat([links, sym])

            sequences = pd.concat([sequence_dataframe(line)
                                   for line in set(links['line'])])
            links = pd.merge(links, sequences, on=['a', 'b', 'line'])
            links['dir_line'] = links.apply(lambda r: str(r['line']) if r[
                                            'direction'] else str(r['line']) + '_bis', axis=1)

        link_integer_columns = ['a', 'b', 'line',
                                'headway', 'capacity', 'modeId', 'maxTraffic']
        links[link_integer_columns] = links[link_integer_columns].astype(int)
        links['geometry'] = spatial.linestring_geometry(
            links, nodes['geometry'].to_dict(), 'a', 'b')

        links['length'] = links['geometry'].apply(lambda g: g.length)

        return links, nodes

    def get_project(self, project_name):
        project_id = self.get_project_id(project_name)
        url = self.api_projects + '/' + str(project_id)
        return json.loads(requests.get(url).text)

    def links_nodes_zones(
        self,
        project_name,
        scenario_name,
        include_style=False,
        include_items=False
    ):

        project_id = self.get_project_id(project_name)
        url = self.api_projects + '/' + str(project_id)
        res = json.loads(requests.get(url).text)

        scenarios = res['scenarios']
        networks = res['networks']

        scenario = [s for s in scenarios if s['name'] == scenario_name][0]
        network = [n for n in networks if n['id'] == scenario['network_id']][0]

        zonal_data_id = scenario['zonal_data_id']

        url = self.api_zonaldata + '/' + str(zonal_data_id)

        r = requests.get(url)
        features = json.loads(r.text)['geojson']['features']

        properties = [feature['properties'] for feature in features]
        geometries = [shape(feature['geometry']) for feature in features]

        zones = pd.DataFrame(properties)
        zones['geometry'] = pd.Series(geometries)

        links, nodes = self.graph_from_network(
            network,
            include_items=include_items
        )

        links['length'] = skims.distance_from_geometry(links['geometry'])
        links['speed'] = links['speed'].fillna(np.sqrt(links['length']))
        links['time'] = links['length'] / 1000 / links['speed'] * 3600

        area_factor = skims.area_factor(zones['geometry'])

        zones['area'] = zones['geometry'].apply(
            lambda g: g.area * area_factor)

        zones['pop'] = zones['area'] * zones['pop_dens'] / 1000000
        zones['emp'] = zones['area'] * zones['emp_dens'] / 1000000

        if include_style:
            zones['geometry'] = zones['geometry'].apply(
                spatial.buffer_until_polygon
            )
            nodes['width'] = 3
            links['width'] = 5
            nodes['color'] = 'gray'

        return links, nodes, zones
