# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm

from syspy.spatial import spatial
from syspy.io.pandasshp import pandasshp
from syspy.io.pandasdbf import pandasdbf

def head_string(links, trip_id):
    return  'LINE NAME="%s", ONEWAY=T, ' % trip_id

def stop_and_node_string(links, trip_id, road=True):
    if 'road_node_list' not in links.columns:
        road=False
    line = links.loc[links['trip_id'] == trip_id]
    if road:
        try:
            dicts = []
            for nodes in list(line['road_node_list']):
                dicts.append({'nodes': nodes[1:-1], 'stop': nodes[-1]})

            s = 'N=%s' %( str(line['road_a'].iloc[0]))

            for chunk in dicts:
                for node in chunk['nodes']:
                    s += ', ' +  '-' + str(node)
                s += ', ' + str(chunk['stop']) 
        except IndexError: # road_node_list is empty
            return stop_and_node_string(links, trip_id, road=False)
    else:
        line = links.loc[links['trip_id'] == trip_id]
        s = 'N=%s' %( str(line['a'].iloc[0]))
        for b in line['b']:
            s += ', %s'%b

    return s


def lin_string(links, trip_id, custom_head=head_string):
    return custom_head(links, trip_id) +  stop_and_node_string(links, trip_id)


class cubeModel():
    def __init__(self):
        pass

    def to_lin(
        self, 
        path_or_buf, 
        custom_head=head_string, 
        separate=False,
        route_type=None,
        **kwargs
        ):

        if separate:
            route_types = set(self.links.route_type)
            for route_type in route_types:
                route_type_path = path_or_buf + r'/route_type_%s.lin' % str(route_type)
                self.to_lin(
                    route_type_path, 
                    custom_head=custom_head,
                    separate=False,
                    route_type=route_type
                )

        else:
            links = self.links.loc[self.links['a'] != self.links['b']].copy()
            if route_type is not None:
                links = links.loc[links['route_type'] == route_type]
            
            if len(links) > 0:
                lines = sorted(list(set(links['trip_id'])))
                lin = ''
                for trip_id in tqdm(lines):
                    lin += lin_string(links, trip_id, custom_head)
                    lin += ' \n'
                    
                with open(path_or_buf, 'w') as file:
                    file.write(lin)
                
        

    def to_net(
        self, 
        folder, 
        zone_index_to_int=int, 
        node_index_to_int=int,
        keep_link_columns=[],
        keep_node_columns=[],
        keep_zone_columns=[],
        ntleg_type=10,
        footpath_type=10,
        separate=True,
        road_network=True,
        ):
        """
        dump all the files that you will need to build a network in the folder
        """
        self.check_link_references() # no orphan node
        road_links = self.road_links.copy() if road_network else pd.DataFrame()
        road_nodes = self.road_nodes.copy() if road_network else pd.DataFrame()
        pt_links = self.links.copy()
        road_links['type'] = 0
        pt_links['type'] = pt_links['route_type']
        links = pd.concat([road_links, pt_links])
        

        zones = spatial.add_geometry_coordinates(
            self.zones, columns=['x', 'y'])

        zones[['x', 'y']] = np.round(zones[['x', 'y']], 6)
        nodes = pd.concat([road_nodes, self.nodes.copy()])

        nodes = spatial.add_geometry_coordinates(nodes, columns=['x', 'y'])

        nodes[['x', 'y']] = np.round(nodes[['x', 'y']], 6)

        zones['n'] = zones.index
        nodes['n'] = nodes.index

        zones['n'] = zones['n'].apply(zone_index_to_int)
        nodes['n'] = nodes['n'].apply(node_index_to_int)
        links['a'] = links['a'].apply(node_index_to_int)
        links['b'] = links['b'].apply(node_index_to_int)

        nodes = nodes.drop_duplicates(subset = ['n'])

        nodes = nodes[['x', 'y', 'n', 'geometry'] + keep_node_columns]
        zones = zones[['x', 'y', 'n', 'geometry'] + keep_zone_columns]

        ntlegs = self.zone_to_road
        access = ntlegs.loc[ntlegs['direction'] == 'access'].copy()
        eggress = ntlegs.loc[ntlegs['direction'] == 'eggress'].copy()

        access['a'] = access['a'].apply(zone_index_to_int)
        eggress['b'] = eggress['b'].apply(zone_index_to_int)

        access['b'] = access['b'].apply(node_index_to_int)
        eggress['a'] = eggress['a'].apply(node_index_to_int)
        
        ntlegs = pd.concat([access, eggress])

        # footpaths
        footpaths = self.footpaths.copy()
        footpaths['type'] = footpath_type
        footpaths['a'] = footpaths['a'].apply(node_index_to_int)
        footpaths['b'] = footpaths['b'].apply(node_index_to_int)

        ntlegs['distance'] = np.round(ntlegs['length'])
        links['distance'] = np.round(links['length'])
        footpaths['distance'] = np.round(footpaths['length'])

        ntlegs['type'] = ntleg_type
        ntlegs = ntlegs[['a', 'b', 'distance', 'geometry', 'type'] + keep_link_columns]
        links = links[['a', 'b', 'distance', 'geometry', 'type'] + keep_link_columns]
        footpaths = footpaths[['a', 'b', 'distance', 'geometry', 'type'] + keep_link_columns]

        links.drop_duplicates(subset=['a', 'b', 'type'], inplace=True)


        road_links = links[links['type'] == 0]
        pt_links = links[links['type'] > 0].copy()

        if road_network:
            pandasshp.write_shp(folder + 'road_links.shp', road_links, re_write=True)
        
        pandasshp.write_shp(folder +'nodes.shp', nodes, re_write=True)
        pandasshp.write_shp(folder +'zones.shp', zones, re_write=True)
        pandasshp.write_shp(folder +'ntlegs.shp', ntlegs, re_write=True)
        pandasshp.write_shp(folder +'footpaths.shp', footpaths, re_write=True)

        pandasshp.write_shp(folder + 'pt_links.shp', pt_links, re_write=True)
        if separate:
            for pt_type in set(pt_links['type']):
                type_links = pt_links.loc[pt_links['type'] == pt_type].copy()
                pandasshp.write_shp(
                    folder + 'pt_links_route_type_%s.shp' % str(pt_type), 
                    type_links,
                    re_write=True
                )


    def to_mat(self, folder, zone_index_to_int=int, volume_column='volume_pt'):
        volumes = pd.merge(
            self.pt_los[['origin', 'destination']], 
            self.volumes, on=['origin', 'destination'], 
            how='left').fillna(0)

    
        for c in ['origin', 'destination']:
            volumes[c] = volumes[c].apply(zone_index_to_int)
            
        volumes.sort_values(['origin', 'destination'], inplace=True)
        volumes.columns=['O', 'D', 'TRIPS']

        pandasdbf.write_dbf(volumes, folder + 'od.dbf' )

    def to_cube(
        self,
        folder, 
        custom_head=head_string,
        zone_index_to_int=int, 
        node_index_to_int=int
        ):

        self.to_lin(folder + 'lines.lin', custom_head=custom_head)

        self.to_net(
            folder = folder, 
            zone_index_to_int=zone_index_to_int, 
            node_index_to_int=node_index_to_int
        )

        self.to_mat(folder=folder, zone_index_to_int=zone_index_to_int)

