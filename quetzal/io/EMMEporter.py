import geopandas as gpd
import pandas as pd
import json
from syspy.spatial.geometries import line_list_to_polyline

def shorter_links(links,original_links = False, add_dict = {} ):
    '''
    script to shorten the links of Emme that are keeping on the node to have Quetzal links with only alighting or boardings
    if original_links == True, we keep the emme links in liste in the new table'
    Geometric_node_columns is the field that permit to filter geometric nodes that have buged pickup_type using other columns
    you will need to pass a list of two columns, first boarding then alighting

    '''
    crs = links.crs
    links = links.copy()
    # Safeguard: Ensure key columns are present
    required_cols = ['pickup_type', 'drop_off_type', 'a', 'b','geometry','trip_id','link_sequence','length','speed','time']
    missing_cols = [col for col in required_cols if col not in links.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    links = links.sort_values(['trip_id', 'link_sequence'])
    first_last = links.reset_index().groupby('trip_id')['index'].agg(['first', 'last'])
    links['stop'] = True
    links['next_pickup'] = links['pickup_type'].shift(-1).fillna(0).astype(int)
    links['prev_drop_off'] = links['drop_off_type'].shift(+1).fillna(0).astype(int)
    links.loc[(links['pickup_type'] != 0) & (links['prev_drop_off'] != 0), 'stop'] = False
    #if the one before is false it means that you can 
    links.loc[first_last['first'], 'stop'] = True
    #links.loc[first_last['last'],'stop'] = True
    links['cumsum'] = links['stop'].cumsum()
    links = links.drop(columns=['next_pickup', 'prev_drop_off'])
    #Aggregate of links and columns
    #ajouter dans parametre agg_dict() update mon agg_dict()
    agg_dict = {col: 'first' for col in links.columns}
    del agg_dict['cumsum']
    del agg_dict['stop']
    del agg_dict['speed']
    agg_dict['a'] = 'first'
    agg_dict['b'] = 'last'
    agg_dict['geometry'] = line_list_to_polyline
    agg_dict['road_link_list'] = lambda x: sum(x, [])
    agg_dict['time'] = sum
    agg_dict['length'] = sum
    agg_dict['drop_off_type'] = 'last'
    agg_dict.update(add_dict)
    if original_links is True:
        links['original_links'] = [*zip(links['a'], links['b'])]
        agg_dict['original_links'] = list
    if 'selectLink' in links.columns:
        agg_dict['selectLink'] = lambda x: 'yes' if 'yes' in set(x) else None
    for col in links.columns:
        print(col, ' --> ', agg_dict.get(col, '! not agg !'))
    links = links.reset_index().groupby('cumsum').agg(agg_dict)
    links['speed'] = 3.6 * links['length'] / links['time']
    # fix link_sequence
    lengths_sequence = links.groupby('trip_id')['a'].agg(len).values
    link_sequence = [i + 1 for l in lengths_sequence for i in range(l)]
    links['link_sequence'] = link_sequence
    links.index = 'link_' + links.index.astype(str)
    links.index.name = 'index'
    links = gpd.GeoDataFrame(links)
    links = links.set_crs(crs)
    
    return(links)