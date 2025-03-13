import geopandas as gpd
import pandas as pd
import json
from syspy.spatial.geometries import line_list_to_polyline

def shorter_links(links,board = 'pickup_type' ,alight ='drop_off_type',original_links = False):
    '''
    script to shorten the links of Emme that are keeping on the node to have Quetzal links with only alighting or boardings
    if original_links == True, we keep the emme links in liste in the new table'
    '''
    crs = links.crs
    links = links.copy()
    # Safeguard: Ensure key columns are present
    required_cols = [board, alight, 'a', 'b','geometry','trip_id','link_sequence','length','speed','time']
    missing_cols = [col for col in required_cols if col not in links.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    links = links.sort_values(['trip_id', 'link_sequence'])
    first_last = links.reset_index().groupby('trip_id')['index'].agg(['first', 'last'])
    links.loc[first_last['first'], board] = 0
    links.loc[first_last['first'], alight] = 0
    links.loc[first_last['last'], alight] = 0
    links.loc[first_last['last'], board] = 0
    links['stop'] = True
    links['next_pickup'] = links[board].shift(-1).fillna(0).astype(int)
    links['prev_drop_off'] = links[alight].shift(+1).fillna(0).astype(int)
    links['stop'] = True
    links.loc[(links[alight] != 0) & (links['next_pickup'] != 0), 'stop'] = False
    links.loc[(links[board] != 0) & (links['prev_drop_off'] != 0), 'stop'] = False
    links['cumsum'] = links['stop'].cumsum()
    links = links.drop(columns=['next_pickup', 'prev_drop_off'])
    #Aggregate of links and columns
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