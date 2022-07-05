import ntpath
import shutil
import uuid
from quetzal.model import analysismodel, docmodel, plotmodel
from quetzal.model.integritymodel import deprecated_method





def read_hdf(filepath, *args, **kwargs):
    m = StepModel(hdf_database=filepath, *args, **kwargs)
    return m


def read_zip(filepath, *args, **kwargs):
    try:
        m = StepModel(zip_database=filepath, *args, **kwargs)
        return m
    except Exception:
        # the zip is a zipped hdf and can not be decompressed
        return read_zipped_hdf(filepath, *args, **kwargs)


def read_zipped_hdf(filepath, *args, **kwargs):
    filedir = ntpath.dirname(filepath)
    tempdir = filedir + '/quetzal_temp' + '-' + str(uuid.uuid4())
    shutil.unpack_archive(filepath, tempdir)
    m = read_hdf(tempdir + r'/model.hdf', *args, **kwargs)
    shutil.rmtree(tempdir)
    return m


def read_json(folder, **kwargs):
    m = StepModel(json_folder=folder, **kwargs)
    return m


def read_zippedpickles(folder, *args, **kwarg):
    m = StepModel(zippedpickles_folder=folder, *args, **kwarg)
    return m


class StepModel(
    plotmodel.PlotModel,
    analysismodel.AnalysisModel,
    docmodel.DocModel,
):

    """Object StepModel : contains most of the transport model with : 
        * Attributes are the caracteristics of the model
        * Methods are the steps and functions of the model

    Attributes
    ----------
    zones : geodataframe
        Zoning system of the model.
        
        Main columns :
            area, population, geometry
    
    centroids : geodataframe
        Centroids of the zoning system of the model.
        Usually created by method preparation_ntlegs.
        
        Main columns :
            area, population, geometry
    
    segments : list
        Demand segments of the model.
        Created by the user.
    
    volumes : dataframe
    	Volumes per OD pair.
        Usually created by method step_distribution or user input.
        
        Main columns : 
            origin, destination, volume per demand segment
    
    epsg : string
    	Projection
    
    links : geodataframe
        Links of the public transport system and pt routes caracteristics.
        Each line of the geodataframe correspond to a section of a PT route between two nodes=stops ('a' and 'b').
        Usually created by shapefile importer or GTFS importer.
        
        Main columns : 
            * initial and final nodes of the link : 'a', 'b' 
            * caracteristics of the line : 'trip_id', 'route_id', 'service_id', 'trip_headsign', 'trip_short_name','direction_id', 'block_id','agency_id', 'route_short_name','route_long_name','route_type'
            * caracteristics of the section :'arrival_time', 'drop_off_type', 'time', 'headway', 'pattern_id','link_sequence', 'departure_time', 'pickup_type', 'geometry', 'length', 'duration', 'cost'
            * caracteristics of the roads sections which supports the link section (created by method preparation_cast_network) : 'road_a', 'road_b', 'road_node_list','road_link_list', 'road_length'
            * results of the step_assignment : 'volume', 'boardings', 'alightings'

    nodes: geodataframe
        Public transport stations.
        Usually created by shapefile importer or GTFS importer.
        
        Main columns : 
            geometry, name
    
    road_links: geodataframe
        Links (edges) of the road network - between two road_nodes.
        Usually created by shapefile importer or OSMNX.
        
        Main columns :
            * initial and final road_nodes of the link : 'a', 'b', 
            * caracteristics of the section : 'length', 'geometry', 'time', 'walk_time', 'capacity'
            * description of the jam function : 'vdf', 'alpha', 'beta', 'limit', 'penalty'
            * results of the step_road_pathfinder : 'flow', 'jam_time'

    road_nodes: geodataframe
        Nodes of the road network.
        Usually created by shapefile importer or OSMNX.
        
        Main columns : 
            geometry
    
    zone_to_road : geodataframe
        Connectors, aka non-transit leg (ntleg), from zones to road_nodes.
        Usually created by method preparation_ntlegs.
        
        Main columns : 
            * initial and final zone/road_nodes of the ntleg :'a', 'b'
            * description of the ntleg : 'rank', 'distance', 'geometry', 'direction'
            * description og the speed and time :'speed_factor', 'short_leg_speed', 'long_leg_speed', 'speed', 'time', 'walk_time'
    
    zone_to_transit : geodataframe
        Connectors, aka non-transit leg (ntleg), from zones to nodes (PT stations).
        Usually created by method preparation_ntlegs.
        
        Main columns : 
            * initial and final zone/nodes(stop) of the ntleg :'a', 'b'
            * description of the ntleg : 'rank', 'distance', 'geometry', 'direction'
            * description og the speed and time :'speed_factor', 'short_leg_speed', 'long_leg_speed', 'speed', 'time', 'walk_time'
    
    road_to_transit : geodataframe
        Connectors, aka non-transit leg (ntleg), from road_nodes to nodes (pt stations).
        Usually created by method preparation_ntlegs.
        
        Main columns : 
            * initial and final node/road_nodes of the ntleg :'a', 'b'
            * description of the ntleg : 'rank', 'distance', 'geometry', 'direction'
            * description og the speed and time :'speed_factor', 'short_leg_speed', 'long_leg_speed', 'speed', 'time', 'walk_time'
    
    footpaths : geodataframe
        Pedestrian links between stations to allow connections.
        Usually created by method preparation_footpaths.
        
        Main columns : 
            'a', 'b', 'geometry', 'time'
    
    pt_los : dataframe
        Level of service of the pt network - for each OD pair, possible paths and their caracteristics. Each line of the df being a path.
        Usually created by method step_pt_pathfinder.
        
        Main columns :
            * OD pair : 'origin', 'destination'
            * description of the path : 'gtime', 'path', 'pathfinder_session', 'reversed', 'all_walk', 'ntransfers', 'time_link_path', 'length_link_path' 
            * links taken by the path : 'link_path', 'footpaths', 'ntlegs',  'boarding_links', 'alighting_links'
            * nodes where transfer is made : 'boardings', 'alightings', 'transfers', 'node_path'
            * results of the step_logit : utilities (per demand segment), probabilities (per demand segment)
    
    car_los	: dataframe
        Level of service of the car network - for each OD pair results of pathfinder with/without capacity restriction.
        Usually created by method step_road_pathfinder.

    pr_los : dataframe
        Level of service of the park and ride network - for each OD pair, possible paths and their caracteristics. Each line of the df being a path.
        Usually created by method step_pr_pathfinder.
        
        Main columns :
            * OD pair : 'origin', 'destination'
            * description of the path : 'gtime', 'path', 'pathfinder_session', 'reversed', 'all_walk', 'ntransfers', 'time_link_path', 'length_link_path' 
            * links taken by the path : 'link_path', 'footpaths', 'ntlegs',  'boarding_links', 'alighting_links'
            * nodes where transfer is made : 'boardings', 'alightings', 'transfers', 'node_path'
            * results of the step_logit : utilities (per demand segment), probabilities (per demand segment)
        
        Main columns :
            * OD pair : 'origin', 'destination'
            * description of the path : 'time', 'path', 'gtime'
    
    los : dataframe
        Concatenation of the two tables pt_los and car_los to perform logit
    
    utility_values : dataframe
        Values of the utility parameters per mode per segment.
        Usually created by method preparation_logit - with parameters from the parameters file
    
    mode_utility : dataframe
        Modal constants per mode per segment.
        Usually created by method preparation_logit - with parameters from the parameters file
    
    mode_nests : dataframe
    	Structure of the nested logit.
        Usually created by method preparation_logit - with parameters from the parameters file
    
    logit_scales : dataframe
    	Parameter phi of the nested logit.
        Usually created by method preparation_logit - with parameters from the parameters file
    
    utilities : dataframe
    	Agregaded utilities per OD per mode.
        Usually created by method step_logit.
    
    probabilities : dataframe
    	Agregaded probabilities per OD per mode.
        Usually created by method step_logit.



    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# DEPRECATION
# FOR DOCUMENTATION : parametrize so that those functions do not appear ?


# deprecation method will be replaced by other data flow
StepModel.step_modal_split = deprecated_method(StepModel.step_modal_split)
StepModel.step_pathfinder = deprecated_method(StepModel.step_pt_pathfinder)

# moved to analysismodel
StepModel.checkpoints = deprecated_method(StepModel.analysis_checkpoints)
StepModel.step_desire = deprecated_method(StepModel.analysis_desire)
StepModel.linear_solver = deprecated_method(StepModel.analysis_linear_solver)
StepModel.step_analysis = deprecated_method(StepModel.analysis_summary)
StepModel.build_lines = deprecated_method(StepModel.analysis_lines)

# moved to preparationmodel
StepModel.step_footpaths = deprecated_method(StepModel.preparation_footpaths)
StepModel.step_ntlegs = deprecated_method(StepModel.preparation_ntlegs)
StepModel.step_cast_network = deprecated_method(
    StepModel.preparation_cast_network)
StepModel.renumber_nodes = deprecated_method(
    StepModel.preparation_clusterize_nodes)
StepModel.renumber = deprecated_method(StepModel.preparation_clusterize_zones)

# moved to integritymodel integrity_test
StepModel.assert_convex_road_digraph = deprecated_method(
    StepModel.integrity_test_isolated_roads)
StepModel.assert_lines_integrity = deprecated_method(
    StepModel.integrity_test_sequences)
StepModel.assert_no_circular_lines = deprecated_method(
    StepModel.integrity_test_circular_lines)
StepModel.assert_no_collision = deprecated_method(
    StepModel.integrity_test_collision)
StepModel.assert_no_dead_ends = deprecated_method(
    StepModel.integrity_test_dead_ends)
StepModel.assert_nodeset_consistency = deprecated_method(
    StepModel.integrity_test_nodeset_consistency)

# moved to integritymodel integrity_fix
StepModel.add_type_prefixes = deprecated_method(
    StepModel.integrity_fix_collision)
StepModel.get_lines_integrity = deprecated_method(
    StepModel.integrity_fix_sequences)
StepModel.get_no_circular_lines = deprecated_method(
    StepModel.integrity_fix_circular_lines)
StepModel.get_no_collision = deprecated_method(
    StepModel.integrity_fix_collision)
StepModel.clean_road_network = deprecated_method(
    StepModel.integrity_fix_road_network)

# renamed
