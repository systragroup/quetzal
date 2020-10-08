# -*- coding: utf-8 -*-

from quetzal.model import analysismodel, docmodel, plotmodel

import warnings
from functools import wraps
import shutil
import ntpath
import uuid


def deprecated_method(method):
    @wraps(method)
    def decorated(self, *args, **kwargs):
        message = 'Deprecated: replaced by %s' % method.__name__
        warnings.warn(
            message,
            DeprecationWarning
        )
        print(message)
        return method(self, *args, **kwargs)

    decorated.__doc__ = 'deprecated! ' + str(decorated.__doc__)
    return decorated


def read_hdf(filepath, *args, **kwargs):
    m = StepModel(hdf_database=filepath, *args, **kwargs)
    return m

def read_zip(filepath, *args, **kwargs):
    try:
        m = StepModel(zip_database=filepath, *args, **kwargs)
        return m
    except : 
        # the zip is a zipped hdf and can not be decompressed
        return read_zipped_hdf(filepath, *args, **kwargs)

def read_zipped_hdf(filepath, *args, **kwargs):
    filedir = ntpath.dirname(filepath)
    tempdir = filedir + '/quetzal_temp' + '-' + str(uuid.uuid4())
    shutil.unpack_archive(filepath, tempdir)
    m = read_hdf(tempdir + r'/model.hdf', *args, **kwargs)
    shutil.rmtree(tempdir)
    return m


def read_json(folder):
    m = StepModel(json_folder=folder)
    return m


class StepModel(
    plotmodel.PlotModel,
    analysismodel.AnalysisModel, 
    docmodel.DocModel,
    ):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
 

# DEPRECATION

# deprecation method will be replaced by other data flow
StepModel.step_modal_split = deprecated_method(StepModel.step_modal_split)
StepModel.step_pathfinder = deprecated_method(StepModel.step_pathfinder)

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
