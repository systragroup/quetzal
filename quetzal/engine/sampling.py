import numpy as np
import pandas as pd

def resample(series, sample_weight=None, sample_size=None): 

    if sample_weight is None and sample_size is None :
        sample_weight = 1 

    # Build sample size and weight according to the other
    if sample_size :
        assert sample_weight is None
        sample_weight = series.sum() / sample_size
    elif sample_weight is not None:
        sample_size = series.sum() / sample_weight
        
    # make sure everything is consistant in the end
    sample_size = sample_size.astype(int)
    if sample_size == 0 :
        sample_size = 1

    sample_weight =  series.sum() / sample_size

    # resampleing
    probabilities = series / series.sum()
    
    np.random.seed(0)
    sample = np.random.choice(
        a=series.index, 
        size=sample_size,
        p=probabilities
    )

    sparse = pd.Series(sample).value_counts() * sample_weight
    if series.index.nlevels > 1 :
        sparse.index = pd.MultiIndex.from_tuples(sparse.index)
        sparse.index.names = series.index.names
    return sparse

def resample_square(square, sample_weight=1, sample_size=None):
    # remove origins and destinations with a null sum
    square = square.loc[square.sum(axis=1)>0, square.sum(axis=0)>0]
    
    destinations = square.apply(lambda d : resample(d, sample_weight=sample_weight, sample_size=sample_size))
    origins = square.apply(lambda o : resample(o, sample_weight=sample_weight, sample_size=sample_size), axis=1)

    origins.index.name = 'origin'
    origins.columns.name = 'destination'

    destinations.index.name = 'origin'
    destinations.columns.name = 'destination'
    o = origins.reindex(destinations.columns, axis=1)
    d = destinations.reindex(origins.index, axis=0)

    sparse = (o.fillna(0) + d.fillna(0)) / 2
    return sparse


def proportional_fitting(square, sum_axis_0=None, sum_axis_1=None, tolerance=1e-3, maxiter=20):

    temp = square.copy()
    c = np.inf

    for i in range(maxiter):
        if c<tolerance :
            print(i)
            return temp 
    
        sum_0 = temp.sum(axis=0)
        multiply_0 = sum_axis_0 / sum_0
        temp = temp * multiply_0

        sum_1 = temp.sum(axis=1)
        multiply_1 = sum_axis_1 / sum_1

        temp = (temp.T * multiply_1).T

        c = multiply_0.max() + multiply_1.max() -2

    print('cannot fit both axis')
    return square

def sample_od(od_indexed_series, bidimentional_sampling=True, fit_sums=True, sample_weight=1, sample_size=None):
    square = od_indexed_series.unstack()

    if bidimentional_sampling :
        sparse=resample_square(square, sample_weight=sample_weight, sample_size=sample_size)
    else : 
        sparse=resample(od_indexed_series, sample_weight=sample_weight, sample_size=sample_size).unstack()

        if fit_sums:
            sparse = proportional_fitting(sparse, sum_axis_0=square.sum(axis=0), sum_axis_1=square.sum(axis=1))

    return sparse.replace(0, np.nan).stack()


def get_average_block_length(pool, n_od, max_length=15, min_od=100, max_blocks=100):
    min_block_length = len(pool) / max_blocks
    
    if n_od <= min_od: 
        return 1
    else :
        to_return = min(n_od / len(pool), max_length)
        return max(to_return, min_block_length)

def get_od_blocks(od_pool, n_od=100, block_length=None, pop_origins=True, pop_destinations=True, max_destinations=100,seed=42):
    np.random.seed(seed)
    pool_origins = od_pool
    pool_destinations = od_pool
    
    od = []
    blocks = []
    remaining_od = n_od-len(od)
    i = 0
    while remaining_od>0:
        i+=1
        int_block_length = int(block_length + np.random.rand()) 
        # statistically equall to block_length

        n_origins = min(int_block_length, len(pool_origins))
        origins = np.random.choice(pool_origins, n_origins, replace=False)

        n_destinations = int(np.round(remaining_od/len(pool_origins)))
        n_destinations = min(n_destinations, len(pool_destinations))
        n_destinations = min(n_destinations, max_destinations)
        #print(n_destinations)
        destinations =  np.random.choice(pool_destinations, n_destinations, replace=False)

        block = []
        for o in origins: 
            for d in destinations :
                od.append((o, d))
                block.append((o, d))
        if len(block)>0:
            blocks.append(block)

        if pop_origins :
            pool_origins = [i for i in pool_origins if i not in origins]
            if len(pool_origins) == 0: # all the destinations have been covered
                pool_origins = od_pool # we re-initiate with the pool
                print('reset origins')
                
        if pop_destinations :
            pool_destinations = [i for i in pool_destinations if i not in destinations]
            if len(pool_destinations) == 0: # all the destinations have been covered
                pool_destinations = od_pool # we re-initiate with the pool
                print('reset destinations')
                
        remaining_od = n_od-len(od)
        
    return od, blocks