import numpy as np
import pandas as pd
from typing import Optional


def resample(square:pd.DataFrame,
            sample_weight:Optional[int]=None, 
            sample_size:Optional[int]=None, 
            seed:int=0, 
            axis:int=1)-> pd.DataFrame: 
    '''
    square : matrix (pd.Dataframe). return a Matrix in the same form.
    sample_weight: weight of each sample (if 10. each commute will represent 10 person's commute).
    sample_size: number of random samples in the distribution. (if 100, 100 random commutes will be selected)
    '''
    
    if sample_weight is None and sample_size is None :
        sample_weight = 1 
    if axis==0:
        square = square.T

    mat=np.zeros(square.values.shape)
    for i,arr in enumerate(square.values):
        sample = resample_arr(arr, sample_weight=sample_weight, sample_size=sample_size,seed=seed)
        mat[i,:] = sample

    mat = pd.DataFrame(mat,columns=square.columns,index=square.index)
    if axis==0:
        mat = mat.T

    return mat 

def resample_arr(arr:np.array,
                  sample_weight:Optional[int]=None, 
                  sample_size:Optional[int]=None, 
                  seed:int=0) -> np.array:
    '''
    takes an np.array, weight or size. return resamples array.
    sampling done randomly with the original array probability distribution.
    '''
    np.random.seed(seed)
    total = arr.sum()
    # Build sample size and weight according to the other
    if sample_size :
        assert sample_weight is None
        sample_weight = total / sample_size
    elif sample_weight is not None:
        sample_size = total / sample_weight

    # make sure everything is consistant in the end
    sample_size = int(sample_size)
    if sample_size == 0 :
        sample_size = 1

    weight =  total / sample_size

    # resampleing
    probabilities = arr / total

    sample = np.random.choice(
        a=[i for i in range(len(arr))], 
        size=sample_size,
        p=probabilities
    )
    res = np.zeros(len(arr))
    for v in sample:
        res[v] += weight
    
    return res

def resample_square(square, sample_weight=1, sample_size=None,**kwargs):
    # remove origins and destinations with a null sum
    square = square.fillna(0)
    square = square.loc[square.sum(axis=1)>0, square.sum(axis=0)>0]

    destinations = resample(square, sample_weight=sample_weight, sample_size=sample_size,axis=0,**kwargs)
    origins = resample(square, sample_weight=sample_weight, sample_size=sample_size,axis=1,**kwargs)

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

def sample_od(od_indexed_series, bidimentional_sampling=True, fit_sums=True, sample_weight=1, sample_size=None,**kwargs):
    square = od_indexed_series.unstack()

    if bidimentional_sampling :
        sparse=resample_square(square, sample_weight=sample_weight, sample_size=sample_size,**kwargs)
    else : 
        od_indexed_series[:] = resample_arr(od_indexed_series.values, sample_weight=sample_weight, sample_size=sample_size,**kwargs)
        sparse = od_indexed_series.unstack()
        
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

def get_od_blocks(od_pool, n_od=100, block_length=None, pop_origins=True, pop_destinations=True, max_destinations=100,seed=42, no_o_is_d=True):
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
        if no_o_is_d: # no origin is destination as the time is 0 and its a waste
            filtered_pool_destinations = [d for d in pool_destinations if d not in origins]
            destinations =  np.random.choice(filtered_pool_destinations, n_destinations, replace=False)
        else:
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