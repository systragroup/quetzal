from numba import jit
import numpy as np


# vdf and vdfd (derivative)

# Eval strings

free_flow = 'time'

default_bpr = 'time * (1 + {alpha} * (flow/capacity)**{beta})'.format(alpha=0.15, beta=4)

# Python functions


def limit_factory(expression: str, limit: float = None):
    # add a limit time*limit to an expression function.
    # if limit is None. use limit column. else: its a constant (df[time_col] * limit)
    def wrapped(df, flow_col, time_col):
        expr = expression.replace('flow', flow_col).replace('time', time_col)
        res = df.eval(expr)
        if limit == None:
            return res.clip(upper=df[time_col] * df['limit'])
        else:
            return res.clip(upper=df[time_col] * limit)

    return wrapped


limited_bpr = limit_factory(default_bpr)


# Numba functions


@jit(nopython=True)
def jit_limited_bpr(mat):
    # columns in mat : 'alpha','beta','limit','flow','time','penalty','capacity'
    # der return the first derivative (for the find beta...)
    jam_time = []
    for i in range(mat.shape[0]):
        alpha = mat[i, 0]
        beta = mat[i, 1]
        limit = mat[i, 2]
        V = mat[i, 3]
        t0 = mat[i, 4]
        penalty = mat[i, 5]
        Q = mat[i, 6]
        res = t0 * (1 + alpha * np.power(V / Q, beta))
        if res > t0 * limit:  # we plateau the curve at limit.
            jam_time.append(t0 * limit + penalty)
        else:
            jam_time.append(res + penalty)
    return jam_time


@jit(nopython=True)
def jit_default_bpr(mat):
    # columns in mat : 'alpha','beta','limit','flow','time','penalty','capacity'
    # der return the first derivative (for the find beta...)
    jam_time = []
    for i in range(mat.shape[0]):
        alpha = mat[i, 0]
        beta = mat[i, 1]
        V = mat[i, 3]
        t0 = mat[i, 4]
        penalty = mat[i, 5]
        Q = mat[i, 6]
        V0 = mat[i, 7]
        V += V0
        res = t0 * (1 + alpha * np.power(V / Q, beta))
        jam_time.append(res + penalty)
    return jam_time


@jit(nopython=True)
def jit_free_flow(mat):
    # columns in mat : 'alpha','beta','limit','flow','time','penalty','capacity'
    # der return the derivative (for the find beta...)
    t0 = mat[:, 4]
    penalty = mat[:, 5]
    return t0 + penalty

    # columns in mat : 'alpha','beta','limit','flow','time','penalty','capacity'
    # der return the derivative (for the find beta...)
    t0 = mat[:, 4]
    return t0 * 0
