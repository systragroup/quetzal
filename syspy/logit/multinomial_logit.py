import numpy as np

def nest_probabilities(utilities, phi=1):
    print('test')
    exponential_df = np.exp(np.multiply(utilities, 1 / phi))
    exponential_s = exponential_df.T.sum()
    probability_df = exponential_df.apply(lambda s: s / exponential_s)
    return probability_df.T


def nest_utility(utilities, phi=1):
    exponential_df = np.exp(np.multiply(utilities, 1 / phi))
    exponential_s = exponential_df.T.sum()
    emu = np.log(exponential_s)
    composite_utility = phi * emu
    return composite_utility
