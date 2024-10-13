import numpy as np
import pandas as pd

N_BINS = 20

position_bins = pd.cut([-1.2, 0.6], bins=N_BINS, retbins=True)[1][1:-1]
velocity_bins = pd.cut([-0.07, 0.07], bins=N_BINS, retbins=True)[1][1:-1]

def build_state(features):
    state_no = 0
    for i, feat in enumerate(features):
        state_no += (N_BINS ** i) * (feat - 1)
    return int(state_no)

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

def discretize(state):
    position, velocity = state
    return build_state([to_bin(position, position_bins),
                        to_bin(velocity, velocity_bins)])
