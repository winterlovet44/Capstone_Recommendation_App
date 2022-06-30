
import os
import pickle
import numpy as np
import pandas as pd
from scipy import sparse


DTYPE = np.float32


def check_os_environ():
    """Helper function for check environs configuration."""

    pass


def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
        f.close()
    return model

def save_model(model, path):
    with open(path, "wb") as f:
        model = pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

