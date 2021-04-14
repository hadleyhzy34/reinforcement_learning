import tensorflow as tf
import numpy as np
from vizdoom import *

import random
import time
from skimage import transform

from collections import deque
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

def create_env():
    '''
    create our environment
    '''
    game = DoomGame()
    
    #load the correct configuration
    game.load_config("deadly_corridor.cfg")

    #load the correct scenario
    game.set_doom_scenario_path('deadly_corridor.wad')

    game.init()

    possible_actions = np.identity(7, dtype=int).tolist()

    return game, possible_actions


game, possible_actions = create_env()
