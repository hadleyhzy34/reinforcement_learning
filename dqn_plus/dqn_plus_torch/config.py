import torch

#agent parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
TAU = 0.001
GAMMA = 0.99

#training parameters
#RAM_ENV_NAME = 'CartPole-v1'
RAM_ENV_NAME = 'LunarLander-v2'
VISUAL_ENV_NAME = 'Pong-v0'
CONSTANT = 90

#Training parameters
RAM_NUM_EPISODE = 1000
VISUAL_NUM_EPISODE = 3000
EPS_INIT = 1
EPS_DECAY = 0.995
EPS_MIN = 0.05
MAX_T = 1500
NUM_FRAME = 2


#replay buffer parameters
alpha = 0.5
epsilon = 0.05
td_init = 1
