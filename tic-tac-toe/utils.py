import numpy as np

def initialV_x(env, state_winner_triples):
    #initialize state values as follows
    V = np.zeros(env.num_states)
    for state,winner,ended in state_winner_triples:
        if ended:
            if winner == env.x:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V


def initialV_o(env,state_winner_triples):
    V = np.zeros(env.num_states)
    for state,winner,ended in state_winner_triples:
        if ended:
            if winner == env.o:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V


def play_game(p1, p2, env, draw = False):
    #loops until the game is over
    current_player = None
    while not env.game_over():
        #alternate between players
        #p1 always starts first
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1

