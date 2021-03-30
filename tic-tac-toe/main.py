import numpy as np
from environment import Environment
from agent import Agent
from human import Human
from utils import initialV_x,initialV_o,play_game,get_state_hash_and_winner


if __name__ == '__main__':
    #train the agent
    p1 = Agent()
    p2 = Agent()

    #set initial V for p1 and p2
    env = Environment(3)
    state_winner_triples = get_state_hash_and_winner(env)

    Vx = initialV_x(env, state_winner_triples)
    p1.setV(Vx)
    Vo = initialV_o(env, state_winner_triples)
    p2.setV(Vo)

    # give each player their symbol
    p1.set_symbol(env.x)
    p2.set_symbol(env.o)

    T = 10000
    for t in range(T):
        if t % 200 == 0:
            print(t)
        play_game(p1, p2, Environment(3))

    # play human vs. agent
    # do you think the agent learned to play the game well?
    human = Human()
    human.set_symbol(env.o)
    while True:
        p1.set_verbose(True)
        play_game(p1, human, Environment(3), draw=2)
        # I made the agent player 1 because I wanted to see if it would
        # select the center as its starting move. If you want the agent
        # to go second you can switch the human and AI.
        answer = input("Play again? [Y/n]: ")
        if answer and answer.lower()[0] == 'n':
            break
