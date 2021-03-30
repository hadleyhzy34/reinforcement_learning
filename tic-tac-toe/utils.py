import numpy as np

#for each state, assign value 0,0.5,1 to value function array, winner/lose/unknown:1,0,0.5
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

        #draw the board before the user who wants to see it makes a move
        if draw:
            if draw == 1 and current_player == p1:
                env.draw_board()
            if draw == 2 and current_player == p2:
                env.draw_board()

        #current player make a move
        current_player.take_action(env)

        #update state history
        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)

    if draw:
        env.draw_board()

    #do the value function update
    p1.update(env)
    p2.update(env)

#for each state of board, check its corresponding state, if its ended and who the winner is
def get_state_hash_and_winner(env, i=0, j=0):
  results = []

  for v in (0, env.x, env.o):
    env.board[i,j] = v # if empty board it should already be 0
    if j == 2:
      # j goes back to 0, increase i, unless i = 2, then we are done
      if i == 2:
        # the board is full, collect results and return
        state = env.get_state()
        ended = env.game_over(force_recalculate=True)
        winner = env.winner
        results.append((state, winner, ended))
      else:
        results += get_state_hash_and_winner(env, i + 1, 0)
    else:
      # increment j, i stays the same
      results += get_state_hash_and_winner(env, i, j + 1)

  return results
