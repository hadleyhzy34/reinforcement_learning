import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)
state_space = env.observation_space.shape[0]


# SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Actor(nn.Module):
    def __init__(self, state_space, lr=0.0001, **kwargs):
        super().__init__()
        self.affine = nn.Linear(state_space, 128)

        # actor's layer
        self.action_head = nn.Linear(128,2)

        # optimizer
        # self.optimizer = optim.Adam(self.parameters(), lr=3e-2)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        '''
        forward of both actor
        '''
        x = F.relu(self.affine(x))

        # actor: choses action to take from state_space by returning probability of each action 
        action_prob = F.softmax(self.action_head(x), dim=-1)

        return action_prob
    
    def select_action(self, state):
        # import ipdb;ipdb.set_trace()
        state = torch.from_numpy(state).float()
        actions_prob = self(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(actions_prob)

        # and sample an action based on sample distribution
        action = m.sample()

        # save to action buffer
        # model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        # the action to take (left or right)
        return action.item(), m.log_prob(action)

    def learn(self, action_prob, td_error):
        loss = -action_prob * td_error.detach()
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return


class Critic(nn.Module):
    def __init__(self, state_space, lr=0.0001, **kwargs):
        super().__init__()
        self.affine = nn.Linear(state_space, 128)

        #critic's layer
        self.value_head = nn.Linear(128,1)

        # optimizer
        # self.optimizer = optim.Adam(self.parameters(), lr=3e-2)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        '''
        forward of both critic
        '''
        x = F.relu(self.affine(x))

        #critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        return state_values

    def learn(self, s, r, s_):
        s = torch.from_numpy(s).float()
        s_ = torch.from_numpy(s_).float()
        with torch.no_grad():
            v_ = self(s_)

        # import ipdb;ipdb.set_trace()
        # td_error = r + args.gamma * v(s') - v(s)
        # td_error = torch.mean((r + args.gamma * v_) - self(s))
        td_error = r + args.gamma * v_ - self(s)

        loss =  td_error.square()
        # loss =F.smooth_l1_loss(r+args.gamma * v_, self(s))
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return td_error

eps = np.finfo(np.float32).eps.item()


def main():
    running_reward = 10
    actor = Actor(state_space)
    critic = Critic(state_space)

    # run inifinitely many episodes
    for i_episode in count(1):
        # import ipdb;ipdb.set_trace()
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            pre_state = state
            action, actions_prob = actor.select_action(state)

            # take the action
            state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            # if done:
            #     print(f'current reward is: {reward}')
            #     ep_reward -= 20
            
            # # backpropogation for w: value function weights update, s,r,s_
            # td_error = critic.learn(pre_state, reward, state)

            # # backpropgation for theta: policy distribution update: action_prob, td_error
            # actor.learn(actions_prob, td_error)

            if done:
                reward = -20

            td_error = critic.learn(pre_state, reward, state)
            actor.learn(actions_prob, td_error)

            # model.rewards.append(reward)
            ep_reward += reward

            if done:
                # ep_reward -= 20
                print(f'failed steps: {t}')
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # # perform backprop
        # finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()