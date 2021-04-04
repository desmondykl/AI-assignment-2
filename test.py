import argparse

import random

from environment import TreasureCube
import numpy as np
import matplotlib.pyplot as plt
import pprint 
import pandas as pd

# you need to implement your agent based on one RL algorithm
class RandomAgent(object):
    def __init__(self):
        self.action_space = ['left','right','forward','backward','up','down'] # in TreasureCube
        self.Q = []

    def take_action(self, state):
        action = random.choice(self.action_space)
        return action

    # implement your train/update function to update self.V or self.Q
    # you should pass arguments to the train function
    def train(self, state, action, next_state, reward):
        pass
    
class Q_learnAgent(object):
    def __init__(self):
        self.action_space = ['left','right','forward','backward','up','down'] # in TreasureCube
        self.dim = 4
        self.explorationRate = 0.01
        self.discountFactor = 0.99
        self.learningRate = 0.5
        self.Q = {} 
        for z in range(self.dim):
            for x in range(self.dim):
                for y in range(self.dim):
                        allPossibleAction = {}
                        for a in self.action_space:
                            allPossibleAction[a] = 0
                        self.Q[str(z)+str(x)+str(y)] = allPossibleAction
        
    def print_Qtable(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 500)
        pd.set_option('display.max_colwidth', None)
        Qtable = pd.DataFrame.from_dict(self.Q, orient='index', columns=self.action_space)
        print(Qtable)
        Qtable.to_csv (r'export_dataframe.csv', index = False, header=True)

    def take_action(self, state):
        if random.random() < self.explorationRate:
            action = random.choice(self.action_space)
        else:
            Q_value_action = self.Q[state]
            maxKey = max(Q_value_action, key=Q_value_action.get)
            maxValue =Q_value_action[maxKey]
            BestAction = []
            for a in Q_value_action:
                if Q_value_action[a] == maxValue:
                    BestAction.append(a)
            action = np.random.choice(BestAction)
        return action

    # implement your train/update function to update self.V or self.Q
    # you should pass arguments to the train function
    def train(self, state, action, next_state, reward):
        #old estimation
        Q_old = self.Q[state][action]
        #new sample
        Q_old_next = self.Q[next_state]
        max_Q_old_next = Q_old_next[max(Q_old_next, key=Q_old_next.get)]
        #new estimation
        self.Q[state][action] = Q_old + self.learningRate*(reward + self.discountFactor*max_Q_old_next - Q_old)
        
def test_cube(max_episode, max_step):
    env = TreasureCube(max_step=max_step)
    agent = Q_learnAgent()

    reward_per_episode = []
    for epsisode_num in range(0, max_episode):
        state = env.reset()
        terminate = False
        t = 0
        episode_reward = 0
        while not terminate:
            action = agent.take_action(state)
            reward, terminate, next_state = env.step(action)
            episode_reward += reward
            # you can comment the following two lines, if the output is too much
            #env.render() # comment
            #print(f'step: {t} state:{state} action: {action}, reward: {reward}') # comment
            t += 1
            agent.train(state, action, next_state, reward)
            state = next_state
        reward_per_episode.append(episode_reward)
        print(f'epsisode: {epsisode_num}, total_steps: {t} episode reward: {episode_reward}')
    agent.print_Qtable()
    plt.title('Episode rewards vs Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Episode rewards')
    plt.plot(reward_per_episode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--max_episode', type=int, default=500)
    parser.add_argument('--max_step', type=int, default=500)
    args = parser.parse_args()

    test_cube(args.max_episode, args.max_step)
