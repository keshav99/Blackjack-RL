import gym
from MCAgent import MCAgent
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('Blackjack-v1', natural=True, sab=True) # sab true for single agent blackjack
agent = MCAgent(eps=0.3, gamma=0.98)
num_episodes = 200000
wins = []
total_wins = 0

for i in range(1, num_episodes + 1):
    state = env.reset()[0] # reset the environment and get the initial state
    done = False

    while not done:
        action = agent.choose_action(state) # agent chooses action
        next_state, reward, done, _ = env.step(action)  # then the environment takes the action
        agent.episode.append((state, action, reward)) # the episode is appended with the state, action and reward
        state = next_state
    
    if reward > 0:
        total_wins += 1
    if i % 1000 == 0:
        wins.append(total_wins / i)
    agent.update_q() # update the Q values and policy after each episode
    if i % 20000 == 0:
        print(f"Episode {i}: Total Wins: {total_wins}")

plt.plot(range(0, len(wins) * 1000, 1000), wins)
plt.title('Win Rate vs Episodes')
plt.xlabel('Episodes')
plt.ylabel('Win Rate')
plt.grid()
plt.show()

