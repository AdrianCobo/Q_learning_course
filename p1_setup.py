# pip install gym
# pip install gym[classic_control] # install pygame
# env = gym.make(name) -> set up the game
# env.reset() -> reset the game
# env.step(Action) -> simulation go forward with that action

import gym
import time

env = gym.make("MountainCar-v0",render_mode="rgb_array")
print(env.action_space.n) # print the possibles actions per step
# everytime we do an action, will get the new state and a reward, if the enviroment is done/complete, and some extra info in some env
state = env.reset() # position in x axes and velocity = observation space

print(env.observation_space.high) # maximum values of the state space variables
print(env.observation_space.low) # minimum values of the state space variables

# Q learning create a table which tells the model for every possible value of the state space variables wich output it should return. For 
# a continuous function of 8 decimals that table can have a savage size so we will dicretize this funciton in groups of ranges of size 20 for each range.

DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

print(discrete_os_win_size) # tell us how large each bucket is (range incremets for each bucket)

done = True
while not done:
    action = 2 # always go right!
    new_state, reward, done,_,_ = env.step(action)
    # env.render()
    print(reward, new_state)

env.close()