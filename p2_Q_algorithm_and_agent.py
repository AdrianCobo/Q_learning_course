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

# lets create the q_table:

import numpy as np

# size is going to be 20x20x3. 3 because we have 3 possible acctions per state
# the choice of being -2 to 0 is variable but we made all the values negatives because the reward at each step is -1
# and the flag is a 0 reward(when finish) so it have sense to be all negatives at the beginning
# for every 3 possible actions values we will see at the q_table what is the acction that have the most reward(explotacion) or random(exploracion)
# q_value = value per possible action per unique state
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# qlearning formula:
# new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
# discount = measure how much we want to care about FUTURE reward rather than immediate reward. use to be High between 0 to 1. 0 = random always 1 = never explore
# max_future_q = we get it after having done the action and then we actualize our previous values using partialy the best next q_value
# con el tiempo, este valor de recompensa que hemos alcanzado se propaga lentamente hacia atras (basico pero funciona bastante bien)


import matplotlib.pyplot as plt

# Q-Learning settings
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000 # how many iterations of the game we want to play

# exploration settings
epsilon = 1 # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYIN = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYIN-START_EPSILON_DECAYING) # reduce epsilont untill we reach de EPISODES/2

# convert from continuous state space to discrete state space
def get_discrete_state(state):
    global discrete_os_win_size
    # discrete_os_win_size = np.asarray(discrete_os_win_size)
    if type(state) == tuple:
        state = tuple(state[0])
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return (int(discrete_state[0]), int(discrete_state[1])) # we use this tuple to look up the 3 Q values for the available actions in the q-table

SHOW_EVERY = 1000

state_pos = []

for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False
    print(episode)
    
    if episode % SHOW_EVERY == 0:
        state_pos = []
        plot = True
        print(episode)
    else:
        plot = False
    while not done:

        if np.random.random() > epsilon:
            # get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done,_,_ = env.step(action)

        new_discrete_state = get_discrete_state(new_state)
        
        if episode % SHOW_EVERY == 0:
            state_pos.append(new_discrete_state[0])
        # env.render()

        # If simulation did not end yet after last step - update Q table
        if not done:

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q

        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            # reward
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    # Decaying epsilon is being done every episode if episode number is within decaying range (episode 0-EPISODES/2)
    if END_EPSILON_DECAYIN >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    if episode % SHOW_EVERY == 0:
        fig = plt.figure()
        plt.scatter(state_pos,range(len(state_pos)), label="train_acc",marker="x")
        plt.legend(loc=2)
        plt.show()

env.close()