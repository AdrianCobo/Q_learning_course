# pip install gym
# pip install gym[classic_control] # install pygame
# env = gym.make(name) -> set up the game
# env.reset() -> reset the game
# env.step(Action) -> simulation go forward with that action

import matplotlib.pyplot as plt
import numpy as np
import gym
import time

env = gym.make("MountainCar-v0", render_mode="rgb_array")
# Q-Learning settings
LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 4000  # how many iterations of the game we want to play
# EPISODES = 10000
#EPISODES = 25000  
SHOW_EVERY = 3000
STATS_EVERY = 100

# convert from continuous state space to discrete state space
# DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
DISCRETE_OS_SIZE = [40] * len(env.observation_space.high) # for testing

discrete_os_win_size = (env.observation_space.high -
                        env.observation_space.low)/DISCRETE_OS_SIZE

# exploration settings
epsilon = 1  # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYIN = EPISODES//2
# END_EPSILON_DECAYIN = EPISODES # testing
# reduce epsilont untill we reach de EPISODES/2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYIN-START_EPSILON_DECAYING)

# lets create the q_table:
# size is going to be 20x20x3. 3 because we have 3 possible acctions per state
# the choice of being -2 to 0 is variable but we made all the values negatives because the reward at each step is -1
# and the flag is a 0 reward(when finish) so it have sense to be all negatives at the beginning
# for every 3 possible actions values we will see at the q_table what is the acction that have the most reward(explotacion) or random(exploracion)
# q_value = value per possible action per unique state
q_table = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# for stats:
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

def get_discrete_state(state):
    global discrete_os_win_size
    # discrete_os_win_size = np.asarray(discrete_os_win_size)
    if type(state) == tuple: # needed because  of gym types
        state = tuple(state[0])
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    # we use this tuple to look up the 3 Q values for the available actions in the q-table
    return (int(discrete_state[0]), int(discrete_state[1]))

state_pos = []

for episode in range(EPISODES):
    episode_reward = 0

    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode % 100 == 0:
        # save qtable for just use it then or continue trainning
        np.save(f"qtables/{episode}-qtable.npy", q_table)

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

        new_state, reward, done, _, _ = env.step(action)

        # save reward for episode reward stats
        episode_reward += reward

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
            new_q = (1 - LEARNING_RATE) * current_q + \
                LEARNING_RATE * (reward + DISCOUNT * max_future_q)

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

    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY: # el modulo te va a dar cero cuando sea episode = STATS_EVERY. 0 = False
        average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

    # plot robot position
    if episode % SHOW_EVERY == 0:
        fig = plt.figure()
        plt.scatter(state_pos, range(len(state_pos)),
                    label="train_acc", marker="x")
        plt.legend(loc=2)
        plt.show()

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.grid(True)
plt.show()
