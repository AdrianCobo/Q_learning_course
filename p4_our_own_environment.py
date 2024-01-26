import numpy as np # for array stuff and random
from PIL import Image # for creating visual of our env
import cv2 # for showing our visual live
import matplotlib.pyplot as plt # for graphing our mean rewards over
import pickle # to save/load Q-Tables
from matplotlib import style # to make pretty charts because it matters
import time # using this to keep track of our saved Q-Tables

style.use("ggplot")

# its surprising that this model works really weel because it can only move diagonally and the agent would be
# able to eat the food the half of the time but it have quite succes

# first section explanation
# after having our table we have to change: and comment them down
SHOW_EVERY = 1
epsilon = 0.0
start_q_table = "qtable-1701893253.pikle"

# or just for retraining:
#start_q_table = "qtable-1701893070.pikle" # it is not the same as before
#start_q_table = "qtable-1701893212.pikle" # it is not the same as before
#start_q_table = "qtable-1701893253.pikle" # it is not the same as before
# comment enemy.move() and food.move() during trainning process
# see the results with the first section explanation

SIZE = 10 # playable table size
# if you want a 20x20 you need 2.5 million episodes for trainning but each episode dont need too much time.

HM_EPISODES = 25000
MOVE_PENALTY = 1 # fell free to tinker with these!
ENEMY_PENALTY = 300 # fell free to tinker with these!
FOOD_REWARD = 25 # fell free to tinker with these!
# epsilon = 0.9 # randomness
EPS_DECAY = 0.9998 # Every episode will be epsilon * EPS_DECAY
# SHOW_EVERY = 3000 # how ofthen to play through env visually

# start_q_table = None # if we have a pickled Q table, we'll put the filename of it here.

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1 # player key in dict
FOOD_N = 2 # food key in dict
ENEMY_N = 3 # enemy key in dict

# the dict! Using just for colors
d = {1: (255,175,0), # blueish color
     2: (0, 255, 0), #green
     3: (0, 0, 255)} # red

class Blob: # the square players
    def __init__(self):
        # initiate the position randomly
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self): # for debuging we implement a string function
        return f"{self.x}, {self.y}"
    
    def __sub__(self,other):
        return(self.x - other.x, self.y - other.y)
    
    def action(self, choice): # move based on an action
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if choice == 0:
            self.move(x=1 , y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y = 1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):
        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1,2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # if we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1

# blob tests:
# player = Blob()
# food = Blob()
# enemy = Blob()

# print(player)
# print(food)
# print(player-food)
# player.move()
# print(player-food)
# player.action(2)
# print(player-food)

# create the Q-table

if start_q_table is None:
    # initialize the q-table
    q_table = {}
    for i in range(-SIZE+1, SIZE):
        for ii in range(-SIZE+1, SIZE):
            for iii in range(-SIZE+1,SIZE):
                for iiii in range(-SIZE+1, SIZE):
                    # (diferencia_comida_x, diferencia_comida_y),(diferencia_enemy_x, diferencia_enemigo_y)
                    q_table[((i, ii), (iii, iiii))] = [np.random.uniform(-5,0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)
# to look in our table:
# print(q_table[((-9,2),(3,9))])

# start iterating over episodes
episode_rewards = []

for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    # start the actual frames/steps of the episode:
    episode_reward = 0
    for i in range(200):
        obs = (player-food, player-enemy)
        # print(obs)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)

        # Take the action!
        player.action(action)
        # its better for trainning not to move this and just move it ater training
        # because the position of the player, food, and enemy is alwais being initialized to false
        #### MAYBE ### 
        enemy.move()
        food.move()
        ##############

        # rewarding
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        # work up our Q-table and Q-value information:
        new_obs = (player-food, player-enemy) # new observation
        max_future_q = np.max(q_table[new_obs]) # max Q value for this new obs
        current_q = q_table[obs][action] # current Q for our chosen action

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1- LEARNING_RATE)* current_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q)
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
            env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue
            env[enemy.x][enemy.y] = d[ENEMY_N]  # sets the enemy location to red
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward

        # and of game
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

            # finis our per-episode loop:
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

# graph and save things:

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pikle", "wb") as f:
    pickle.dump(q_table,f)