# importante: deepqlearning tardaría hora s lo que Qlearinng podria aprender en minutos ya que hay que hacer
# predicciones constantemente de cada posible accion en un estado y ajustar los pesos del modelo constantemente pero
# Qlearning se queda corto enseguida debido a que solo vale para problemas simples, en cuanto se complica el problema 
# se requiere de demasiada memoria para la tabla q y ademas hay que discretizar el espacio, cosa que para DeepQlearning no hace 
# falta y se hace un uso muchisimo mas eficiente de la memoria pudiendo usarse incluso capas convolucionales para espacios graficos.
# A diferencia de Qlearingn con DeepQlearning se entrena el modelo para todas las posibles acciones del estado y no solo la que toma el agente
# esto pasa ya que el modelo predice valores de la tabla q para cada accion de ese estado. Lo que se podría hacer es crear un modelo por estad
# y esto no pasaría pero no se suele hacer

# Muchas veces lo que se hace es entrenar 2 modelos, 1 que se entrena constantemente y otro que nos da los valores de la tabla q
# y se actualiza de vez en cueando.

# construyamos la clase de nuestro agente para el mismo juego del caso anterior.
# IMPORTANTE!!!!!!!!!!!!!!!!!!!
# recuerda poner el minimo para guardar el modelo segun el minimo del mejor modelo que tengas guardado en la carpeta models
# To Do: usar la media de funcion de perdidad del modelo en lugar de minimo ya que siempre que se come al enemigo son -300 puntos de penalización pero claro,
# el modelo sería menos seguro

#pip install keras tensorflow
import numpy as np
#import keras.backend.tensorflow_backend as backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import numpy as np
import tensorflow as tf

import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

#python3 -m pip install tensorflow[and-cuda] # for using tensorflow and cuda with gpu
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Cargamos la info del mundo de BLOB visto en el p4.py y creamos la clase del entorno de BLOB (BlobEnv)
class Blob: # the square players
    def __init__(self, size):
        # initiate the position randomly
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self): # for debuging we implement a string function
        return f"BLOB ({self.x}, {self.y})"
    
    def __sub__(self,other):
        return(self.x - other.x, self.y - other.y)
    
    def __eq__(self, other):
        return (self.x == other.x and self.y == other.y) # True si 2 BLOB en la misma casilla
    
    def action(self, choice): # move based on an action
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8) # testing with 8
        '''
        collision = False
        if choice == 0:
            collision = self.move(x=1 , y=1)
        elif choice == 1:
            collision = self.move(x=-1, y=-1)
        elif choice == 2:
            collision = self.move(x=-1, y=1)
        elif choice == 3:
            collision = self.move(x=1, y=-1)

        elif choice == 4:
            collision = self.move(x=1 , y=0)
        elif choice == 5:
            collision = self.move(x=-1, y=0)
        elif choice == 6:
            collision = self.move(x=0, y=1)
        elif choice == 7:
            collision = self.move(x=0, y=-1)

        #elif choice == 8:
        #    self.move(x=0, y=0)
        return collision

    def move(self, x=False, y=False):
        collision = False
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
            collision = True
        elif self.x > self.size-1:
            self.x = self.size-1
            collision = True
        
        if self.y < 0:
            self.y = 0
            collision = True
        elif self.y > self.size-1:
            self.y = self.size-1
            collision = True
        return collision


class BlobEnv: # the square players
    SIZE = 10 # playable table size
    RETURN_IMAGES = True
    MOVE_PENALTY = 1 # fell free to tinker with these!
    ENEMY_PENALTY = 200 # fell free to tinker with these!
    COLLISION_PENALTY = 200
    FOOD_REWARD = 500 # fell free to tinker with these!
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3) # mapa de 10x10x3canales
    # testing with 8 no queremos que se quede parado
    ACTION_SPACE_SIZE = 8 # 9 posibles movimientos ahora (arriba, abajo,etc y los diagonales así como quieto)
    PLAYER_N = 1 # player key in dict
    FOOD_N = 2 # food key in dict
    ENEMY_N = 3 # enemy key in dict

    # the dict! Using just for colors
    d = {1: (255,175,0), # blueish color
        2: (0, 255, 0), #green
        3: (0, 0, 255)} # red

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player: # si la comida y muñeco estan en la misma posición
            self.food = Blob(self.SIZE) # creamos otro blob (para que no empiecen en la misma posición)
        self.enemy = Blob(self.SIZE)
        
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE) # hacemos que el enemigo no inicie ni en la casilla de la comida ni del jugador

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:# habria que revisar este caso ya que no tiene sentido que se pudiese tratar igual una imagen que una lista de distancias
            observation = (self.player-self.food) + (self.player-self.enemy) # [distancia comida(x,y), distancia enemigo(x,y)]
        return observation
    
    def step(self, action):
        self.episode_step += 1
        collision = self.player.action(action)

        #### MAYBE ####
        # enemy.move()
        # food.move()
        ###############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.food) + (self.player - self.enemy)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        elif collision:
            reward = -self.COLLISION_PENALTY
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or reward == -self.COLLISION_PENALTY or self.episode_step >= 20:
            done = True
        
        return new_observation, reward, done
    
    def render(self):
        img = self.get_image()
        img = img.resize((800,800)) # resizing so we can see our agent
        cv2.imshow("Simulation: Master AI", np.array(img)) # show it!
        cv2.waitKey(200) # pon el valor a 0 si quieres verlo realmente pero tarda bastante en entrenar los episodios requeridos para visualizar (cada 50 ep)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8) # # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N] # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N] # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N] # sets the player tile to blue
        img = Image.fromarray(env, 'RGB') # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img
    
env = BlobEnv()

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

MODEL_NAME = "256X2"

# Environment settings
EPISODES = 200 # "like epochs"

SHOW_PREVIEW = True

class DQNAgent:
    def __init__(self,model="None"):
        if model == "None":
            # Main model
            self.model = self.create_model()
        else:
            self.model = load_model(model)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1,*state.shape)/255)[0]

model_already_created = "models/modelo_legendario_256X2___500.00max__355.90avg_-200.00min__1706264767.model"
agent = DQNAgent(model_already_created)

print(" Presiona cualquier boton para comenzar la simulación")
init_img = np.zeros((10, 10, 3), dtype=np.uint8)
init_img = Image.fromarray(init_img, 'RGB')
init_img = init_img.resize((800,800)) # resizing so we can see our agent
cv2.imshow("Simulation: Master AI", np.array(init_img))
cv2.waitKey() # espera a pulsar un boton para simular
# hacemos que el agente participe con el entorno
for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episodes'): # chars use ASCII and calling the "unit" an "episode"

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # ITERATE OVER THE STEPS PER EPISODE:
    # reset flag and start iterating until episode ends
    done = False
    while not done:

        action = np.argmax(agent.get_qs(current_state)) # la accion con mas valor/recompensa
        print(action)
        
        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        env.render()

        current_state = new_state
        step +=1
