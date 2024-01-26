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
# vamos a aumentar la recompensa y penalización por comerse la comida o chocar con el enemigo lo suficiente como para que sea rentable ir a comer
# y no chocarse ya que con 200 movimientos por episodio si la recomensa es baja muchas veces le sale mejor quedarse quieto que ir a por comida.
# Tambien vamos a considerar como caso penalizable chocarse contra el borde de tablero y fin de simulación y vamos a guardar el modelo por si acaso cada
# 200 episodios

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


import signal
import sys
# ctrl + c = save model
def signal_handler(sig, frame):
    agent.model.save(f'models/auto_saved_{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
    print('Model_saved!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

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
        img = img.resize((300,300)) # resizing so we can see our agent
        cv2.imshow("image", np.array(img)) # show it!
        cv2.waitKey(100) # pon el valor a 0 si quieres verlo realmente pero tarda bastante en entrenar los episodios requeridos para visualizar (cada 50 ep)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8) # # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N] # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N] # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N] # sets the player tile to blue
        img = Image.fromarray(env, 'RGB') # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img
    
env = BlobEnv()

# For stats
ep_rewards = [-20000]

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

# Para dotar de memoria al modelo vamos a crear un array con 50mil jugadas con información de (observation space, action, reward, new observation space, done).
# De estas 50mil jugadas seleccionaremos 1000 aleatorias para que se entrene el modelo por ese tamaño batch. Si lo actualizasemos 1 vez por jugada y no una vez
# por lote, los valores de las redes fluctuarian muchisimo por eso es mejor esperar y usar 1000 jugadas por funcion fit. Ademas que tardaría muchisimo tiempo mas.
# una vez que llegamos a minimo mil jugadas y las tenemos almacenada empezamos ya a usar mil de manera aleatoria para entrenar en cada iteración 
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 50  # How many steps (samples) to use for training of the REPLAY_MEMORY data
UPDATE_TARGET_EVERY = 5 # Terminal states (end of episodes). We pass the weights of the main model(fit every episode) to the predicctoins model(the one that 
# give us the q values) every 5 episodes simulated
MODEL_NAME = "256X2"
MIN_REWARD = -20000  # For model save
#MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20_000 # "like epochs"

# Exploration settings
#epsilon = 1  # not a constant, going to be decayed. Epsilon for new model
epsilon = 0.7
EPSILON_DECAY = 0.9975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = True

# Esta clase evita que por cada .predict que hagamos se registre en un ficher datos (esto lo hace queras). Para nuestro caso
# no nos es util ya que ejecuatamos muchismas veces esta funcion para entrenar a nuestro modelo (1 vez por cada posible accion que evaluamos)
# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
                self.writer.flush()

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

class DQNAgent:
    def __init__(self,model="None"):
        if model == "None":
            # Main model
            self.model = self.create_model()
        else:
            self.model = load_model(model)

        # Target model that is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0 # Variable de control para que el modelo de prediccion obtenga los mismos pesos que el modelo de entrenamiento cada x episodios/epochs

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3,3), input_shape=env.OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256,(3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2)) # se queda con el pixel vecino de mayor valor usando un kernel de 2x2
        model.add(Dropout(0.2))# apaga neuronas aleatorias para evitar overfitting

        model.add(Flatten()) # para que pasemos de un array en 2D a uno lineal que pueda usar la capa lineal
        model.add(Dense(64)) # capa neuronal fully connected de 64 neuronas
        model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear")) # capa de salida con neuronas = nuemero de acciones y funcion de activación lineal (para regresion)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model
    
    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1,*state.shape)/255)[0]
    
    # cada elemento de un minibatch esta compuesto de: (current_state, action, reward, new_current_state, done)
    # funcion trian: basicamente espera a que haya un minimo de jugasa para entrenar formando un batch,
    # despues,coge de ese almacen de jugadas un batch aleatorio y relaiza predicciones con el modelo que se 
    # entrena en cada episodio (una prediccion(valor q para cada posible accion) por cada estado actual que compone el batch) y 
    # realiza también una predicion para cada estado posterior a los que habiamos usado para entrenar al otro modelo por tanto 
    # tenemos los valores q segun el modelo que se entrena en cada episodio para el estado tx, y los valores q para los estados tx+1 
    # predichos por el modelo que se entrena cada x veces para que haya menos oscilaciones.

    # despues para cada item del minibatch seleccionamos el valor de q mas alto que ha predicho el modelo que se entrena cada x veces y lo usamos para
    # actualizar la recompensa. Esta recompensa modificada substituirá el valor de q para la accion realizada por el agente para pasar al estado de tx+1 y
    # entrenar al modelo que entrena en cada episodio con el estado actual como dato de entrada y qs obtenidas por ese mismo modelo pero con la q 
    # correspondiente a la accion realizada por el agente para ir al estado tx+1 modificada. Esto se hace con cada elemento del lote hasta completar un batch
    
    # Basicamente se entrena al modelo que entrena en cada iteracion con los estados del minibatch de tx como datos de X y como datos de y(salida) se usa
    # los valores de q obtenidos por este mismo modelo pero con la modificación del valor de q correspondiente a la acción que hace el agente en tx para pasar
    # a tx+1. Este valor se cambia al usarse la recompensa para esa acción tomada + (máximo valor de q predicho por el agente que se entrena cada X iteraciones 
    # en el estado tx+1)* tasa de descuento(variable empleada para la fase de exploración) si la simulación no ha acabo o solo el valor de la recompensa si el
    # simulación ha acabado en esa accion

    # Finalmente aumentamos el contador de episodios sin haber entrenado al modelo de las predicciones y si este contador llega a un valor 
    # establecido, entrenamos al modelo de las predicciones usando los pesos del modelo que se ha entrenado en cada episodio y ponemos este contador 
    # a 0

    # para entrenar cogemos un minibatch al hazar de todo el conjunto de datos guardados por que no queremos que se entrene solo en funcion del estado que
    # se simula, queremos que aprenda para todos los estados posibles del mundo en el que podría estar. Piensa que para un tablero de tamaño 10x10 solo
    # para una figura ya son 100 posibles estados por ende cogemos 40 al hazar de un banco de datos de 25000 para que entrene bien. Aunque parezca mucho,
    # no es extraño para este caso ya que piensa que tambien se toma una posible accion distinta (de las 8 que usamos) para cada estado y demás. Lo que
    # pasa que si que es verdad que tarda la vida en entrenar.

    # Trains main network every step during episode
    def trian(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        # Get a minibatch(a batch) of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)
        
        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)
        
        # with this q_values we are ready to update our model:

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here 
            if not done:
                max_feautre_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_feautre_q
            else:
                new_q = reward

            # update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our trainining data
            X.append(current_state)
            y.append(current_qs)
        
        # Fit on all samples as one batch and log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1
        
        # If counter reaches set value, update target network with wights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

model_already_created = "models/modelo_experto256X2___500.00max__428.08avg_-202.00min__1706257848.model"
agent = DQNAgent(model_already_created)

# hacemos que el agente participe con el entorno
for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episodes'): # chars use ASCII and calling the "unit" an "episode"
    # update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # ITERATE OVER THE STEPS PER EPISODE:
    # reset flag and start iterating until episode ends
    done = False
    while not done:
        # This part stays mostly the same as p4.py, the change is to query a model for Q values instead of qtable
        if np.random.random() > epsilon:
            # Get action from Q table (model)
            action = np.argmax(agent.get_qs(current_state)) # la accion con mas valor/recompensa
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)
        
        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state,action,reward, new_state, done))
        agent.trian(done, step)

        current_state = new_state
        step +=1
    
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
        cv2.destroyAllWindows()
        print(min_reward)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward > MIN_REWARD or not episode % 200:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
            MIN_REWARD = min_reward
            print("hola")

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
