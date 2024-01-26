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

#pip install keras tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import numpy as np
import tensorflow as tf

import time


# Para dotar de memoria al modelo vamos a crear un array con 50mil jugadas con información de (observation space, action, reward, new observation space, done).
# De estas 50mil jugadas seleccionaremos 1000 aleatorias para que se entrene el modelo por ese tamaño batch. Si lo actualizasemos 1 vez por jugada y no una vez
# por lote, los valores de las redes fluctuarian muchisimo por eso es mejor esperar y usar 1000 jugadas por funcion fit. Ademas que tardaría muchisimo tiempo mas.
# una vez que llegamos a minimo mil jugadas y las tenemos almacenada empezamos ya a usar mil de manera aleatoria para entrenar en cada iteración 
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MODEL_NAME = "256X2"

# Esta clase evita que por cada .predict que hagamos se registre en un ficher datos (esto lo hace queras). Para nuestro caso
# no nos es util ya que ejecuatamos muchismas veces esta funcion para entrenar a nuestro modelo (1 vez por cada posible accion que evaluamos)
# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class DQNAgent:
    def __init__(self):

        # main model that gets trained every step
        self.model = self.create_model()

        # Target model that is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0


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
    def get_qs(self, state, step):
        return self.model.predict(np.array(state).reshape(-1,*state.shape)/255)[0]
    
    # Trains main network every step during episode
    def trian(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return