import gym, numpy, random, scipy
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K_backend
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import load_model


class Memory:       
    non_zero   = 0.0001    # Evita que la funcion sea cero - nonzero function!
    exponente  = 0.9  # Cuanto mas grande mejores Q values (bueno) - cuanto mas pequeno mas tienden a cero los Q values (malo)

    def __init__(self, storage):
        self.cerebro = arbol_binario_suma(storage)

    def priority_while_remember(self, error):
        return (error + self.non_zero) ** self.exponente # proportional priority
        ## NOTE: USE DIFFERENT CASES of PRIORITIZATION for the THESIS
        ## IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT !!!!!!
                ## IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT !!!!!!
        # como "Rank-based prioritization", o cosas asi Paper: PRIORITIZED EXPERIENCE REPLAY (2016)

    def create_new_memory(self, error, new_memory):
        priority = self.priority_while_remember(error)
        self.cerebro.add(priority, new_memory) 

    def take_random_bunch_of_memory(self, trozo_batch):
        batch = []
        trozo = self.cerebro.total()/trozo_batch
        for i in range(trozo_batch):
            x0 = trozo * i
            x1 = trozo * (i + 1)
            Y = random.uniform(x0, x1)
            (index, priority, data) = self.cerebro.get(Y)
            batch.append((index, data))
        return batch

    def update(self, index, error):
        priority = self.priority_while_remember(error)
        self.cerebro.update(index, priority)
