import gym, numpy, random, scipy
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K_backend
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import load_model


class arbol_binario_suma: 
# Esto es como el Cerebro, osea la forma en que la memoria funciona
# Esta basado en sumatorios de Arboles binarios - ideal para tomar muestras de la memoria sin tener que ordenar todos los recuerdos en un vector
# fuente = http://algorithms.tutorialhorizon.com/convert-binary-tree-to-its-sum-tree/)
    contador = 0
    def __init__(self, storage):
        self.storage = storage
        self.data = numpy.zeros(storage, dtype=object)
        self.arbol = numpy.zeros(2 * storage - 1)

    def propagate(self, indice, update):
        padre = (indice - 1) // 2
        self.arbol[padre] += update
        if padre != 0:
            self.propagate(padre, update)

    def retrieve(self, indice, Y):
        left = 2 * indice + 1
        right = left + 1
        if left >= len(self.arbol):
            return indice
        if Y <= self.arbol[left]:
            return self.retrieve(left, Y)
        else:
            return self.retrieve(right, Y - self.arbol[left])

    def get(self, Y):
        indice = self.retrieve(0, Y)
        data_indice = indice - self.storage + 1
        return (indice, self.arbol[indice], self.data[data_indice])

    def add(self, priority, data):
        indice = self.contador + self.storage - 1
        self.data[self.contador] = data
        self.update(indice, priority)
        self.contador += 1
        if self.contador >= self.storage:
            self.contador = 0

    def update(self, indice, priority):
        update = priority - self.arbol[indice]
        self.arbol[indice] = priority
        self.propagate(indice, update)

    def total(self):
        return self.arbol[0]
        

