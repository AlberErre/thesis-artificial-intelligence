import gym, numpy, random, scipy
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K_backend
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import load_model

""" IMPORTANTISIMO:
El Tamano de la pantalla es la clave, cuando mas pequeno mas generaliza y mas rapido procesa la CPU
screen__alto = 210/X    cuanto mayor sea X --> mayor generalizacion y mayor velocidad de procesamiento
screen_ancho = 160/X    Sin embargo, si la X es mayor a (), la red convolucional no funciona, ese es el limite
"""
# screen parameters:
screen__alto = int(210/5)  		
screen_ancho = int(160/5)  		

#### MAIN HYPER-PARAMETERS ####
memory_size = 20000 			 
batch_size = 32 				# Trozos de la memoria que se cogen para tomar decisiones
DoubleQ_update_frequency = 10000 		# Update the targets (predictions), para el doubleQ learning
learning_rate = 0.5 			 	# For the convolutional model, this model is used to make the predictions of the Q values
gamma = 0.99 					# If this is low, it takes a LOT of time to finish 100 iterations - the higher the better
epsilon = 0.6   				# e-greedy policy (exploracion!) - higher value, more probability to explore
epsilon_decay = 0.99   			 	# Speed of decay --> "epsilon_decay = 1" means NO DECAY (infinite exploration)

class Brain:
    def __init__(self, state_space, actions_space):
        self.state_space = state_space
        self.actions_space = actions_space
        self.model = self.Convolutional_model()
        self.DoubleQ_model = self.Convolutional_model()  # Second network (target) for DoubleQ learning

    def Convolutional_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4,4), activation='relu', input_shape=(self.state_space), data_format='channels_first'))
        model.add(Conv2D(64, (4, 4), strides=(2,2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=actions_space, activation='linear'))
        opt = RMSprop(lr=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=opt)

        return model

    def train(self, real_value, predicted_value, epochs=1, verbose=0):
        self.model.fit(real_value, predicted_value, batch_size=32, epochs=epochs, verbose=verbose)

    def predict(self, Q, DoubleQ=False): # DoubleQ = false/true, dependiendo de la clase Agent
        if DoubleQ:
            return self.DoubleQ_model.predict(Q)
        else:
            return self.model.predict(Q)

    def Max_Q_target(self, Q, DoubleQ=False):
        return self.predict(Q.reshape(1, 2, screen__alto, screen_ancho), DoubleQ).flatten()

    def update_DoubleQLearning(self):
        self.DoubleQ_model.set_weights(self.model.get_weights())

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
        
class Memory:       
    non_zero   = 0.0001    # Evita que la funcion sea cero - nonzero function!
    exponente  = 0.9  	   # Cuanto mas grande mejores Q values (bueno) - cuanto mas pequeno mas tienden a cero los Q values (malo)

    def __init__(self, storage):
        self.cerebro = arbol_binario_suma(storage)

    def priority_while_remember(self, error):
        return (error + self.non_zero) ** self.exponente # proportional priority

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

class Primitive_Agent: 
    memory = Memory(memory_size)
    exp = 0

    def __init__(self, actions_space):
        self.actions_space = actions_space

    def pickAction(self, state): # This agent selects actions randomly, just to fill the memory
        return random.randint(0, self.actions_space - 1)

    def save_in_memory(self, sample):  # as "(state, action, reward, next_state)" format
        error = abs(sample[2]) 
        self.memory.create_new_memory(error, sample)
        self.exp += 1

    def remember_from_memory(self): # This agent does not remember anything
        pass 

class Agent:
    num_iteraciones_inEpisode = 0

    def __init__(self, state_space, actions_space):
        self.state_space = state_space
        self.actions_space = actions_space
        self.brain = Brain(state_space, actions_space)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
    def pickAction(self, this_state):
        # Take random action with "epsilon" probability
    	if random.random() < self.epsilon:
            return random.randint(0, self.actions_space - 1)
        else:
            return numpy.argmax(self.brain.Max_Q_target(this_state)) # SIEMPRE Selecciona la Q max

    def save_in_memory(self, sample):  # as "(state, action, reward, next_state)" format
        real_value, predicted_value, error = self.evaluation([(0, sample)])
        self.memory.create_new_memory(error[0], sample)
        self.num_iteraciones_inEpisode += 1
        self.epsilon = self.epsilon * self.epsilon_decay # epsilon decay

        # Update el DoubleQ learning value cada X numero de iteraciones (usando su modulo %)
        if self.num_iteraciones_inEpisode % DoubleQ_update_frequency == 0:
            self.brain.update_DoubleQLearning()

    def remember_from_memory(self):    
        memory_batch = self.memory.take_random_bunch_of_memory(batch_size)
        real_value, predicted_value, error = self.evaluation(memory_batch)
        # update error/priority
        for i in range(len(memory_batch)):
            index = memory_batch[i][0]
            self.memory.update(index, error[i])

        self.brain.train(real_value, predicted_value)

    def evaluation(self, memory_batch): 
        # Evaluate possible scenarios given a batch of memory
        end_state = numpy.zeros(self.state_space)
        states = numpy.array([M[1][0] for M in memory_batch])
        final_states = numpy.array([(end_state if M[1][3] is None else M[1][3]) for M in memory_batch])
        prediction = agent.brain.predict(states)

        # Double Q Learning, to avoid overestimations in Q 
        prediction_originalQLearning = agent.brain.predict(final_states, DoubleQ=False)
        prediction_DoubleQLearning = agent.brain.predict(final_states, DoubleQ=True)

        real_value = numpy.zeros((len(memory_batch), 2, screen__alto, screen_ancho))
        predicted_value = numpy.zeros((len(memory_batch), self.actions_space))
        error = numpy.zeros(len(memory_batch))
        
        for i in range(len(memory_batch)):
            M = memory_batch[i][1]
            state =      M[0]
            action =     M[1]
            reward =     M[2] 
            next_state = M[3]
            target =     prediction[i]
            init_value = target[action]

            if next_state is None: 
                target[action] = reward # Si no hay siguiente estado, solo tomamos en cuenta la reward
            else:
                Qmax_value = prediction_DoubleQLearning[i][ numpy.argmax(prediction_originalQLearning[i]) ]
                target[action] = reward + gamma * Qmax_value # overwrite "init_value"
                Q_obtained = target[action]
                print("Q value obtained: " + str(Q_obtained))
                # Dependiendo de cada cuanto actualizemos el valor del DoubleQ, cambia el Qmax_value
                # obtendremos un Qmax_value mayor si actualizamos mas veces (en teoria)
                                        
            real_value[i] = state
            predicted_value[i] = target

            error[i] = abs(init_value - target[action])

        return (real_value, predicted_value, error)

def Color2Gray(color_image): # Como en computer vision, usamos RGB para cambiar la imagen a tono de grises y poder procesarla
	RGB = scipy.misc.imresize(color_image, (screen__alto, screen_ancho), interp='bilinear')
	R = RGB[:,:,0]
	G = RGB[:,:,1]
	B = RGB[:,:,2]
	gray_color = (0.299*R) + (0.587*G) + (0.114*B)
	gray_image = gray_color.astype('float32')/(128-1) # Normalizar como en computer vision, sin esto todos los estados parecen el mismo y no obtiene buenos Q values
	#plt.imshow(gray_image, cmap='gray')
	#plt.show()
	#print(type(gray_image)) # Debe ser un numpy.ndarray
	#print(gray_image.shape) # Debe ser un vector de 2 dimensiones, sin la tercera columna de los colores
	#print(RGB.shape) # Este Debe ser un vector de 3 dimensiones, porque tiene COLOR
	return gray_image 

class Play: # This is the ENVIRONMENT
    def __init__(self, game):
        self.game = game
        self.env = gym.make(game)

    def run(self, agent):                
        image = self.env.reset()
        gray_image = Color2Gray(image) # Needed for convolutional network
        state = numpy.array([gray_image, gray_image]) # state = last two gray pixel images

        total_reward = 0
        num_iteraciones = 0 # Steps
        while True:        
            #self.env.render()
            action = agent.pickAction(state) # pick an action given current state
            image, reward, done, info = self.env.step(action)
            next_state = numpy.array([state[1], Color2Gray(image)]) # numpy.array[last screen, new screen]
            if done: # if terminal state appears
                next_state = None
            agent.save_in_memory((state, action, reward, next_state)) # here the agent learns and save this information in memory
            agent.remember_from_memory()
            num_iteraciones = num_iteraciones + 1 # Para contar los "steps" para saber cada cuando actualizar el DoubleQ learning        
            state = next_state
            total_reward = total_reward + reward
            if agent is not Primitive_Agent: # Print the Q values with the second agent only
                print("reward obtained by taken action " + str(action) + " is: " + str(reward))
            if done:
                break # finish episode

        return total_reward, num_iteraciones

#-------------------- SAME GAME ----------------------------
"""
GAME = 'DemonAttack-v0' # Assault-v0 da muy buenos resultados
atari = Play(GAME)

state_space  = (2, screen__alto, screen_ancho)
actions_space = atari.env.action_space.n
agent = Agent(state_space, actions_space)
Primitive_Agent = Primitive_Agent(actions_space)

# Plots variables
x_axis = numpy.array(range(100)) # to print the results! - must be equal than "num_episodios"
cumulated_reward_vector = numpy.ndarray(0)
MAXReward_perEpisode = numpy.ndarray(0)
iterations_taken_per_episode = numpy.ndarray(0)

try:
    print("Iniciando ATARI con agente aleatorio (agente primitivo)... ")
    while Primitive_Agent.exp < memory_size:
        primitive_total_reward = atari.run(Primitive_Agent)[0]
        print("Experiencia obtenida: " + str(Primitive_Agent.exp) + " Capacidad total: " + str(memory_size))
        print("Reward total obtenida: " + str(primitive_total_reward))
    agent.memory = Primitive_Agent.memory
    Primitive_Agent = None # borramos el agente que creamos (primitive)

    print("Iniciando ATARI con agente con memoria (learning)... ")
    episodio = 1
    num_episodios = 100 # numero episodios que juega - must be equal than x_axis
    cumulated_sum = 0
    lista_de_cum_rewards = []
    while episodio <= num_episodios:
        episodio_total_reward = atari.run(agent)[0]
        num_iteraciones_perGame = atari.run(agent)[1]
        print("Jugando episodio " + str(episodio) + "... ")
        print("Reward total obtenida: " + str(episodio_total_reward))
        print("Numero de iteraciones en este episodio: " + str(num_iteraciones_perGame))
        cumulated_reward_vector = numpy.append(cumulated_reward_vector,[int(episodio_total_reward)])
        lista_de_cum_rewards.append(int(episodio_total_reward))
        MAXReward_perEpisode = numpy.append(MAXReward_perEpisode,[int(max(lista_de_cum_rewards))])
        iterations_taken_per_episode = numpy.append(iterations_taken_per_episode,[int(num_iteraciones_perGame)])
        cumulated_sum = cumulated_sum + episodio_total_reward
        episodio = episodio + 1

finally:
    print("juego finalizado")
    average_reward_per_episode = cumulated_sum / num_episodios
    print("average reward per episode: " + str(average_reward_per_episode))
    agent.brain.model.save_weights("atariWeights_10episodes_LR-01_gamma-01_epsilon-01_doublefreq-10000_memosize-20000_batch-32.h5")
    agent.brain.model.save("atariModel_10episodes_LR-01_gamma-01_epsilon-01_doublefreq-10000_memosize-20000_batch-32.h5")
        # Guardamos modelo y weights de la CNN - osea los patrones importantes del juego que luego exportaremos
    del agent.brain.model  # deletes the existing model
    print("Modelo guardado")

    # PLOTS
    plt.style.use('ggplot')
    plt.plot(x_axis, numpy.transpose(MAXReward_perEpisode),'r', label="Max Reward obtained") # plot max value reached over time
    plt.plot(x_axis, numpy.transpose(cumulated_reward_vector),'b', label="Reward (Score)")
    plt.plot(x_axis, numpy.transpose(iterations_taken_per_episode), 'k', label= 'Number iterations to solve Game') # plot max value reached over time
    plt.xlabel("Number of episodes (Games)")
    plt.ylabel("Score per episode - Reward")
    plt.title("LearningRate: " + str(learning_rate) + " Epsilon: " + str(epsilon) + " DoubleQ update freq: " + str(DoubleQ_update_frequency) + "\n Memory size: " + str(memory_size) + " Batch size: " + str(batch_size) + " Gamma" + str(gamma))
    plt.legend()
    plt.show()


"""
#-------------------- EXPORTABLE GAME ----------------------------

GAME = 'DemonAttack-v0'
EXPORTABLE_GAME = 'Assault-v0' #juego que vamos a utilizar para el primitive agent y luego exportarlo

atari = Play(GAME)
atari_exportable = Play(EXPORTABLE_GAME)

state_space  = (2, screen__alto, screen_ancho)
actions_space = atari.env.action_space.n
exportable_actions_space = atari_exportable.env.action_space.n
agent = Agent(state_space, actions_space)
Primitive_Agent = Primitive_Agent(exportable_actions_space)

# Plots variables
x_axis = numpy.array(range(5)) # to print the results! - must be equal than "num_episodios"
cumulated_reward_vector = numpy.ndarray(0)
MAXReward_perEpisode = numpy.ndarray(0)
iterations_taken_per_episode = numpy.ndarray(0)

try:
    print("Iniciando ATARI con agente aleatorio (primitivo)... ")
    while Primitive_Agent.exp < memory_size:
        primitive_total_reward = atari_exportable.run(Primitive_Agent)[0]
        print("Experiencia obtenida: " + str(Primitive_Agent.exp) + " Capacidad total: " + str(memory_size))
        print("Reward total obtenida: " + str(primitive_total_reward))
    agent.memory = Primitive_Agent.memory
    Primitive_Agent = None # borramos el agente que creamos (primitive)

    print("Iniciando ATARI con agente con memoria (learning)... ")
    episodio = 1
    num_episodios = 5 # numero episodios que juega - must be equal than x_axis
    cumulated_sum = 0
    lista_de_cum_rewards = []
    while episodio <= num_episodios:
        episodio_total_reward = atari.run(agent)[0]
        num_iteraciones_perGame = atari.run(agent)[1]
        print("Jugando episodio " + str(episodio) + "... ")
        print("Reward total obtenida: " + str(episodio_total_reward))
        print("Numero de iteraciones en este episodio: " + str(num_iteraciones_perGame))
        cumulated_reward_vector = numpy.append(cumulated_reward_vector,[int(episodio_total_reward)])
        lista_de_cum_rewards.append(int(episodio_total_reward))
        MAXReward_perEpisode = numpy.append(MAXReward_perEpisode,[int(max(lista_de_cum_rewards))])
        iterations_taken_per_episode = numpy.append(iterations_taken_per_episode,[int(num_iteraciones_perGame)])
        cumulated_sum = cumulated_sum + episodio_total_reward
        episodio = episodio + 1

finally:
    print("juego finalizado")
    average_reward_per_episode = cumulated_sum / num_episodios
    print("average reward per episode: " + str(average_reward_per_episode))
    agent.brain.model.save_weights("atariWeights_10episodes_LR-01_gamma-01_epsilon-01_doublefreq-10000_memosize-20000_batch-32.h5")
    agent.brain.model.save("atariModel_10episodes_LR-01_gamma-01_epsilon-01_doublefreq-10000_memosize-20000_batch-32.h5")
        # Guardamos modelo y weights de la CNN - osea los patrones importantes del juego que luego exportaremos
    del agent.brain.model  # deletes the existing model
    print("Modelo guardado")

    # PLOTS
    plt.style.use('ggplot')
    plt.plot(x_axis, numpy.transpose(MAXReward_perEpisode),'r', label="Max Reward obtained") # plot max value reached over time
    plt.plot(x_axis, numpy.transpose(cumulated_reward_vector),'b', label="Reward (Score)")
    plt.plot(x_axis, numpy.transpose(iterations_taken_per_episode), 'k', label= 'Number iterations to solve Game') # plot max value reached over time
    plt.xlabel("Number of episodes (Games)")
    plt.ylabel("Score per episode - Reward")
    plt.title("LearningRate: " + str(learning_rate) + " Epsilon: " + str(epsilon) + " DoubleQ update freq: " + str(DoubleQ_update_frequency) + "\n Memory size: " + str(memory_size) + " Batch size: " + str(batch_size) + " Gamma" + str(gamma))
    plt.legend()
    plt.show()
