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


