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

