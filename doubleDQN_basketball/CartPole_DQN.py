import sys
import gym
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import pdb
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import tensorflow as tf
import pylab

EPISODES = 30000
img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.t = 0
        self.max_Q = 0
        self.train = True
        self.render = True
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        if(self.train):
            self.epsilon = 1.0
        else:
            self.epsilon = 1e-6
        self.epsilon_decay = 0.999
        self.epsilon_min = 1e-6
        self.batch_size = 32
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=10000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()
        # copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()

    def build_model(self):
        print("Now we build the model")
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse',optimizer=adam)
        print("We finish building the model")
        return model

    def process_image(self, s_t):
        s_t = skimage.color.rgb2gray(s_t)
        s_t = skimage.transform.resize(s_t, (img_rows, img_cols))
        s_t = s_t / 255.0
        return s_t
        

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(s_t)
            return np.argmax(q_value[0])

    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)


        # load the saved model
        #w we do the experience replay
        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
        #pdb.set_trace()
        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)
        targets = self.model.predict(state_t)
        self.max_Q = np.max(targets[0])
        #The key point of Double DQN
        #selection of action is from model
        #update is from target model
        #target = self.model.predict(state_t)
        target_val = self.model.predict(state_t1)
        target_val_ = self.target_model.predict(state_t1)
        #pdb.set_trace()
        #a = np.argmax(self.model.predict(state_t1),axis=1)
        for i in range(batch_size):
            if terminal[i]:
                targets[i][action_t[i]] = reward_t[i]
            else:
                a = np.argmax(target_val[i])
                targets[i][action_t[i]] = reward_t[i] + self.discount_factor * (target_val_[i][a])

        #pdb.set_trace()
        #targets[range(batch_size), action_t] = reward_t + self.discount_factor*np.invert(terminal) * self.target_model.predict(state_t1)[a]
        #Q_sa = self.target_model.predict(state_t1)
        #targets[range(batch_size), action_t] = reward_t + self.discount_factor*np.max(Q_sa, axis=1)*np.invert(terminal)

        self.model.train_on_batch(state_t, targets)

    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)

#Here is the main loop
if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    #pdb.set_trace()
    # in case of CartPole-v1, you can play until 500 time step
    env = gym.make('DoubleDunk-v0')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    max_score = -999

    if (not agent.train):
        print("Now we load the saved model")
        agent.load_model("./save_model/Cartpole_DQN14.h5")



    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        #pdb.set_trace()
       
        x_t = agent.process_image(state)
        s_t = np.stack((x_t,x_t,x_t,x_t),axis=2)
        #In Keras, need to reshape
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2]) #1*80*80*4       
        
        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            x_t1 = agent.process_image(next_state)
            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)        

            # if an action make the episode end, then gives penalty of -100
            #reward = reward if not done or score == 499 else -100
            if (reward > 0):
                print("WOO, I GOT POSITIVE REWARD")


            # save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(s_t, action, reward, s_t1, done)
            # every time step do the training
            if (agent.train):
                agent.train_replay()
            score += reward
            print("score", score)
            s_t = s_t1
            agent.t = agent.t + 1
            print("EPISODE",  e, "TIMESTEP", agent.t,"/ ACTION", action, "/ REWARD", reward, "/ Q_MAX " , agent.max_Q)

            if done:
                env.reset()
                # every episode update the target model to be same with model
                agent.update_target_model()

                # every episode, plot the play time
                scores.append(score)
                episodes.append(e)
                agent.save_model("./save_model/Cartpole_DQN14.h5")
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/Cartpole_DQN14.png")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon)



 




































