import math
import random
from collections import defaultdict
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()
# import tensorflow as tf


class Player(object):
    
    """ Player class details """
    
    counter = 0         # used to count instances
    player_id = -1

    def __init__(self, pop_size, action="C", reward=0.0, fitness=0.0, strategyType= "random") :
        """ Create a new Player with action, reward, fitness """
        self.__action = action
        self.__reward = reward
        self.__rounds = 0
        self.__fitness = fitness
        self.__nextAction = None
        self.__strategyType = strategyType
        # self.__state = [0] * 60
        self.__state = []
        type(self).counter += 1    
        self.__uniqueId = type(self).counter - 1  # id start from 0

    def __str__(self) :
         """ toString() """
         return  str(self.__uniqueId) + ": (" + str(self.__action) + "," \
         + str(self.__reward) + "," + str(self.__fitness)  + ")"
        
    def get_nextAction(self):
        return self.__nextAction
    
    def set_strategyType(self, new_strategy):
        self.__strategyType = new_strategy
        #self.__strInstance.set_strategyType(new_strategy)

    def get_strategyType(self):
        return self.__strategyType
    
    def set_fitness(self, new_fitness) :
        self.__fitness = self.__fitness + new_fitness
        
    def get_fitness(self) :
        return self.__fitness
        
    def set_rounds(self, rounds):
        self.__rounds = rounds
         
    def set_reward(self, new_reward) :
        self.__reward = new_reward
        
    def get_reward(self) :
        return self.__reward
    
    def set_state(self, new_state):
        # history infomation ------------
        state = self.__state
        n = len(new_state)
        state = new_state + state[:-n]
        self.__state = state
    
        # self.__state = new_state

    def get_state(self):
        return self.__state
    
    def get_uniqueId(self):
        return self.__uniqueId

             
# Every time the play ground change the real action of player
# it needs to generate a new possiable next action for use 
    def set_action(self, new_action) :
        self.__action = new_action
        #self.__strInstance.set_currentAction(new_action)
        #self.set_nextAction()
        
    def get_action(self) :
        return self.__action

    @classmethod
    def PlayerInstances(cls) :
        return cls, Player.counter

class QLearningPlayer(Player):
    actions = ["C", "D"]
    learning_rate = 0.01
    discount_factor = 0.9
    epsilon = 0.1
    q_table = defaultdict(lambda: [0.0, 0.0])

    def learn(self, s, a, s_, r):
        learning_rate = type(self).learning_rate
        discount_factor = type(self).discount_factor
        q_table = type(self).q_table

        old_q = q_table[s][a]
        new_q = r + discount_factor * max(q_table[s_])
        q_table[s][a] += learning_rate * (new_q - old_q)
        type(self).q_table = q_table

    def choose_action(self, s_):
        actions = type(self).actions
        epsilon = type(self).epsilon
        q_table = type(self).q_table

        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            q_value = q_table[s_]
            if q_value[0] > q_value[1]:
                action = actions[0]
            elif q_value[0] < q_value[1]:
                action = actions[1]
            else:
                action = random.choice(actions)
        return action


# Player use actor-critic reinforcement learning
# Hyper-parameters
GAMMA = 0.9

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=10,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            l2 = tf.layers.dense(
                inputs=l1,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )

            l3 = tf.layers.dense(
                inputs=l2,
                units=5,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l3'
            ) 

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                # activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # probability of each action
        # # underflow
        p = probs.ravel()

        if np.isnan(p[0]):
            return "NaN"
        else:
            return np.random.choice(np.arange(probs.shape[1]), p=p)
        
        # return np.random.choice(np.arange(probs.shape[1]), p=p)



        # converge? /  0: C  1: D
        # if p[0] < 0.01:
        #     return True, np.random.choice(np.arange(probs.shape[1]), p=p)
        # else:
        #     return False, np.random.choice(np.arange(probs.shape[1]), p=p)

        # if np.isnan(p[0]):
        #     if random.random() <= 0.5:
        #         return 0
        #     else:
        #         return 1
        # return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=10,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            l2= tf.layers.dense(
                inputs=l1,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )

            l3 = tf.layers.dense(
                inputs=l2,
                units=5,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l3'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        # print("s ", s)
        # print("s_ ", s_)

        v_ = self.sess.run(self.v, {self.s: s_})
        # print("v_", v_)
        # print("v", v)

        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        # print("r", r)
        # print("td_error", td_error)
        return td_error