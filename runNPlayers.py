from Player import Player
from Player import QLearningPlayer
from Player import Actor
from Player import Critic
from collections import defaultdict
from random import shuffle
import random
import math
import csv
import os,errno
import os.path
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()
# import tensorflow as tf

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_processes = comm.Get_size()

class game(object):
    def __init__(self, pop_size, pop_update_rule, F, cost, M, group_size, mutation_rate):
        self.pop_size = pop_size
        self.pop_update_rule = pop_update_rule
        self.F = F
        self.cost = cost
        self.M = M
        self.group_size = group_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.cooperators = []
        self.defectors = []
        self.lst = []
        self.avReward = 0
        self.cooperator_count = 0
        self.defector_count = 0
        self.NaN = False
    
    def set_pop_update_rule(self, new_update_rule):
        self.pop_update_rule = new_update_rule

    def get_cooperators(self):
        return self.cooperators
        
    def get_defectors(self):
        return self.defectors

    def get_population(self):
        return self.population

    def csvDataBuilder(self, run_number, round):
        lst = self.lst
        lste = [run_number, self.pop_size, self.pop_update_rule, self.F, self.cost, self.M, self.group_size, self.mutation_rate,  self.cooperators[round], self.defectors[round], self.avReward, round]
        if self.NaN:
            lste[8] = "NaN"
            lste[9] = "NaN"
        lst.append(lste)

    def get_lst(self):
        return self.lst

    # calculate the average reward of each round.
    def average_reward(self):
        population = self.population
        pop_size = self.pop_size
        totalreward = 0
        for player in population:
            totalreward = totalreward + player.get_reward()
        self.avReward = totalreward/pop_size
    
    def cal_average_reward(self):
        population = self.population
        pop_size = self.pop_size
        totalreward = 0
        for player in population:
            totalreward = totalreward + player.get_reward()
        return totalreward/pop_size

    def init_population(self):
        population = []
        pop_size = self.pop_size
        pop_update_rule = self.pop_update_rule
        group_size = self.group_size
        cooperator_count = 0
        defector_count = 0
        for i in range(pop_size):
            # if random.random() <= 0.5:
            if i < pop_size / 2:
                action = "D"
                defector_count = defector_count + 1
            else:                                
                action = "C"
                cooperator_count = cooperator_count + 1
            
            if pop_update_rule == "q_learning" or pop_update_rule == "q_learning60":
                player = QLearningPlayer(pop_size,action,0,0,pop_update_rule)
            else:
                player = Player(pop_size,action,0,0,pop_update_rule)
            population.append(player)
        self.population = population
        self.cooperator_count = cooperator_count
        self.defector_count = defector_count
        self.cooperators.append(cooperator_count / pop_size)
        self.defectors.append(defector_count / pop_size)

    # Randomly ditribute players into different groups and play game
    def grouping_playing(self):
        population = self.population
        group_size = self.group_size
        pop_size = self.pop_size
        shuffle(population)

        i = 0
        while i < pop_size:
            if i + group_size < pop_size:
                group = population[i:i+group_size]
            else:
                group = population[i:pop_size]
            self.play_game(group)
            i = i + group_size
        
        # if pop_update_rule == "q_learning":
        #     self.average_reward()

    # Calculate the reward of players in a group
    def play_game(self, group):
        F = self.F
        cost = self.cost
        M = self.M
        pop_size = self.pop_size
        pop_update_rule = self.pop_update_rule
        updateRL = False
        updateID = -1

        group_size = len(group)
        cooperator_count = 0
        for i, player in enumerate(group):
            if player.get_action() == 'C':
                cooperator_count = cooperator_count + 1
            if player.get_uniqueId() == Player.player_id:
                updateRL = True
                updateID = i
        sigma = 0
        delta_count = cooperator_count - M
    
        if delta_count < 0:           # Heaviside function
            sigma = 0
        else:
            sigma = 1
    
        payD = (cooperator_count * F / group_size * cost) * sigma
        payC = payD - cost
        
        # state feature
        # C% in last group, avgR in last group, its action, its reward, pop C%, 
        f1 = cooperator_count / group_size
        f2 = (cooperator_count * payC + (group_size - cooperator_count) * payD) / group_size
        f5 = self.cooperator_count / pop_size
        f6 = self.cal_average_reward()
        

        # Update RL algorithms after observe focal player's next game state s'
        if updateRL:
            focal_player = group[updateID]
            s = focal_player.get_state()
            a = 0 if focal_player.get_action() == "C" else 1
            r = payC if a == 0 else payD
            # r = (cooperator_count * payC + (group_size - cooperator_count) * payD) / group_size

            f3 = a
            f4 = payC if f3 == 0 else payD
            s_ = [f1, f2, f3, f4, f5, f6]
            # s_ = [f1, f2, f3, f4, f5, f6] + s[:-6]
            # s_ = [f1]
            if pop_update_rule == "q_learning60":
                # s = s[0]
                # s_ = s_[0]
                s = " ".join(str(i) for i in s)
                s_ = " ".join(str(i) for i in s_)
                focal_player.learn(s, a, s_, r)    # Q table update
            elif pop_update_rule == "q_learning":
                # s = s[4]
                # s_ = s_[4]
                s = s[0]
                s_ = s_[0]
                focal_player.learn(s, a, s_, r)    # Q table update
            elif pop_update_rule == "actor_critic":
                # s = np.array(s[0:1])
                # s_ = np.array(s_[0:1])
                # s = np.array(s[4:])
                # s_ = np.array(s_[4:])
                s = np.array(s)
                s_ = np.array(s_)
                td_error = critic.learn(s, r, s_)  # Critic update
                actor.learn(s, a, td_error)        # Actor update

        for player in group:
            # set state  -----------------------------------------
            f3 = 0 if player.get_action() == "C" else 1
            f4 = payC if f3 == 0 else payD
            player.set_state([f1,f2,f3,f4,f5,f6])
            # player.set_state([f1])
            # ----------------------------------------------------
            if player.get_action() == 'C' :
                player.set_reward(payC)  # the reward in the current round
                player.set_fitness(payC) # the accumulted rewards
            else:
                player.set_reward(payD)
                player.set_fitness(payD)

    # Random update one player
    def update_population(self,round):
        pop_size = self.pop_size
        pop_update_rule = self.pop_update_rule
        population = self.population
        cooperator_count = self.cooperator_count
        defector_count = self.defector_count
        cooperators = self.cooperators
        defectors = self.defectors
        mutation_rate = self.mutation_rate
        
        i = random.randint(0,pop_size-1)
        Player.player_id = i
        focal_player = population[i]

        # update focal player using different update rules
        action_old = focal_player.get_action()
        if pop_update_rule == "random":
            j = i
            while j == i:
                j = random.randint(0,pop_size-1)
            player_j = population[j]
            action_new = player_j.get_action()
        
        elif pop_update_rule == "q_learning60":
            # The focal player doesn't know which group he will be distributed to,
            # use the historical (last round) game state to decide whether to cooperate or defect
            # s = focal_player.get_state()[0]
            s = focal_player.get_state()
            s = " ".join(str(i) for i in s)
            action_new = focal_player.choose_action(s)

        elif pop_update_rule == "q_learning":
            # s = focal_player.get_state()[4]
            s = focal_player.get_state()[0]
            action_new = focal_player.choose_action(s)
        
        elif pop_update_rule == "actor_critic":
            # The focal player doesn't know which group he will be distributed to,
            # use the historical (last round) game state to decide whether to cooperate or defect
            # s = np.array(focal_player.get_state()[0:1])
            # s = np.array(focal_player.get_state()[4:])

            s = np.array(focal_player.get_state())
            a = actor.choose_action(s)
            if a == "NaN":
                action_new = action_old
                self.NaN = True
            else:
                action_new = "C" if a == 0 else "D"
                self.NaN = False
            # action_new = "C" if a == 0 else "D"

            # if a == 0:
            #     action_new = "C"
            # elif a == 1:
            #     action_new = "D"
            # else:
            #     action_new = focal_player.get_action()
        else:
            assert(False)
        
        # mutation
        if random.random() < mutation_rate:
            if action_new == 'C':
                action_new = 'D'
            elif action_new == 'D':
                action_new = 'C'
            else:
                assert(False)
        # set the new action to player i
        focal_player.set_action(action_new)
            
        if action_new != action_old:
            if action_old == 'D':
                cooperator_count = cooperator_count + 1
                defector_count = defector_count - 1
            elif action_old == 'C':
                cooperator_count = cooperator_count - 1
                defector_count = defector_count + 1
            else:
                assert(False)

        cooperators.append(cooperator_count / pop_size)
        defectors.append(defector_count / pop_size)
        
        self.cooperators = cooperators
        self.defectors = defectors
        self.population = population
        self.cooperator_count = cooperator_count
        self.defector_count = defector_count      

def fileSaving(filename, data, writing_model):
    #i = 0
    with open(filename, writing_model) as f:
        f_csv=csv.writer(f, quotechar= "|", quoting = csv.QUOTE_MINIMAL)
        for line in data:
            f_csv.writerow(line)

# main function

if __name__ == "__main__":

    # pop_update_rule_list = ["random", "popular", "accumulated_best", "accumulated_worst", "accumulated_better", \
    #     "current_best", "current_worst", "current_better", "fermi", "q_learning", "actor_critic"]
    # pop_update_rule_list = ["random", "tf.reset_default_graph()", "q_learning"]
    pop_update_rule_list = ["actor_critic"]
    # pop_update_rule_list = ["q_learning"]
    
    # pop_size_list = [500, 1000, 1500, 2000]          # population size
    pop_size_list = [500]          # population size
    
    cost_list = [1]               # cost


    # g = [10, 50, 100, 250]
    group_size_list = [250]         # the size of group_size  value * pop_size
    # group_size_list = [0.01, 0.05, 0.1, 0.5, 1]         # the size of group

    
    # mutation_rate_list = [0.001]  # the probability of mutation
    mutation_rate_list = [0]
    
    # runs = 50                    # total number of independent runs
    runs = 50

    # tmax = 3000
    # time_list = range(0,3001,30)

    # tmax = 10000
    # time_list = range(0,10010,100)

    # tmax = 20000
    # time_list = range(0,20010,100)

    tmax = 40000
    time_list = range(0,40010,100)

    # write all data every 10th time_list step
    # (and not at first and last, already done specifically)
    #snapshot_time_list = [time_list[i] for i in range(1, len(time_list)-1, len(time_list)/10)]

    param_list = [] # list of parameter tuples: build list then run each
    param_count = 0
    for pop_size in pop_size_list:
        for pop_update_rule in pop_update_rule_list:
            for group_size in group_size_list:
                g = group_size
                F_list = [ g/8*i for i in range(9)]
                F_list[0] = 1
                M_list = F_list
                #-------------
                F_list = F_list[:6]
                #-------------
                # F_list = [250]
                # M_list = [125]
                for F in F_list:
                    for M in M_list:
                        for cost in cost_list:
                            for mutation_rate in mutation_rate_list:
                                for run_number in range(runs):
                                    param_count += 1
                                    
                                    param_list.append((pop_size, pop_update_rule, F, cost, M, group_size, mutation_rate, run_number))

    print("{} of total {} models to run".format(len(param_list), param_count))
    print("{} models per MPI task".format(int(math.ceil(float(len(param_list))/mpi_processes))))


    # Write out current configuration
    try:
        os.mkdir("./results/")
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass

    # now that we have list of parameter tuples, execute in parallel
    # using MPI: each MPI process will process
    # ceil(num_jobs / mpi_processes) of the parmeter tuples in the list
    num_jobs = len(param_list)
    job_i = rank

    txt = "results/" + "result"+ str(rank) + ".csv"
    fileSaving(txt, [], "w")

    while job_i < num_jobs:
        # print(Player.counter)
        # print(Player.player_id)
        # print(QLearningPlayer.counter)
        # print(QLearningPlayer.player_id)
        # print(QLearningPlayer.q_table)


        # print("{} / {}".format(job_i, num_jobs))

        (pop_size, pop_update_rule, F, cost, M, group_size, mutation_rate, run_number) = param_list[job_i]
        
        random.seed()
        rounds = []
        
        games = game(pop_size, pop_update_rule, F, cost, M, group_size, mutation_rate)
        games.init_population()
        
        rounds.append(0)
        games.csvDataBuilder(run_number, rounds[0])

        if pop_update_rule == "actor_critic":
            tf.reset_default_graph()
            N_F = 6    # feature dimension  np.array([ co-cooperator_number, action(0 or 1)])
            N_A = 2    # action             [ "C", "D" ]
            LR_A = 0.001  # Learning rate for actor
            LR_C = 0.01   # Learning rate for critic

            np.random.seed(0)
            tf.random.set_random_seed(0)
            sess = tf.Session()
            actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)  # Initial Actor
            critic = Critic(sess, n_features=N_F, lr=LR_C)     # Initial Critic
            sess.run(tf.global_variables_initializer())        # Initial variables

    
        for roundx in range(1,tmax) :
            # print("roundx = {}, tmax = {}".format(roundx, tmax))
            games.grouping_playing()
            rounds.append(roundx)
            games.update_population(roundx)
            if roundx in time_list:
                games.average_reward()
                games.csvDataBuilder(run_number, roundx)
        games.csvDataBuilder(run_number, roundx)

        txt = "results/" + "result"+ str(rank) + ".csv"
        fileSaving(txt, games.get_lst(), "a")
    
        job_i += mpi_processes

        # reset class variable
        Player.counter = 0
        Player.player_id = -1

        if pop_update_rule == "q_learning":
            # QLearningPlayer.actions = ["C", "D"]
            # QLearningPlayer.learning_rate = 0.01
            # QLearningPlayer.discount_factor = 0.9
            # QLearningPlayer.epsilon = 0.1
            QLearningPlayer.counter = 0
            QLearningPlayer.player_id = -1
            QLearningPlayer.q_table = defaultdict(lambda: [0.0, 0.0])

        # print
        # printl = sorted(QLearningPlayer.q_table.items(), key = lambda kv:kv[0])
        # scale = 100000
        # for (a, [b, c]) in printl:
        #     if b > c:
        #         print("{} [{}, {}] C".format(a, b*scale, c*scale))
        #     elif b < c:
        #         print("{} [{}, {}] D".format(a, b*scale, c*scale))
        #     else:
        #         print("{} [{}, {}] random".format(a, b*scale, c*scale))