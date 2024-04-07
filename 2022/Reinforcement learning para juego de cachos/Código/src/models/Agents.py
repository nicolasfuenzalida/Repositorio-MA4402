# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 23:01:23 2022

@author: javie
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pickle
from itertools import product
import pandas as pd
import seaborn as sns
import bz2



# Red Neuronal con la que funcionará el agente DQN

# La red recibe un ESTADO y devuelve una distribución de valores para cada entrada (de forma (BATCH_SIZE, |Acciones|))
class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions, bn = True):
        super(DeepQNetwork, self).__init__()
        self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions = input_dims, fc1_dims, fc2_dims, n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.BatchNorm1d(self.fc2_dims)
        # Devuelve un vector del largo del número de acciones posibles, pq a cada uno le asocia un valor
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.bn = bn

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        if self.bn:
            x = self.bn1(x)
        x = F.relu(self.fc2(x))
        if self.bn:
            x = self.bn2(x)
        actions = self.fc3(x)

        return actions

# Agente que se decide con una "tabla" en memoria, dictándole qué hacer desde cada estado
class AgentTable:
    def __init__(self, gamma, epsilon, lr, game_env,
                 max_mem_size=100000, eps_end=0.05, eps_dec=5e-4, eps_method = "Lineal", name = "Agente"):
        self.name = name
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.eps_method = eps_method
        self.lr = lr
        self.game_env = game_env
        self.action_space = self.game_env.action_space
        self.n_actions = len(self.action_space)
        self.action_space_ind = [i for i in range(self.n_actions)]
        self.action_to_ind = dict([(act, i) for i, act in enumerate(self.action_space)])
        self.iter_cntr = 0
        self.Q_table = self.init_Q_table()
        
    def init_Q_table(self):
        Q = {}
        for n_dados in range(1,6):
            for n_dados_op in range(1,6):
                for x in product(range(1,7),repeat=n_dados):
                    bonito = np.zeros(6,dtype = np.int32)
                    for v in x:
                        bonito[v-1] += 1
                    #for y in product(range(0, n_dados + n_dados_op +1), ["Ases", "Tontos", "Trenes", "Cuadras", "Quinas", "Sextas"]):
                    for y in product(range(0, 5 + 5 +1), ["Ases", "Tontos", "Trenes", "Cuadras", "Quinas", "Sextas"]):
                        Q[tuple(bonito), (n_dados, n_dados_op), y] = np.random.randn(self.n_actions)#, dtype = np.float32) 
                    Q[tuple(bonito), (n_dados, n_dados_op), None] = np.random.randn(self.n_actions)#, dtype = np.float32) 
                    
        return Q
   
    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1
    
    def epsilon_step(self):
        if self.eps_method == "Lineal":
            eps = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        elif self.eps_method == "Exponencial":
            eps = self.epsilon*self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        else:
            eps = self.epsilon
        return eps
            
    def valid_actions(self, observation):
        _, n_dices, lb = observation
        return [0,1] + [i for i in range(2,self.n_actions) if self.game_env.verify_subida(self.action_space[i], lb, n_dices)]
     
    def best_valid_action(self, observation, actions):
        # Observation es el estado en que estamos:
        # Actions es el array que representa Q(s,a) dado por la red en ese estado:
        idx = self.valid_actions(observation)
        #print(idx, actions.shape, actions[idx])
        max_val = np.max(actions[idx])
        #print(max_val)
        #print(np.where(actions == max_val)[-1][0])
        #return np.argmax(actions)
        act = np.where(actions == max_val)
        #elijo el [-1] para que siempre opte antes por subir o calzar antes que dudar al tiro.
        return act[-1][0]
    
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            agent_dice, n_dices, last_bet = observation
            Q_on_state = self.Q_table[tuple(agent_dice), tuple(n_dices), last_bet]
            action = self.best_valid_action(observation, Q_on_state)#np.argmax(Q_on_state)
        else:
            action = np.random.choice(self.valid_actions(observation))
        return action
    
    
    def learn(self, trajectory, reward):
        for i, (state, action) in enumerate(trajectory):
            agent_dice, n_dices, last_bet = state

            q_val = self.Q_table[tuple(agent_dice), tuple(n_dices), last_bet][action]

            self.Q_table[tuple(agent_dice), tuple(n_dices), last_bet][action] = q_val + self.lr * (reward - q_val)
            self.iter_cntr += 1
            self.epsilon = self.epsilon_step()
            
    def save_table(self, name):
        nombre = name + ".pickle"
        a = self.Q_table
        #with open(nombre, 'wb') as handle:
        ofile = bz2.BZ2File(name,'wb')
        pickle.dump(a, ofile)
        ofile.close()
        #with bz2.BZ2File(nombre,'wb') as handle:
        #        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pass
    
    def load_table(self, name):
        nombre = name + ".pickle"
        ifile = open(name,'rb')
        a = pickle.loads(bz2.decompress(ifile.read()))
        ifile.close()
        #with open(nombre, 'rb') as handle:
        #    a = pickle.load(handle)
        self.Q_table = a
        pass
    
class Agent_SARSA(AgentTable):
    def learn(self, state, action, reward, state2, action2, done):
        agent_dice, n_dices, last_bet = state
        agent_dice2, n_dices2, last_bet2 = state2
        
        predict = self.Q_table[tuple(agent_dice), tuple(n_dices), last_bet][action]
        if not done:
            target = reward + self.gamma * self.Q_table[tuple(agent_dice2), tuple(n_dices2), last_bet2][action2]
        else:
            target = reward
        self.Q_table[tuple(agent_dice), tuple(n_dices), last_bet][action] = predict + self.lr * (target - predict)
        self.iter_cntr += 1
        self.epsilon = self.epsilon_step()

class Agent_QLearning(AgentTable):
    def learn(self, state, action, reward, state2, action2, done):
        agent_dice, n_dices, last_bet = state
        agent_dice2, n_dices2, last_bet2 = state2
        
        predict = self.Q_table[tuple(agent_dice), tuple(n_dices), last_bet][action]
        
        if not done:
            target = reward + self.gamma * np.max(self.Q_table[tuple(agent_dice2), tuple(n_dices2), last_bet2])
        else:
            target = reward
        self.Q_table[tuple(agent_dice), tuple(n_dices), last_bet][action] = predict + self.lr * (target - predict)
        self.iter_cntr += 1
        self.epsilon = self.epsilon_step()

# Agente que funciona con una Red Neuronal Profunda

class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, game_env,
                 max_mem_size=100000, eps_end=0.05, eps_dec=5e-4, eps_method = "Lineal", name = "Agente"):
        self.name = name
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.eps_method = eps_method
        self.lr = lr
        
        self.game_env = game_env
        self.action_space = self.game_env.action_space
        self.n_actions = len(self.action_space)
        self.action_space_ind = [i for i in range(self.n_actions)]
        self.action_to_ind = dict([(act, i) for i, act in enumerate(self.action_space)])
        
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = DeepQNetwork(lr, n_actions=self.n_actions,
                                   input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=128, bn = False)
        self.Q_target = DeepQNetwork(lr, n_actions=self.n_actions,
                                   input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=128, bn= False)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.Q_target.eval()
        
        #memory = ReplayMemory(10000)
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.dice_names = {"Ases":1, "Tontos":2, "Trenes":3, "Cuadras":4, "Quinas":5, "Sextas":6}
        
    def state_to_Tensor(self, state):
        dice_agent, n_dices, lb = state
        if lb is None:
            L = [k for k in dice_agent] + [k for k in n_dices] + [-1, -1]
        else:
            (valor, pinta) = lb
            L = [k for k in dice_agent] + [k for k in n_dices] + [valor] + [self.dice_names[pinta]]
        return torch.tensor([L]).float()
    
    def epsilon_step(self):
        if self.eps_method == "Lineal":
            eps = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        elif self.eps_method == "Exponencial":
            eps = self.epsilon*self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        else:
            eps = self.epsilon
        return eps    
    
    def valid_actions(self, observation):
        _, n_dices, lb = observation
        return [0,1] + [i for i in range(2,self.n_actions) if self.game_env.verify_subida(self.action_space[i], lb, n_dices)]
    
    def best_valid_action(self, observation, actions):
        # Observation es el estado en que estamos:
        # Actions es el array que representa Q(s,a) dado por la red en ese estado:
        idx = torch.tensor(self.valid_actions(observation))
        #print(idx, actions.shape)
        max_val = torch.max(torch.index_select(actions, 1, idx.to(self.Q_eval.device))).item()
        #print(max_val)
        #return torch.argmax(actions).item()
        act = (actions == max_val).nonzero(as_tuple=True)
        #elijo el [-1] para que siempre opte antes por subir o calzar antes que dudar al tiro.
        return act[-1].item()

    
    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = self.state_to_Tensor(observation).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = self.best_valid_action(observation, actions)
            #print(action)
        else:
            action = np.random.choice(self.valid_actions(observation))

        return action
    
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(
                self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = torch.tensor(
                self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(
                self.terminal_memory[batch]).to(self.Q_eval.device)
        #print(self.Q_eval)
        #print(state_batch.shape)
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        #q_next = self.Q_eval.forward(new_state_batch)
        q_next = self.Q_target.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon_step()
        
    def save(self, name):
        torch.save(self.Q_eval.state_dict(), name)
        
    def load(self, name):
        self.Q_eval.load_state_dict(torch.load(name,map_location=torch.device('cpu')))
        self.Q_target.load_state_dict(torch.load(name,map_location=torch.device('cpu')))

if __name__ == "__main__":
    from EnvironmentCachos import AmbienteCacho, input_NPC, Strategic_NPC, Agent_NPC, Agent_Human
    from utils import train_over_games, play_one_game, plot_winrate, winrate_every, plot_time_per_game, plot_epsilon, Q_plot, plot_multi
    # Tests del Tabular
    #game_env = AmbienteCacho(2)
    #at = AgentTable(0.99, 0.1, 0.001, len(game_env.action_space))
    #ind_action = at.choose_action((np.array([1, 3, 0, 0, 1, 0]), [5, 5], (3, 'Quinas')))
    #print(game_env.action_space[ind_action])
    
    # Tests del DQN
    #game_env = AmbienteCacho(2)
    #agent = Agent(gamma=0.99, epsilon=0.5, batch_size=64, game_env=game_env, eps_end=0.01, input_dims=[10], lr=0.001)
    #estado = torch.cat([agent.state_to_Tensor(((0,0,1,1,0,0),(2,5), None)), agent.state_to_Tensor(((0,0,1,1,0,0),(2,5), None))])

    #print(agent.Q_eval)
    # game_env = AmbienteCacho(2)
    # agente = AgentTable(gamma=0.99, epsilon=0.01, game_env = game_env, eps_end=0.01, lr=0.001)
        
    #agente.load_table("..\\..\\..\\models\\NEWTEST_Save10000eps_dec1e5.pt")
    # with open("..\\..\\..\\models\\NEWTEST_Save10000eps_dec1e5.pt.pickle", 'rb') as handle:
    #     a = pickle.load(handle)
    #scores, eps_history = train_over_games_DQN(game_env, agente, 100, verbose =True)
    #    
    #print(np.mean(scores))
    #plot_winrate(scores)
    # n_games = 100
    # every = n_games//100
    # scores_dict = {"x": every* np.arange(n_games//every)}
    # times_dict = {"x": np.arange(n_games)}
    # epsilon_dict = {"x": np.arange(n_games)}
    
    # game_env = AmbienteCacho(2, NPC_type = Strategic_NPC)
    # agente_MC = AgentTable(0.99, 0.5, 0.001, game_env)
    # scores, eps_hist, times = train_over_games(game_env, agente_MC, n_games, mode = "Tabular", verbose = True)
    # print(np.mean(scores))
    # scores_dict["MC Control"] = winrate_every(scores, every)
    # times_dict["MC Control"] = times
    # epsilon_dict["MC Control"] = eps_hist
    # scores_dict["WinRate"] = winrate_every(scores, every)
    # epsilon_dict["Epsilon"] = eps_hist
    
    # df_wr = pd.DataFrame(scores_dict)
    # df_epsilons = pd.DataFrame(epsilon_dict)
    # plot_multi(df_wr, df_epsilons)
    
    # agente_SARSA = Agent_SARSA(0.99, 0.5, 0.1, game_env)
    # scores, eps_hist, times = train_over_games(game_env, agente_SARSA, n_games, mode = "1-step", verbose =True)
    # print(np.mean(scores))
    # scores_dict["SARSA"] = winrate_every(scores, every)
    # times_dict["SARSA"] = times
    # epsilon_dict["SARSA"] = eps_hist
    
    # agente_QLearning = Agent_QLearning(0.99, 0.5, 0.1, game_env)
    # scores, eps_hist, times = train_over_games(game_env, agente_QLearning, n_games, mode =  "1-step", verbose =True)
    # print(np.mean(scores))
    # scores_dict["QLearning"] = winrate_every(scores, every)
    # times_dict["QLearning"] = times
    # epsilon_dict["QLearning"] = eps_hist
    
    # df_wr = pd.DataFrame(scores_dict)
    # df_times = pd.DataFrame(times_dict)
    # df_epsilons = pd.DataFrame(epsilon_dict)
    # plot_winrate(df_wr, every)
    # plot_time_per_game(df_times, m = 10*every)
    # print(df_times.mean())
    # plot_epsilon(df_epsilons, m = every, metodo = "Lineal")
    
    game_env = AmbienteCacho(2)
    agente = Agent(gamma=0.99, epsilon=0.01, batch_size=64, game_env = game_env, eps_end=0.01, input_dims=[10], lr=0.001)
    #agente.load("..\\..\\models\\NEWTEST_Save10000eps_dec1e5.pt")
    agente.load("..\\..\\models\\QNetwork_10000eps_0.001decNEW.pt")
    A_NPC = Agent_NPC(agente, game_env.action_space)
    game_env.ow_NPC([A_NPC])
    #print(game_env.players)
    #agente_SARSA = Agent_SARSA(0.99, 1, 0.1, game_env)
    #n_games = 1000
    #scores, eps_hist = train_over_games_SARSA(game_env, agente_SARSA, n_games, verbose =True)
    #print(np.mean(scores))
    
    game_env.play_human()
    # game_env = AmbienteCacho(2)
    # agente = Agent(gamma=0.99, epsilon=0.01, batch_size=64, game_env = game_env, eps_end=0.01, input_dims=[10], lr=0.001)
    # Q_plot(agente, ((0,1,1,1,1,1),(5,5), (2, "Trenes")), tabular = False)