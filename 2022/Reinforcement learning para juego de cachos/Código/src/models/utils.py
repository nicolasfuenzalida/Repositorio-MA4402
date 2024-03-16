# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:53:42 2022

@author: javie
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from itertools import product
import time
import pandas as pd

def timeSince(since):
    now = time.time_ns()
    s = now - since
    return s*10**(-9)# Start writing code here...

def plot_multi(df_wr, df_eps, spacing=.1, **kwargs):

    from pandas.plotting._matplotlib.style import get_standard_colors
    cols_wr, cols_eps = df_wr.columns, df_eps.columns
    colors = get_standard_colors(num_colors=len(cols_eps) + len(cols_wr))
    
    fig, ax = plt.subplots()
    # First axis
    for i, col in enumerate(cols_wr):
        if col != "x":
            df_wr.loc[:, col].plot(x = "x", ax = ax, label=col, color = colors[i], **kwargs)
    ax.set_ylabel(ylabel= "Winrate")
    lines, labels = ax.get_legend_handles_labels()

    ax_new = ax.twinx()
    ax_new.spines['right'].set_position(('axes', 1 + spacing))
    for j, col in enumerate(cols_eps):
        if col != "x":
            df_eps.loc[:, col].plot(x = "x",ax=ax_new, label=col, color=colors[j], **kwargs)
    ax_new.set_ylabel(ylabel="Epsilon")
    
    line, label = ax_new.get_legend_handles_labels()
    lines += line
    labels += label

    ax.legend(lines, labels, loc=0)
    return ax

def winrate_every(scores, m = 10):
    hist = []
    for k in range(m, len(scores)+m, m):
        #print(k)
        win_rate_last_m = (np.mean(scores[k-m:k]) + 1)/2
        hist.append(win_rate_last_m)
    return hist

def plot_epsilon(df_epsilons, m = 10, metodo = "Lineal", save = None):
    fig, ax = plt.subplots()
    df_epsilons = df_epsilons.iloc[::m, :]
    for col in df_epsilons.columns:
        if col != "x":
            sns.lineplot(x = "x", y = col, data = df_epsilons, ax = ax, label = "Método {} ($\\varepsilon_0 = $ {}, $\\varepsilon$ mínimo = {})".format(col, round(df_epsilons[col].iloc[0],2), round(df_epsilons[col].iloc[-1],4)))
    ax.set(xlabel = "Número de Juegos evaluados", ylabel = "Valor de $\\varepsilon$", title = "Evolución del parámetro $\\varepsilon$ (Método {})".format(metodo))
    ax.legend()
    if save:
        plt.savefig(save)
    plt.show()
    
def plot_winrate(df_wr, m, save = None):
    fig, ax = plt.subplots()
    for col in df_wr.columns:
        if col != "x":
            sns.lineplot(x = "x", y = col, data = df_wr, ax = ax, label = "Método {}".format(col))
    ax.set(xlabel = "Número de Juegos evaluados", ylabel = "Proporción de Juegos Ganados (Win-Rate)", title = "Evolución del Win-Rate del Agente (calculado cada {} juegos)".format(m))
    ax.legend()
    if save:
        plt.savefig(save)
    plt.show()
    
def plot_time_per_game(df_times, m, save = None):
    fig, ax = plt.subplots()
    #df_times = df_times.iloc[::m, :]
    df_times = df_times.rolling(m).mean().iloc[::m, :]
    for col in df_times.columns:
        if col != "x":
            sns.lineplot(x = "x", y = col, data = df_times, ax = ax, label = "Método {} (Promedio = {} s)".format(col, round(np.nanmean(df_times[col]), 4)))
    ax.set(xlabel = "Número de Juegos evaluados", ylabel = "Tiempo por Juego (s)", title = "Evolución del Tiempo de Cálculo en cada juego", yscale = "log")
    ax.legend()
    if save:
        plt.savefig(save)
    plt.show()
    
    
def Q_plot(agente, estado, tabular = True):
    
    if tabular:
        df1 = pd.DataFrame(agente.Q_table[estado].T)
        #df1 = df.loc[:,[estado]]
    else:
        state = agente.state_to_Tensor(estado).to(agente.Q_eval.device)
        df1 = pd.DataFrame(agente.Q_target.forward(state).cpu().detach().numpy().T)
    rows = {}
    accs = agente.action_space
    validas = agente.valid_actions(estado)
    L=[]
    L_mask = []
    pre_mask = []
    for i in range(len(accs)):
        rows[i] = agente.action_space[i]
        if i in validas:
            pre_mask.append(0)
        else:
            pre_mask.append(1)
    pre_mask = pd.DataFrame(pre_mask)
    df1.rename(index = rows, inplace = 1)
    pre_mask.rename(index = rows, inplace = 1)
    for i in range(sum(estado[1]) + 1):
        row = tuple(df1.iloc[2+i*6:8+i*6,0])
        L.append(row)
        L_mask.append(tuple(pre_mask.iloc[2+i*6:8+i*6,0]))
    LD = [df1.iloc[0,0]]
    DFDudo = pd.DataFrame(LD, columns = ["Dudo"])
    LC = [df1.iloc[1,0]]
    DFCalzo = pd.DataFrame(LC, columns = ["Calzo"])
    DF = pd.DataFrame(L,columns = ["Ases", "Tontos", "Trenes", "Cuadras", "Quinas", "Sextas"])
    MASK = pd.DataFrame(L_mask,columns = ["Ases", "Tontos", "Trenes", "Cuadras", "Quinas", "Sextas"])
    DF, MASK = DF.iloc[1:], MASK.iloc[1:]
    
    Vmin = DF.values.min()
    Vmax = DF.values.max()
    fig, ax = plt.subplots(1,4,figsize = (16,8), gridspec_kw={'width_ratios': [10,1, 1,0.3]})
    if tabular:
        fig.suptitle("Heatmap de la Q-table del {}\n".format(agente.name),fontsize = 26)
    else:
        fig.suptitle("Heatmap de la Deep Q-function del {}\n".format(agente.name),fontsize = 26)
    if estado[2] is None:
        lb = "Primera Apuesta"
    else:
        lb = str(estado[2][0]) + " " + str(estado[2][1])
    st = "Mano del agente:{}\n Dados agente: {}. Dados oponente: {}. Última apuesta: {} ".format(estado[0],estado[1][0],estado[1][1],lb)
    ax[0].set_title(st, loc = "right")
    sns.heatmap(DF,annot = True, ax = ax[0],cbar = False, mask = MASK)
    
    sns.heatmap(DFCalzo,annot = True, ax = ax[2],yticklabels=False, cbar=False,vmin=Vmin, vmax=Vmax)
    
    sns.heatmap(DFDudo,annot = True, ax = ax[1],yticklabels=False, cbar=False,vmin=Vmin, vmax=Vmax)
    fig.colorbar(ax[2].collections[0], cax=ax[3])
    plt.show()

def train_over_games(game_env, agente, n_games, mode = "No-Learn", Target_Update = 10000, verbose = False, every = 100):
    scores, eps_history, times = [], [], []
    global_cnt = 0
    for kk in range(n_games):
        if kk%(n_games//every) == 0:
            print(20*"--")
            print("Creando el Juego", kk +1)
        if mode == "Tabular":
            score, eps, time = train_one_game_Tabular(game_env, agente, verbose = verbose) 
        elif mode == "1-step":
            score, eps, time = train_one_game_1step(game_env, agente, verbose = verbose)
        elif mode == "Deep":
            score, eps, time, global_cnt = train_one_game_DQN(game_env, agente, cnt_target = global_cnt, Target_Update = Target_Update, verbose = verbose)
        else:
            score, eps, time = play_one_game(game_env, agente, verbose = verbose)
        scores.append(score)
        eps_history.append(eps)
        times.append(time)
        if kk%(n_games//100) == 0:
            print("Resultados Parciales:")
            print("Winrate (últimos {} episodios):".format(every), round((np.mean(scores[-every:]) +1)/2, 2))
            print("Epsilon actual:", round(eps,4))
            print("Tiempo por iteración (promedio últimos {} episodios):".format(every), round(np.mean(times[-every:]),4), "s")
    return scores, eps_history, times

def play_one_game(game_env, agente, verbose = False, graphs = False, tabular = True, NPC = False):
    game_env.reset()
    Done, cnt, win_or_lose, i_starter = False, 0, 0, 0
    t0 = time.time_ns()
    while not Done:
        if cnt >0:
            game_env.update_dice(new_dice)
        
        if verbose:
            print("En esta ronda, los jugadores tenían los dados:")
            print(game_env.all_dice)
        score, done, first_bet, cntt = 0, False, None, 0
        # Primer Turno
        if i_starter != 0:
            first_bet, _ =game_env.players[i_starter].play_bet(game_env.all_dice[i_starter], game_env.dice_pp, None, i_starter)
            if verbose:
                print("El Oponente partió jugando", first_bet)
        observation = (game_env.all_dice[0], game_env.dice_pp, first_bet)
        while not done:
            if graphs and not NPC:
                mano, dices, fb = observation
                Q_plot(agente, (tuple(mano), tuple(dices), fb), tabular)
            if not NPC:
                ind_action = agente.choose_action(observation)
                action = game_env.action_space[ind_action]
            else:
                action, _ = agente.play_bet(observation[0], observation[1], observation[2], 0)
            if verbose:
                print("El Agente jugó", action)
            observation_, reward, done, info = game_env.step(action, verbose = verbose)
            msg, i_starter = info
            score += reward
            observation = observation_
            cntt +=1
        new_dice = observation[1]
        cnt +=1
        if 0 in new_dice:
            Done = True
            if new_dice[0] != 0:
                if verbose:
                    print("AGENT WINS !")
                win_or_lose += 1
            else:
                if verbose:
                    print("OPPONENT WINS !")
                win_or_lose -=1
        if verbose:  
            print(score, info, observation)
            print(20*"----")
    if verbose:
        print("Fin del Juego")
    dt = timeSince(t0)
    if NPC:
        return win_or_lose, None, dt
    return win_or_lose, agente.epsilon, dt

def train_one_game_Tabular(game_env, agente, verbose = False):
    game_env.reset()
    Done, cnt, win_or_lose, i_starter = False, 0, 0, 0
    t0 = time.time_ns()
    while not Done:
        if cnt >0:
            game_env.update_dice(new_dice)
        
        if verbose:
            print("En esta ronda, los jugadores tenían los dados:")
            print(game_env.all_dice)
        score, done, first_bet, cntt = 0, False, None, 0
        # Primer Turno
        if i_starter != 0:
            first_bet, _ =game_env.players[i_starter].play_bet(game_env.all_dice[i_starter], game_env.dice_pp, None, i_starter)
            if verbose:
                print("El Oponente partió jugando", first_bet)
        observation = (game_env.all_dice[0], game_env.dice_pp, first_bet)
        traj = []
        while not done:
            ind_action = agente.choose_action(observation)
            action = game_env.action_space[ind_action]
            if verbose:
                print("El Agente jugó", action)
            observation_, reward, done, info = game_env.step(action, verbose = verbose)
            msg, i_starter = info
            score += reward
            traj.append((observation, ind_action))
            observation = observation_
            cntt +=1
        agente.learn(traj, score)
        new_dice = observation[1]
        cnt +=1
        if 0 in new_dice:
            Done = True
            if new_dice[0] != 0:
                if verbose:
                    print("AGENT WINS !")
                win_or_lose += 1
            else:
                if verbose:
                    print("OPPONENT WINS !")
                win_or_lose -=1
        if verbose:  
            print(score, info, observation)
            print(20*"----")
    if verbose:
        print("Fin del Juego")
    dt = timeSince(t0)
    return win_or_lose, agente.epsilon, dt

def train_one_game_1step(game_env, agente, verbose = False):
    game_env.reset()
    Done = False
    cnt = 0
    win_or_lose = 0
    i_starter = 0
    t0 = time.time_ns()
    while not Done:
        if cnt >0:
            game_env.update_dice(new_dice)
        
        if verbose:
            print("En esta ronda, los jugadores tenían los dados:")
            print(game_env.all_dice)
        score = 0
        done = False
        #Primer Turno:
        first_bet = None
        if i_starter != 0:
            first_bet, _ =game_env.players[i_starter].play_bet(game_env.all_dice[i_starter], game_env.dice_pp, None, i_starter)
            if verbose:
                print("El Oponente partió jugando", first_bet)
        observation = (game_env.all_dice[0], game_env.dice_pp, first_bet)
        ind_action = agente.choose_action(observation)
        action = game_env.action_space[ind_action]
        if verbose:
            print("El Agente jugó", action)
        cntt = 0
        while not done:
            observation2, reward, done, info = game_env.step(action, verbose = verbose)
            msg, i_starter = info
            if not done:
                ind_action2 = agente.choose_action(observation2)
                action2 = game_env.action_space[ind_action2]
                if verbose:
                    print("El Agente jugó", action2)
            else:
                ind_action2 = None
                action2 = None

            agente.learn(observation, ind_action, reward, observation2, ind_action2, done)

            observation = observation2
            ind_action = ind_action2
            action = action2
            score += reward
            #agente.learn(observation, ind_action, reward)
            #observation = observation_
            cntt +=1
        new_dice = observation[1]
        cnt +=1
        if 0 in new_dice:
            Done = True
            if new_dice[0] != 0:
                if verbose:
                    print("AGENT WINS !")
                win_or_lose += 1
            else:
                if verbose:
                    print("OPPONENT WINS !")
                win_or_lose -=1
        if verbose:  
            print(score, info, observation)
            print(20*"----")
    dt = timeSince(t0)
    return win_or_lose, agente.epsilon, dt


def train_one_game_DQN(game_env, agente, cnt_target = 0, Target_Update = 10000, verbose = False):
    game_env.reset()
    Done = False
    cnt = 0
    win_or_lose = 0
    i_starter = 0
    t0 = time.time_ns()
    while not Done:
        if cnt >0:
            game_env.update_dice(new_dice)
        
        if verbose:
            print("En esta ronda, los jugadores tenían los dados:")
            print(game_env.all_dice)
        score = 0
        done = False
        #Primer Turno:
        first_bet = None
        if i_starter != 0:
            first_bet, _ =game_env.players[i_starter].play_bet(game_env.all_dice[i_starter], game_env.dice_pp, None, i_starter)
            if verbose:
                print("El Oponente partió jugando", first_bet)
        observation = (game_env.all_dice[0], game_env.dice_pp, first_bet)
        while not done:
            ind_action = agente.choose_action(observation)
            action = game_env.action_space[ind_action]
            #action, _ = placeholder.play_bet(observation[0], sum(observation[1]), observation[2])
            #action = (2*cntt, "Sextas")
            if verbose:
                print("El Agente jugó", action)
            observation_, reward, done, info = game_env.step(action, verbose = verbose)
            msg, i_starter = info
            score += reward
            #print(observation_, observation)
            agente.store_transition(agente.state_to_Tensor(observation), ind_action, reward, agente.state_to_Tensor(observation_), done)
            agente.learn()
            cnt_target += 1
            observation = observation_
            # Update the target network, copying all weights and biases in DQN
            if cnt_target % Target_Update == 0:
                print(20*"----")
                print("Target Updated")
                print(20*"----")
                agente.Q_target.load_state_dict(agente.Q_eval.state_dict())
        new_dice = observation[1]
        cnt +=1
        if 0 in new_dice:
            Done = True
            if new_dice[0] != 0:
                if verbose:
                    print("AGENT WINS !")
                win_or_lose += 1
            else:
                if verbose:
                    print("OPPONENT WINS !")
                win_or_lose -=1
        if verbose:  
            print(score, info, observation)
            print(20*"----")
    dt = timeSince(t0)
    return win_or_lose, agente.epsilon, dt, cnt_target

