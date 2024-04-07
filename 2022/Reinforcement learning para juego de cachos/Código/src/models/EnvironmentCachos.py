# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 19:21:53 2022

@author: javie
"""
import numpy as np
import string
import torch
from itertools import product

class NPC():
    def __init__(self, name = "Oponente"):
        self.dice_names = {"Ases":1, "Tontos":2, "Trenes":3, "Cuadras":4, "Quinas":5, "Sextas":6}
        self.map_dice_names = ["Ases", "Tontos", "Trenes", "Cuadras", "Quinas", "Sextas"] 
        self.dice_values = [1,2,3,4,5,6]
        self.name = name
        pass
    
    def set_name(self, name):
        self.name = name
        return self
    
    def sample_dice(self, N):
        sampled_dice_values = np.random.choice(self.dice_values, size=N, replace=True)
        count = np.zeros(6,dtype = np.int32)
        for v in sampled_dice_values:
            count[v-1] += 1
        return count
    
    def play_bet(self, dices, n_dices, last_bet, ind):
        n_dice_total = sum(n_dices)
        valor, pinta = last_bet
        if pinta =="Ases":
            esperado_por_pinta = n_dice_total/6
        else:
            # Porque en esencia estamos buscando 2 pintas
            esperado_por_pinta = n_dice_total/3
        if abs(valor - esperado_por_pinta) < 0.1:
            env_response = "Calzo"
        elif valor < esperado_por_pinta:
            env_response = (valor + 1, pinta)
        elif valor > esperado_por_pinta:
            env_response = "Dudo"
        return env_response, last_bet
    def __str__(self):
        return self.name
    
class Strategic_NPC(NPC):
    def play_bet(self, dices, n_dices, last_bet, ind):
        n_dice_total = sum(n_dices)
        mis_dados = sum(dices)
        mi_mejor_pinta = np.argmax(dices)
        if last_bet is None:
            return (max(1,n_dice_total//6), self.map_dice_names[1 + np.argmax(dices[1:])]), last_bet
        valor, pinta = last_bet
        n_pinta_en_juego = self.dice_names[pinta]-1
        if pinta =="Ases":
            mis_ases = dices[0]
            esperado_por_pinta = mis_ases + (n_dice_total - mis_dados)/6
        else:
            # Porque en esencia estamos buscando 2 pintas
            mis_deesapinta = dices[0] + dices[n_pinta_en_juego]
            esperado_por_pinta = mis_deesapinta + (n_dice_total - mis_dados)/3
        if abs(valor - esperado_por_pinta) < 0.1:
            env_response = "Calzo"
        elif valor < esperado_por_pinta:
            if n_pinta_en_juego < mi_mejor_pinta:
                if n_pinta_en_juego == 0:
                    env_response = (2*valor + 1, self.map_dice_names[mi_mejor_pinta])
                else:
                    env_response = (valor, self.map_dice_names[mi_mejor_pinta])    
            else:
                env_response = (valor + 1, pinta)
        elif valor > esperado_por_pinta:
            env_response = "Dudo"
        return env_response, last_bet

class Agent_NPC(NPC):
    def __init__(self, agente, action_space, name = "Oponente"):
        self.dice_names = {"Ases":1, "Tontos":2, "Trenes":3, "Cuadras":4, "Quinas":5, "Sextas":6}
        self.map_dice_names = ["Ases", "Tontos", "Trenes", "Cuadras", "Quinas", "Sextas"] 
        self.dice_values = [1,2,3,4,5,6]
        self.name = name
        # ponerle un pretrained agent
        self.agente = agente
        self.action_space = action_space
    
    def play_bet(self, dice, n_dices, last_bet, ind):
        # El agente está acostumbrado a ver los n_dices con: los suyos en el n_dices[0], y el resto siguiendo el orden hacia adelante.
        n_dices_persp_agent = n_dices[ind:] + n_dices[:ind]
        observation = dice, n_dices_persp_agent, last_bet
        ind_action = self.agente.choose_action(observation)
        action = self.action_space[ind_action]
        return action, last_bet

class input_NPC(NPC):
    def play_bet(self, dices, n_dices, last_bet, ind):
        n_dice_total = sum(n_dices)
        mis_dados = sum(dices)
        jugadaAprobada = 0 #Test de si el input es válido
        while not jugadaAprobada:
            acc = str(input("Ingrese su acción: \n"))
            acc = acc.replace(" ","")
            acc = acc.lower()
            valor_aumentado = 0
            pinta_a_subir = ""
            if acc[0].isnumeric():
                for c in list(acc):
                    if c.isnumeric():
                        valor_aumentado = valor_aumentado*10 + int(c)
                    else:
                        pinta_a_subir += c
                #Subo apuesta
                mayus = pinta_a_subir[0]
                resto = pinta_a_subir[1:]
                pinta_a_subir = mayus.upper() + resto
                if pinta_a_subir not in self.map_dice_names:
                    print("Pinta inválida")
                elif last_bet is None and pinta_a_subir == "Ases":
                    print("No puedes comenzar con Ases ")
                else:
                    valor_aumentado = int(acc[0])
                    if last_bet is None:
                        if valor_aumentado >0:
                            jugadaAprobada = 1
                            env_response = (valor_aumentado, pinta_a_subir)
                        else:
                            print("Jugada inválida")
                    else:
                        valor, pinta = last_bet
                        n_pinta_en_juego = self.dice_names[pinta]-1
                        if pinta=="Ases": #Venimos de Ases                            
                            if pinta_a_subir != "Ases": #Quiero cambiar a otra pinta
                                if valor_aumentado < valor*2 or valor_aumentado>n_dice_total:
                                    #Condiciones de fallo: no sube bien o se pasa del total de dados.
                                    print("Jugada inválida")
                                else:
                                    jugadaAprobada = 1
                                    env_response = (valor_aumentado,pinta_a_subir)
                            else: #Seguimos en Ases
                                if valor_aumentado <= valor or valor_aumentado>n_dice_total:
                                    #Condiciones de fallo: no sube o se pasa del total de dados
                                    print("Jugada inválida")
                                else:
                                    jugadaAprobada = 1
                                    env_response = (valor_aumentado,pinta_a_subir)
                        else: #No venimos de Ases
                            if pinta_a_subir == "Ases": #Cambio a Ases
                                if valor_aumentado <= valor//2 or valor_aumentado>n_dice_total:
                                    #Condiciones de fallo: baja demasiado 0 se pasa del total de dados.
                                    print("Jugada inválida")
                                else:
                                    jugadaAprobada = 1
                                    env_response = (valor_aumentado,pinta_a_subir)
                            else: #Seguimos en pintas normales
                                if pinta_a_subir == pinta:
                                    if valor_aumentado <= valor or valor_aumentado>n_dice_total:
                                        #Condiciones de fallo: no sube o se pasa del total de dados
                                        print("Jugada inválida")
                                    else:
                                        jugadaAprobada = 1
                                        env_response = (valor_aumentado,pinta_a_subir)
                                else:
                                    if valor_aumentado < valor or valor_aumentado>n_dice_total:
                                        #Condiciones de fallo: no sube o se pasa del total de dados
                                        print("Jugada inválida")
                                    else:
                                        jugadaAprobada = 1
                                        env_response = (valor_aumentado,pinta_a_subir)
            elif acc.lower() == "dudo" or acc.lower() == "calzo":
                acc = acc.lower()
                acc = acc[0].upper() + acc[1:]
                jugadaAprobada = 1
                env_response = acc
            elif acc in ["exit","Exit","Salir","salir"]:
                print("Partida interrumpida")
                return None
            else:
                print("Jugada inválida")
        return env_response, last_bet

# Agente para que un humano juegue
class Agent_Human():
    def __init__(self):
        self.NPC = input_NPC()
        
    def choose_action(self, observation):
        dices, n_dices, last_bet = observation
        action, _ = self.NPC.play_bet(dices, n_dices, last_bet, 0)
        return action    
       
# Definimos el Ambiente de Juego
    
# Quiero responder a esto: observation_, reward, done, info = env.step(action)

class AmbienteCacho():
    def __init__(self, n_players, all_dice = None, NPC_type = Strategic_NPC, fixed_NPC = None):
        self.dice_names = {"Ases":1, "Tontos":2, "Trenes":3, "Cuadras":4, "Quinas":5, "Sextas":6}
        self.dice_pintas = ["Ases", "Tontos", "Trenes", "Cuadras", "Quinas", "Sextas"] 
        self.dice_values = [1,2,3,4,5,6]
        self.m = n_players
        self.players = [None]
        for k in range(n_players -1):
            if not fixed_NPC is None:
                self.players.append(fixed_NPC.set_name("Oponente {}".format(k+1)))
            else:
                self.players.append(NPC_type(name =  "Oponente {}".format(k+1)))
        if all_dice is None:
            self.all_dice = [self.sample_dice(5)] + [p.sample_dice(5) for p in self.players[1:]]
        else:
            self.all_dice = all_dice    
        self.dice_pp = [sum(L) for L in self.all_dice]
        self.n_dice_total = sum(self.dice_pp)
        self.action_space = ["Dudo", "Calzo"] + list(product(range(0,self.n_dice_total +1), self.dice_pintas))
        self.last_bet = None
         # None representa que ahí va el agente
        self.direction = 1
        pass
    
    def sample_dice(self, N):
        sampled_dice_values = np.random.choice(self.dice_values, size=N, replace=True)
        count = np.zeros(6,dtype = np.int32)
        for v in sampled_dice_values:
            count[v-1] += 1
        return count 
    
    def ow_NPC(self, new_NPC):
        self.players[1:] = new_NPC
    
    def reset(self):
        self.all_dice = [self.sample_dice(5) for _ in range(self.m)]
        self.dice_pp = [sum(L) for L in self.all_dice]
        self.n_dice_total = sum(self.dice_pp)
        self.last_bet = None
        
    def update_dice(self, n_dices_pp):
        self.all_dice = [self.sample_dice(k) for k in n_dices_pp]
        self.dice_pp = n_dices_pp
        self.n_dice_total = sum(self.dice_pp)
        self.last_bet = None
    
    def state_to_Tensor(self, state):
        dice_agent, n_dices, (valor, pinta) = state
        L = [k for k in dice_agent] + [k for k in n_dices] + [valor] + [self.dice_names[pinta]]
        return torch.tensor(L).float()
    
    def resolve_round(self, action):
        # Retorna un booleano, indicando si fue buena la acción que se hizo con respecto 
        # a la apuesta, dados los dados que estaban en el juego
        
        # Evaluamos el juego según cómo se cortó la ronda
        total_count = np.zeros(6, dtype = np.int32)
        for dices in self.all_dice:
            total_count += dices
        
        if self.last_bet is None:
            # Significa que es la primera apuesta en el juego
            return False
        valor, pinta = self.last_bet

        if pinta == "Ases":
            # Si son ases, sólo los cuento a ellos
            cuenta_ronda_pinta = total_count[0]
        else:
            # Si son de otra pinta, cuentan los ases y ellos mismos
            cuenta_ronda_pinta = total_count[0] + total_count[self.dice_names[pinta]-1]
    
        if action == "Dudo":
            if cuenta_ronda_pinta < valor: 
                # Bien dudado, el anterior apostó más de lo que había
                return True
            else:
                # Mal dudado, había justo lo que se dijo que había o más
                return False
        elif action == "Calzo":
                if cuenta_ronda_pinta == valor: 
                    # Bien calzado, había justo esa cantidad de ases
                    return True
                else:
                    # Mal calzado, había más o menos que eso
                    return False
    
    def verify_subida(self, action, lb, n_dices):
        valor, pinta = action
        if valor == 0 or valor > sum(n_dices):
            return False
        if lb is None:
            return pinta != "Ases" and valor > 0
        valor_prev, pinta_prev = lb
        if pinta == pinta_prev:
            return valor > valor_prev
        if pinta_prev == "Ases":
            # Si cambié desde las Ases, debe ser a al menos el doble más 1
            return valor >= 2*valor_prev +1
        if pinta == "Ases":
            # Si cambié a Ases, es al menos a la mitad +1
            return valor >= valor_prev//2 + 1
        if self.dice_names[pinta] >= self.dice_names[pinta_prev]:
            # Si cambié a una pinta más grande, puede ser >=
            return valor >= valor_prev
        else:
            # Si NO, debe ser mayor estricto
            return valor > valor_prev
    
    def step(self, action, verbose = False):
        # Retorna new_state, reward, done, (info, i_starter)
        if action == "Dudo" or action == "Calzo":
            bien_hecho = self.resolve_round(action)
            # REHACER ESTO, poner done = True
            if bien_hecho and action == "Calzo":
                nuevos_dados = self.dice_pp.copy()
                nuevos_dados[0] += 1
                if nuevos_dados[0] > 5:
                    nuevos_dados[0] = 5
                return (self.all_dice[0], nuevos_dados, self.last_bet), 1, True, ("Agent Calzó Bien", 0) #self.n_dice_agent +1, self.n_dice_opponent
            elif bien_hecho and action == "Dudo":
                nuevos_dados = self.dice_pp.copy()
                nuevos_dados[-self.direction] -= 1 
                return (self.all_dice[0], nuevos_dados, self.last_bet), 1, True, ("Agent Dudó Bien", -1) #self.n_dice_agent, self.n_dice_opponent -1
            elif not bien_hecho:
                nuevos_dados = self.dice_pp.copy()
                nuevos_dados[0] -= 1
                return (self.all_dice[0], nuevos_dados, self.last_bet), -1, True, ("Agent {} Mal".format(action), 0) #self.n_dice_agent -1, self.n_dice_opponent
        else:
            # El agente hizo una nueva apuesta
            condiciones = self.verify_subida(action, self.last_bet, self.dice_pp)
            if not condiciones:
                # TEST: si apuesta mal, me deja en el mismo estado y me resta reward por gil, pero se sigue jugando
                return (self.all_dice[0], self.dice_pp, self.last_bet), -1, False, ("Agent Apostó mal",0)
            self.last_bet = action
            turn_cntt = self.direction
            while turn_cntt%self.m != 0:
                curr_player = self.players[turn_cntt%self.m]
                env_response, prev_bet = curr_player.play_bet(self.all_dice[turn_cntt], self.dice_pp, self.last_bet, turn_cntt)
                if verbose:
                    print("{} Played: ".format(curr_player), env_response)
                # Esta new_bet es la que le llega de vuelta al agente
                # ESTO VA A HABER QUE CAMBIARLO SI HAY MAS PLAYERS
                if env_response == "Dudo" or env_response == "Calzo":
                    bien_hecho = self.resolve_round(env_response)
                    if bien_hecho and env_response == "Calzo":
                        nuevos_dados = self.dice_pp.copy()
                        nuevos_dados[-self.direction] += 1
                        if nuevos_dados[-self.direction] > 5:
                            nuevos_dados[-self.direction] = 5
                        # ACA EN VOLA NO TIENE QUE SER LAST_BET, SINO LA ULTIMA QUE LE LLEGO AL PLAYER
                        return (self.all_dice[0], nuevos_dados, self.last_bet), -1, True, ("{} Calzó bien".format(curr_player), turn_cntt%self.m) #self.n_dice_agent, self.n_dice_opponent +1
                    elif bien_hecho and env_response == "Dudo":
                        nuevos_dados = self.dice_pp.copy()
                        nuevos_dados[0] -= 1
                        return (self.all_dice[0], nuevos_dados, self.last_bet), -1, True, ("{} Dudó bien".format(curr_player), (turn_cntt -1)%self.m) #self.n_dice_agent-1, self.n_dice_opponent
                    elif not bien_hecho:
                        nuevos_dados = self.dice_pp.copy()
                        nuevos_dados[-self.direction] -= 1
                        return (self.all_dice[0], nuevos_dados, self.last_bet), 1, True, ("{} {} Mal".format(curr_player, env_response), turn_cntt%self.m) #self.n_dice_agent, self.n_dice_opponent -1
                else:
                    self.last_bet = env_response
                    turn_cntt += self.direction
            # Si salgo del while, es pq volví al Agent
            # El ambiente me lleva a un nuevo estado, en que esa es la nueva apuesta
            return (self.all_dice[0], self.dice_pp, env_response), 0, False, ("La ronda subió la apuesta a {}".format(env_response), 0) #self.n_dice_agent -1, self.n_dice_opponent
            #states.append((agent_dice, self.n_dice_opponent, env_response))
            # Con esto, vuelvo a iterar con el agente, viendo qué fue lo último que le llegó
    
    def play_human(self):
        self.reset()
        #Nosotros vamos a ser el nuevo agente
        player_human = Agent_Human()
        self.reset()
        Done = False
        cnt = 0
        i_starter = 0
        while not Done:
            if cnt >0:
                self.update_dice(new_dice)
            done = False
            #observation = env.reset()
            #Primer Turno:
            first_bet = None
            print("Esta ronda tienes los dados: ", self.all_dice[0], ", y los dados por jugador son:", self.dice_pp)
            if i_starter != 0:
                first_bet, _ = self.players[i_starter].play_bet(self.all_dice[i_starter], self.dice_pp, None, i_starter)
                print("El Oponente partió jugando", first_bet)
            observation = (self.all_dice[0], self.dice_pp, first_bet)
            cntt = 0
            opponent_jugo_bien = True
            while not done:
                if opponent_jugo_bien:
                    print("Es tu Turno")
                    action = player_human.choose_action(observation)
                    print("Has jugado :", action)
                observation_, _, done, info = self.step(action)
                if action != "Dudo" and action != "Calzo":
                    print(observation_)
                    #opponent_jugo_bien = self.verify_subida(observation_[2])
                    if opponent_jugo_bien:
                        print("El oponente jugó: ", observation_[2])
                observation = observation_
                cntt +=1
            new_dice = observation[1]
            cnt +=1
            print(info)
            print("Los dados de todos los jugadores eran:", [tuple(e) for e in self.all_dice])
            print(20*"----")
            if 0 in new_dice:
                Done = True
        print("Fin del Juego")
        pass
if __name__ == "__main__":
    dados1 = np.array([2,0,1,2,0,0]) # Tiene 2 ases, 1 Tren y 2 Cuadras
    dados2 = np.array([1,1,0,1,1,1]) # Tiene 1 As, 1 Tonto, 1 Cuadra, 1 Quina y 1 Sexta
    game_env = AmbienteCacho(2, [dados1, dados2])
    
    game_env = AmbienteCacho(2)
    game_env.step((1, "Tontos"), verbose = True)
    #game_env.step("Calzo", verbose = True)
    
    print(game_env.action_space)
    
    print(game_env.state_to_Tensor((np.array([1, 2, 2, 0, 0, 0]), [5, 5], (2, 'Tontos'))).shape)
    
    