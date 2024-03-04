#Se recomienda leer el resumen del proyecto para entender el algoritmo y algunos parametros

import math
import numpy as np
from sklearn.preprocessing import normalize
import random


#dist: array array -> Num
#recibe dos ciudades de la forma ciudad=[x,y] que representa sus coordenadas en el plano
#y calcula la distancia euclideana entre ellas
#ej city1=[0,3], city2=[4,0] retorna 5 

def dist(city1,city2):
    return math.sqrt((city1[1] - city2[1]) ** 2 + (city1[2] - city2[2]) ** 2)
    
#ACO: graph, int, int, num, num, num, num, bool -> Num
#graph: array que contiene ciudades de la forma [n-esima ciudad, ejex, ejey]
#ants: cantidad de hormigas, #iterations: cantidad de iteraciones
#alpha y beta: son dos parámetros que determinan la influencia relativa del rastro de hormonas y la información heurística 
#rho: simboliza la evaporación del rastro de feromonas
#Q: es un factor para la actualizacion de las feromonas
#variable_ant: tupla de bool, variable_ant[0]=True si se desea que varien las hormigas, en dicho caso 
#variable_ant[1] toma increasing o decreasing. variable_ant[1] para mantener las hormigas constantes
#Aco ejecuta n:iterations iteraciones del algoritmo y retorna el camino optimo del TSP
def ACO(graph, ants, iterations, alpha, beta, rho, Q, variable_ant = (False,None)):

    dist_opt = float("inf")
    camino_opt = []

    #Modalidad de hormigas variables
    if variable_ant[0] == True:
        
        # Versión con incrementos
        if variable_ant[1] == "increasing":
            iterator = [ants + i for i in range(iterations)] #se puede modificar
        
        # Version con decrementos
        elif variable_ant[1] == "decreasing":

            # Caso ants ≤ iterations
            if ants <= iterations:
                print("No es posible ejecutar ACO ants decreasing si ants ≤ iterations, se ejecutará el algoritmo de hormigas constantes en su lugar")
                iterator = [ants for i in range(iterations)]
            else:
                iterator = [ants - i for i in range(iterations)] #se puede modificar
    # Caso hormigas constantes
    else:
        iterator = [ants for i in range(iterations)]

    #inicializacion de feromonas
    pheromones = np.array([[1.0 for j in range(len(graph))] for i in range(len(graph))]) #se puede modificar
    
    #calculo distancias inversas (parametros theta heuristicos, se pueden modificar)
    inv_dist = [[0.0 for i in range(len(graph))] for j in range(len(graph))]
    for i in range(len(graph)):
        for j in range(len(graph)):
            if i==j:
                inv_dist[i][j] = 0.0
            else:
                inv_dist[i][j] = 1 / dist(graph[i], graph[j])

    #iteración del algoritmo
    for i in range(iterations):

        # Distribución inicial hormigas
        positions = [random.randint(0,len(graph) - 1) for j in range(iterator[i])]

        paths=[]
        distances = []

        #evaporacion de feromonas
        pheromones *= 1 - rho

        #busqueda de un camino para cada hormiga
        for ant in range(iterator[i]):

            path,distance = move_ant(positions[ant], graph, pheromones, inv_dist, alpha, beta)
            paths.append(path)
            distances.append(distance)

            if distance <= dist_opt:
                dist_opt = distance
                camino_opt = path

        distpath = [[paths[i], distances[i]] for i in range(len(distances))]
        
        # Actualización de feromonas al final de la iteración
        for k in range(len(paths)):
            for j in range(len(paths[k]) - 1):
                pheromones[paths[k][j]][paths[k][j + 1]] += Q / distances[k]
                pheromones[paths[k][j + 1]][paths[k][j]] += Q / distances[k]
            pheromones[paths[k][len(paths[k]) - 1]][paths[k][0]] += Q / distances[k]
            pheromones[paths[k][0]][paths[k][len(paths[k]) - 1]] += Q / distances[k]

    return camino_opt, dist_opt

#move_ant: int, grafo, num, num, num, num
#los parametros se explicaron arriba, position es la posicion que tiene la hormiga en el grafo
#esta funcion implementa el metodo para mover hormigas en cada iteracion
def move_ant(position, graph, pheromones, inv_dist, alpha, beta):

    visited = [position]

    # Loop de busqueda de transiciones
    while len(visited) < len(graph):
        
        transiciones = []
        posibles_transiciones = []

        for ciudad in graph:
            if ciudad[0] not in visited:

                transiciones += [pheromones[position][ciudad[0]] ** alpha * inv_dist[position][ciudad[0]] ** beta]
                posibles_transiciones.append(ciudad[0])

        transiciones = np.array(transiciones)

        next_node = random.choices(posibles_transiciones, weights = transiciones)[0]        
        visited.append(next_node) 
        position = next_node

    distance = sum([dist(graph[visited[i]], graph[visited[i - 1]]) for i in range(1, len(visited))])
    distance += dist(graph[len(visited) - 1], graph[0])

    return visited,distance            
            
