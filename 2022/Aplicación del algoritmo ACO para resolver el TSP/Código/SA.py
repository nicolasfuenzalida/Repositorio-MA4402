#En este archivo se imnplementa el algoritmo de simulated annealing para resolver el TSP
import numpy as np

#omega: array, array -> Num
#recibe una lista sigma que representa un camino del TSP, por ej [2,1,3] representa 2 -> 1 -> 3 -> 2; y recibe una matriz
#D de distancias entre las ciudades (los indices del camino representan ciudades en un plano), 
#para asi retornar el valor omega del camino (largo total del viaje)
def omega(sigma,D):
    #calcula el omega del camino sigma
    d = 0
    for i,_ in enumerate(sigma):
        if i+1!= len(sigma):
            d += D[sigma[i]][sigma[i+1]] 
        else :
            d += D[sigma[i]][sigma[0]]
    return d

#sucesion beta de temperaturas inversas para el algoritmo SA, con C una constante heuristica
#Se puede cambiar por otra sucesion que converga a infinito (en n->inf), puede cambiar significativamente la eficiencia 
beta = lambda n,C : 1/C*np.log(n+np.exp(1)) #beta logaritmico #se puede modificar


#Markov: array, int, array, func -> array
#se simula una transcision del S.A., sigma es el camino actual en la iteracion n-esima, D es la matriz de distancias del grafo
#y bn es una funcion de temperaturas inversas que converge a infinito, retorna un array que es el camino en la iteracion n+1
#dado por la transicion del S.A

def Markov(sigma, n, D, bn = beta):
    # calcula el paso enesimo de la cadena de markov

    C = (len(sigma)-1)*4*np.sqrt(2)
    beta = bn(n,C) # calculamos el beta

    # asignamos el Xn a sigma
    Xn = sigma

    # obtenemos una permutación de Xn (sin cambiar el primer vértice)
    u = np.random.randint(low = 1, high = len(sigma), size = 2)
    while u[0] == u[1]:
        u = np.random.randint(low = 1, high = len(sigma), size = 2)
    
    # Obtenemos tau permutando Xn
    tau = np.copy(Xn)
    tau[u[0]], tau[u[1]] = Xn[u[1]], Xn[u[0]]
    
    # calculamos X_{n+1} con el criterio pedido
    U = np.random.uniform(0,1)
    crit = np.exp(-beta*(omega(tau,D)-omega(Xn,D)))
    if U <= min(1, crit):
        Xn = tau
    return Xn
