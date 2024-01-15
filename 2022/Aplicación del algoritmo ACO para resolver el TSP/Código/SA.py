import numpy as np

def omega(sigma,D):
    #calcula el omega del camino sigma
    d = 0
    for i,_ in enumerate(sigma):
        if i+1!= len(sigma):
            d += D[sigma[i]][sigma[i+1]] 
        else :
            d += D[sigma[i]][sigma[0]]
    return d

beta = lambda n,C : 1/C*np.log(n+np.exp(1)) #beta logaritmico

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
