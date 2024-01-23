import numpy as np
from frec import fre
y = np.matrix(fre)

bigramas = y[:,0] # bigramas de la lista de datos conseguidos
cantidad = []
for i in range(len(fre)):
    cantidad.append(fre[i][1])
#cantidad = [int(x) for x in y[:,1]] 
cantidad_total = sum(cantidad)
frecuencias = [(x/cantidad_total) for x in cantidad] # estos son los datos transformados a frecuencias

def frecuencia_bigrama(b, t):
    """
    b: bigrama que se busca calcular su frecuencia
    t: texto en el cual se analizara la frecuencia
    """
    l = len(t)
    c = 0
    for i in range(l-1):
        if t[i] == b[0] and t[i+1] == b[1]:
            c = c + 1
    frec = c / (l - 1) 
    return frec

def energia_1i(tex,beta):
    """
    tex: es el texto cifrado
    """
    sum = 0
    l = len(frecuencias)
    for i in range(l):
        sum = sum + np.abs(frecuencias[i] - frecuencia_bigrama(bigramas[i,0],tex))
    return beta * sum












