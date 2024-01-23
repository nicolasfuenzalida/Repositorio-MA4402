import numpy as np
import matplotlib.pyplot as plt
import random
from cifrador import *
from f1_energia import *
from frec import *
from FuncionesAuxiliares import *
from Lector import *
from CifradoMonoalfabetico import *
import math


#############################################################
letras = [['a',8.2],['b',1.5],['c',2.8],['d',4.3],['e',12.7],['f',2.2],['g',2],['h',6.1],['i',7],['j',0.15],['k',0.77],['l',4],['m',2.4],['n',6.7],['o',7.5],['p',1.9],['q',0.095],['r',6],['s',6.3],['t',9.1],['u',2.8],['v',0.98],['w',2.4],['x',0.15],['y',2],['z',0.074]]

ysa= np.matrix(fre)

bigramas = ysa[:,0] # bigramas de la lista de datos conseguidos



alfabeto = "abcdefghijklmnopqrstuvwxyz"

l = len(bigramas)
bi = []
for i in range(l):
    bi.append(bigramas[i,0])

for i in alfabeto:
    bi.append(i + ' ')

for i in alfabeto:
    bi.append(' ' + i)
###########################################################

#lo de arriba fue un intento inicial de generar la lista de bigramas, pero que fue desechada al encontrar una mejor opcion


def alfabetizador2(string: str):
    """
    toma un texto, y elimina todos los caracteres no pertenecientes al alfabeto, o 
    que sean el espacio y transforma mayusculas a minusculas
    """
    texto_2 = string.splitlines()
    texto_3 = ''.join(texto_2)
    f = ''.join(c for c in texto_3 if c.isalpha() or c.isspace())
    f_m = f.lower()
    return f_m


#de este texto, se obtuvo la estadistica de los digramas, para compararla
#con la estadistica de la base de datos
texto_1 = lector_mac('ACaseOfIdentity')


frec_u_ex = estadistica_digramas(texto_1,alfabeto+' ')
freqs_uex = np.array([int(k[1]) for k in frec_u_ex])
total1 = np.sum(freqs_uex)
freqs_rel_ex = np.array([t/total1 for t in freqs_uex])

#aqui, freqs_rel_ex son las frecuencias obtenidas de los digramas


def rotacion(cifrado):
    """
    dada una permutacion, esta funcion entrega
    con probabilidad 0.5 una rotacion hacia adelante, y con 
    0.5 una rotacion hacia atras
    """
    l = len(cifrado)
    new = []
    u = np.random.randint(0,2)
    if u ==0:
        new.append(cifrado[l-1])
        for i in range(l-1):
            new.append(cifrado[i])
        return new
    else:
        for i in range(1,l):
            new.append(cifrado[i])
        new.append(cifrado[0])
    return new


def vecinos_p(arra):
    """
    Funcion que genera un vecino de una permutacion
    en el grafo, donde el vecino es una permutacion
    que difiere en 2 terminos
    """
    per = random.sample(range(0,len(arra)),2)
    a = arra[per[0]]
    b = arra[per[1]]
    new = arra.copy()
    new[per[0]] = b
    new[per[1]] = a
    return new


def vecino2(clave,p):
    """
    Esta funcion, recibe una clave, y su periodo p
    y de manera equiprobable, entrega un vecino entre los definidos
    en vecinos_p, y las 2 rotaciones
    """
    pf = math.factorial(p)
    den = math.factorial(p-2)
    frac = pf/den
    x = np.random.randint(0,frac+3)
    if x ==0 or x==1:
        return rotacion(clave)
    else:
        return vecinos_p(clave)


def recortador(texto,i,j,pc):
    """
    toma un texto, una posicion inicial y una final, 
    y entrega un extracto del texo "alfabetizado" de largo
    j-i
    """
    if pc =='mac':
        t = lector_mac(texto)
    else:
        t = lector(texto)
    t2 = alfabetizador2(t)
    return t2[i:j]




def c_inicial(m):
    """
    Funcion que genera una permutacion inicial de largo m
    """
    l = list(range(1,m+1))
    random.shuffle(l)
    return l





def SA_per0(n, text, energia,m, clave_inicial,T0,tf):
    """
    n = numero de iteraciones
    text = mensaje a decodificar
    energia = funcion de energia
    m = largo de la permutacion, que se asume conocido
    clave_inicial = clave inicial, para generar un estado inicial
    T0 = temperatura inicial
    tf = factor de reduccion
    Esta funcion, es la que implementa SA como se vio en el paper de dimitrov y Gligorowski, y
    como se mostro en la presentacion
    """
    t_inicial = d_permutacion(clave_inicial,text)
    clave_actual = clave_inicial
    e_actual = energia(t_inicial)
    T=T0
    n_i = 0
    n_cambio = []
    energias = [energia(t_inicial)]
    for i in range(n):
        n_s = 0
        for i in range(100*m):
            clave = vecino2(clave_actual,m)
            t = alfabetizador(d_permutacion(clave,text))
            t = d_permutacion(clave,text)
            e = energia(t)
            delta = e-e_actual
            if delta <= 0 or np.exp(-(delta/T)) >= 0.5:
                clave_actual = clave
                e_actual = e
                n_s = n_s + 1
                n_i = n_i +1
                energias.append(e)
            else:
                n_i = n_i +1
                energias.append(e_actual)
            if n_s >=8*m:
                T = T*tf
                n_cambio.append(n_i)
                break
        if n_s == 0:
            n_cambio.append(n_i)
            break
        elif 0< n_s <10*m:
            T=T*tf
            n_cambio.append(n_i)
    return clave_actual, n_i, e_actual, energias, n_cambio


def fr_letras(letra,texto):
    i = 0
    largo =len(texto)
    for j in texto:
        if j == letra:
            i = i+1
    return i/largo
    





def tester0(m,texto,e1,cini,n,par):
    """
    Funcion para generar simulaciones
    m = largo permutacion
    texto = texto a cifrar
    e1 = funcion de energia a utilizar
    cini = clave con la que se cifra el texto
    n = numero de simulaciones
    par = parametros T0 y tf que se utilizaran
    Esta funcion permite realizar n simulaciones 
    para descifrar el texto encriptado con la clave cini
    """
    t_p = c_permutacion(cini,texto)
    print('clave cifrado:')
    print (cini)
    print('resultados:')
    print('clave propuesta | numero de iteraciones | energia final')
    plt.clf()
    camb = []
    for i in range(n):
        c_i1 = c_inicial(m)
        c = SA_per0(50,t_p,e1,m,c_i1,par[0],par[1])
        print(c[0:3])
        camb.append(c[4])
        N = np.linspace(0,c[1],c[1]+1)
        plt.plot(N,c[3], label = ('simulacion numero: ', i))
    c_m = 0

    for i in camb:
        if len(i)>c_m:
            c_m = len(i)
    c_f = np.zeros(c_m)

    #print (camb)

    for i in range(c_m):
        c_mi = 0
        tr = 0
        for j in camb:
            if len(j)>=i+1:
                c_mi = c_mi + j[i]
                tr = tr + 1
        c_mif = c_mi/tr
        c_f[i] =c_mif 

    #for i in range(c_m):
    #    plt.axvline(x = c_f[i], label = ('iteracion promedio del cambio numero: ', i), color = 'k', linewidth = 3)

    #esta parte de arriba, se utilizo para visualizar en que iteracion se realizaban aproximadamente los cambios de temperatura
    #pero al final, pensamos que no entregaba informacion util al final, pero si sirvio a la hora de ajustar parametros
    plt.axhline(y=e1(texto), label = 'energia texto sin cifrar', color = 'r')
    plt.title('Comportamiento energia vs iteraciones')
    plt.xlabel('iteraciones')
    plt.ylabel('energia')
    #plt.legend()

    plt.show()
    return



#claves para cifrar, y poder testear efectividad con ellas
ci2 = c_inicial(2)
ci3 = c_inicial(3)
ci4 = c_inicial(4)
ci5 = c_inicial(5)
ci6 = c_inicial(6)
ci7 = c_inicial(7)
ci8 = c_inicial(8)
ci9 = c_inicial(9)
ci10 = c_inicial(10)



#usa frecuencias de digramas empiricos, de la base de datos pero sin considerar el espacio, y es la formula sum(|empirica-texto|), requiere texto sin espacios
def energia_sa1(texto):
    frecuencias_dos_letra = estadistica_digramas(texto, alfabeto)
    freqs_digramas = np.array([k[1] for k in frecuencias_dos_letra])
    tot_digram = np.sum(freqs_digramas)
    freqs_rel_digram = np.array([t/tot_digram for t in freqs_digramas])
    dif_digram = np.abs(freqs_rel_digrama_universo - freqs_rel_digram)
    segundo_termino = np.linalg.norm(dif_digram, ord=1)
    final = segundo_termino
    return final

#0.5 al vecino o 0.5 a la rotacion

#usa datos de frecuencias experimentales y usa la formula sum(|experimental-texto|)
def energia_sa2(texto):
    frecuencias_dos_letra = estadistica_digramas(texto, alfabeto + ' ')
    freqs_digramas = np.array([k[1] for k in frecuencias_dos_letra])
    tot_digram = np.sum(freqs_digramas)
    freqs_rel_digram = np.array([t/tot_digram for t in freqs_digramas])
    dif_digram = np.abs(freqs_rel_ex - freqs_rel_digram)
    segundo_termino = np.linalg.norm(dif_digram, ord=1)
    final = segundo_termino
    return final

#funcion energia ponderada, sum(|experimental-texto|/experimental), usando datos experimentales 
def energia_sa3(texto):
    frecuencias_dos_letra = estadistica_digramas(texto, alfabeto + ' ')
    freqs_digramas = np.array([k[1] for k in frecuencias_dos_letra])
    tot_digram = np.sum(freqs_digramas)
    freqs_rel_digram = np.array([t/tot_digram for t in freqs_digramas])
    dif_digram = np.abs(freqs_rel_ex - freqs_rel_digram)
    divi = freqs_rel_ex +1
    sumatoria = dif_digram/divi
    segundo_termino = np.linalg.norm(sumatoria, ord=1)
    final = segundo_termino
    return  final

#funcoon energia ponderada, sum(|empirica-texto|/emppirica), usando datos empiricos, de la base de datos, requiere texto sin espacios
def energia_sa4(texto):
    frecuencias_dos_letra = estadistica_digramas(texto, alfabeto)
    freqs_digramas = np.array([k[1] for k in frecuencias_dos_letra])
    tot_digram = np.sum(freqs_digramas)
    freqs_rel_digram = np.array([t/tot_digram for t in freqs_digramas])
    dif_digram = np.abs(freqs_rel_digrama_universo - freqs_rel_digram)
    segundo_termino = np.linalg.norm(dif_digram, ord=1)
    divi = freqs_rel_digrama_universo +1
    sumatoria = dif_digram/divi
    segundo_termino = np.linalg.norm(sumatoria, ord=1)
    final = segundo_termino
    return final


########################################################
letras_ex = estadisticas_letras(texto_1,alfabeto)
freql_uex = np.array([int(k[1]) for k in letras_ex])
total2 = np.sum(freql_uex)
freqs_rel_le_ex = np.array([t/total2 for t in freql_uex])
########################################################

#lo de arriba, es anaologo a lo de las frecuencias de digramas de manera experimental
#pero con letras, pues pensamos podia haber sido de utilidad, al final fue desechado



#parametros usados para la simulacion, de la forma [T,tf], con
#T temperatura inicial, y tf factor de cambio, y donde 
#pari representa los parametros usados para la energia i

par1 = [2,0.89] 
par2 = [2,0.892275]
par3 = [2,0.910]
par4 = [2,0.89]



#texto que se analiza, y su version sin espacios(qwe4), para las versiones de energia que lo requieren
qwe2 = recortador('TheFiveOrangePips',0,1000,'mac')#en caso de correr en windows, utilizar 'pc' en lugar de 'mac'
qwe4 = alfabetizador(qwe2)


#para obtener los porcentajes vistos en la tabla, se realizo lo siguiente, 5 veces, para cada largo, es decir
# se genero una clave de cifrado, se aplico el metodo con cada energia 10 veces para esa clave, y luego se ponderaron
# los resultados de efectividad. Aqui, se ve el ejemplo con largo 5

#  tester0(5,qwe4,energia_sa1,ci5,10,par1)
#  tester0(5,qwe2,energia_sa2,ci5,10,par2)
#  tester0(5,qwe2,energia_sa3,ci5,10,par3)
#  tester0(5,qwe4,energia_sa4,ci5,10,par4)

#esto se ve asi, porque en principio, se corrio cada una en el terminal por separado para cada clave,  por miedo a que el pc no aguantara