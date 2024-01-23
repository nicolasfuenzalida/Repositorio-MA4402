from cifrado_nletras import *
import numpy as np
import matplotlib.pyplot as plt

"""
    Acá se implementará simmulated anneling. Se asumira que solo se tiene conocimiento\\
    de strings de largo hasta 2. Entonces para las estadísticas solo se harán uso de\\
    digramas.
"""

def simmulated_anneling_con_n(codificado, base, n, N_max, beta, semilla):
    """
        EFECTIVIDAD DE ESTA FUNCION NO ESTÁ COMPROBADA, SEGUIR BAJO SU PROPIO RIESGO\\
        Hace un simmulated anneling de un cifrado que se asume hace una\\
        biyección entre n-gramas consecutivos, dejando los últimos impolutos\\
        ######################\\
        Inputs:\\
        codificado: string largo con texto codificado\\
        base: string que da los caracteres con los cuales se cifró el texto\\
        n: largo de los n-gramas que se codificaron\\
        N_max: Número máximo de iteraciones\\
        beta: funcion de temperaturas inversas de SA\\
        semilla: entero para poder replicar el experimento\\
        ######################\\
        Outpus\\
        codigo: Intento de código que se usó para codificar la el mensaje\\
        Energias: numpy array de tamano N_max, trae energia de la codificacion en cada paso
    """
    b_act = beta(0)
    diccionario = diccionario_caracteres_n(base, n)
    digramas = len(diccionario)
    codigo = np.copy(diccionario)
    palabras_cod = palabras(codificado, base)
    estadisticas_digramas, monosilabos = estadistica_maestra_optimizado(palabras_cod, base, 2)
    estadisticas_alfabeto, vacio = estadistica_maestra_optimizado(palabras_cod, base, 1)
    vals_esta_digr = np.array([k[1] for k in estadisticas_digramas])
    vals_esta_alfa = np.array([k[1] for k in estadisticas_alfabeto])
    energia_actual = energia_desde_frecuencias(vals_esta_alfa, vals_esta_digr)
    np.random.seed(semilla)
    tamano_espacio = len(base) ** n
    tincadas_cambiazos = np.random.randint(tamano_espacio, size=(2, N_max))
    monedas = np.random.random(N_max)
    Energias = np.zeros(N_max)
    Energias[0] = energia_actual
    for k in range(N_max):
        print(k+1,"/",N_max)
        posible_nueva_biyeccion = np.copy(codigo)
        val_1 = tincadas_cambiazos[0,k]
        val_2 = tincadas_cambiazos[1,k]
        posible_nueva_biyeccion[val_1], posible_nueva_biyeccion[val_2] =\
              posible_nueva_biyeccion[val_2], posible_nueva_biyeccion[val_1]
        texto_con_nuevo_codigo = "".join([palabra+" " for palabra in palabras_cod])
        texto_con_nuevo_codigo = cifrado_largo(texto_con_nuevo_codigo, posible_nueva_biyeccion, base)
        nuevas_palabras = palabras(texto_con_nuevo_codigo, base)
        nuevas_frecuencias_dig, monosilabos = estadistica_maestra_optimizado(nuevas_palabras, base, 2)
        nuevas_freqs_alf, vacio = estadistica_maestra_optimizado(nuevas_palabras, base, 1)
        vals_esta_digr = np.array([k[1] for k in nuevas_frecuencias_dig])
        vals_esta_alfa = np.array([k[1] for k in nuevas_freqs_alf])
        posible_nueva_energia = energia_desde_frecuencias(vals_esta_alfa, vals_esta_digr)
        dif_ener = posible_nueva_energia - energia_actual
        umbral = 1 + (dif_ener >= 0) * (np.exp((-1) * dif_ener * b_act) - 1)
        if monedas[k] < umbral:
            energia_actual = posible_nueva_energia
            codigo = posible_nueva_biyeccion
            palabras_cod = nuevas_palabras
        b_act = beta(k+1)
        Energias[k] = energia_actual
    return codigo, Energias

semilla = 100
diccionario = diccionario_caracteres_n(alfabeto, 2)
clave = permutador_general(diccionario, semilla)
codificado = cifrado_largo(test, clave, alfabeto)
def beta(x):
    return x ** 2
N_max = 1000
intento, energias_obt = simmulated_anneling_con_n(codificado, alfabeto, 2,N_max, beta, semilla)
print(cifrado_largo(codificado, intento, alfabeto))
plt.plot(energias_obt)
plt.axhline(energia(test))
plt.show()
# print(intento)