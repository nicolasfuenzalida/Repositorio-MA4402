import numpy as numpy
import matplotlib.pyplot as plt
from frec import *
from Lector import *
from FuncionesAuxiliares import *
from cifrador import *
from time import time

def creador_clave_aleatoria(base: str, semilla: int) -> str:
    """
        Crea una clave aleatoria a partir de una base y semilla\\
        Devuelve un string para este fin.
        ######################\\
        Inputs:\\
        base: base de caracteres que se van a reordenar para crear una clave\\
        semilla: entero que sirve para poder replicar el numero aleatorio\\
        ######################\\
        Outputs:\\
        final: string de igual largo y mismos caracteres que la base, pero reordenados\\
               de forma aleatoria.
    """
    assert type(base) == str
    lista = np.array(list(base))
    largo = len(lista)
    np.random.seed(semilla)
    randomized = np.random.choice(lista, size=largo, replace=False)
    final = ""
    for k in randomized:
        final += k
    return final


base_usar = alfabeto
clave_de_prueba = creador_clave_aleatoria(base_usar, 100)

"""
Con esto ya tenemos una clave creada de forma aleatoria, pero conocida
Ahora vamos a aplicarla a un texto traducido
"""

test = lector("SpeckledBand")
prueba_cifrado = cifrador(clave_de_prueba, test, base=base_usar)

"""
Este será nuestro sujeto de pruebas que intentaremos decifrar para el cifrado\\
monoalfabetico
"""

def estadisticas_letras(texto: str,base: str) -> np.ndarray:
    """
        Toma una un texto y una base de caracteres\\
        Y devuelve una estadistica de los usos de dichos caracteres\\
        Como una lista de la forma [...,[elemento base, frecuencia],...]\\
        ######################\\
        Inputs:\\
        texto: String del cual se sacará la estadística\\
        base: String con caracteres de los cuales se contarán las frecuencias\\
        ######################\\
        Outputs:\\
        final: numpy array de tamano (base)x2, con la primera coordenada siendo el\\
               caracter a contar, y la segunda la frecuencia de aparición en el texto\\
    """
    tamano = len(base)
    final = np.array([["a",0]] * tamano, dtype=object)
    actual = 0
    for k in base:
        cantidad = texto.count(k)
        final[actual] = [k, cantidad]
        actual += 1
    return final

def crearListaDigramas(base: str) -> np.ndarray:
    """
        Crea lista de digramas de caracteres en una base\\
        ######################\\
        Inputs\\
        base: string de elementos que se considerarán para los digramas\\
        ######################\\
        Outputs:\\
        lista_digramas: numpy array de tamano (base ** 2), es una lista de\\
                        todos los digramas que se pueden armar con caracteres\\
                        en la base
    """
    tamano = len(base)
    lista_digramas = np.array(["aa"] * ((tamano) ** 2))
    contador_actual = 0
    for j in base:
        for k in base:
            digrama = j+k
            lista_digramas[contador_actual] = digrama
            contador_actual += 1
    return lista_digramas

def estadistica_digramas(texto: str, base: str) -> np.ndarray:
    """
        Desde un texto consistente solo en elementos de la base\\
        se crea una lista de frecuencias de los digramas.\\
        ######################\\
        Inputs:\\
        texto: Texto desde el cual se sacará la estadística, es un string\\
        base: String de caracteres que se considerarán para crear los digramas\\
        ######################\\
        Outputs:\\
        final: numpy array de dimensiones (base ** 2) x2, con la primera coordenada\\
               siendo el digrama con su frecuencia contada, y la segunda su frecuencia\\
    """
    digramas = crearListaDigramas(base)
    tamano = len(base)
    final = np.array([[k,0] for k in digramas], dtype=object)
    letra_actual = texto[0]
    pos_letra_actual = base.find(letra_actual)
    for s in range(len(texto)-1):
        letra_siguiente = texto[s+1]
        digrama = letra_actual + letra_siguiente
        if digrama not in digramas:
            continue
        pos_letra_siguiente = base.find(letra_siguiente)
        pos_digrama = pos_letra_actual * tamano + pos_letra_siguiente
        final[pos_digrama,1] += 1
        letra_actual = letra_siguiente
        pos_letra_actual = pos_letra_siguiente
    return final


def ordenar_par(lista: list) -> np.ndarray:
    """
        Toma una lista de frecuencias desordenada\\
        Y la ordena de forma lexicografica\\
        ######################\\
        Inputs:\\
        lista: Lista de frecuencias que corresponde a una lista de python\\
               de tamano (tamano)x2, es decir, es de tamano variable\\
               con la primera coordenada siendo el caracter o secuencia de caracteres\\
               que cuenta, y la segunda cuantas veces las cuenta\\
        ######################\\
        Outputs:\\
        reordenado: numpy array de igual tamano que lista, pero que esta ordenado\\
                    por orden lexicográfico según su primera coordenada (los caracteres a contar)\\
    """
    lista = np.array(lista)
    nombres = np.array([k[0] for k in lista])
    inds = nombres.argsort()
    reordenado = lista[inds]
    return reordenado

frecuencias_universo = ordenar_par(fre)
freqs_universo = np.array([int(k[1]) for k in frecuencias_universo])
total = np.sum(freqs_universo)
freqs_rel_digrama_universo = np.array([t/total for t in freqs_universo])

freqs_rel_letras_universo = np.array([s[1]/100 for s in letras])
# print(freqs_rel_universo)
# print(ordenar_par(fre))
orden_normas = 1

def fitness(texto: str) -> float:
    """
        Dado un texto, que es un string\\
        Devuelve el stiffness del mismo\\
        Correspondiente a la fórmula dada en el paper de\\
        Carter Magoc\\
        ######################\\
        Input:\\
        texto: String desde el cual se sacarán las estadísticas\\
        ######################\\
        Output:\\
        fin: Corresponde a la función fitness del resumen (de Carter Magoc)
    """
    frecuencias_una_letra = estadisticas_letras(texto, alfabeto)
    frecuencias_dos_letra = estadistica_digramas(texto, alfabeto)
    freqs_letras = np.array([k[1] for k in frecuencias_una_letra])
    freqs_digramas = np.array([k[1] for k in frecuencias_dos_letra])
    resultado = energia_desde_frecuencias(freqs_letras, freqs_digramas)
    fin = 1 - resultado
    return fin

def energia_desde_frecuencias(freqs_letras: np.ndarray, freqs_digramas: np.ndarray) -> float:
    """
        Calcula la energía en base a las frecuencias\\
        ######################\\
        Inputs:\\
        freqs_letras: entero, corresponde a la frecuencia absoluta de aparición\\
                      de letras.\\
        freqs_digramas: entero, corresponde a la frecuencia absoluta de aparición\\
                        de digramas.\\
        ######################\\
        Outputs:\\
        resultado: float, corresponde a la función 1-fitness, con fitness\\
                   presentada en el resumen.\\
    """
    tot_letras = np.sum(freqs_letras)
    tot_digram = np.sum(freqs_digramas)
    freqs_rel_letras = np.array([t/tot_letras for t in freqs_letras])
    freqs_rel_digram = np.array([t/tot_digram for t in freqs_digramas])
    dif_letras = np.abs(freqs_rel_letras - freqs_rel_letras_universo)
    dif_digram = np.abs(freqs_rel_digrama_universo - freqs_rel_digram)
    primer_termino = np.linalg.norm(dif_letras, ord=orden_normas)
    segundo_termino = np.linalg.norm(dif_digram, ord=orden_normas)
    final = ((primer_termino + segundo_termino)/4)
    fitness = 1 - final ** 8
    resultado = 1 - fitness
    return resultado

def energia(texto: str) -> float:
    """
        Define la energía como 1-fitness\\
        ######################\\
        Inputs:\\
        texto: String del que se sacará la energía\\
        ######################\\
        Outputs:\\
        en: float, es 1-fitness(texto).
    """
    en = 1-fitness(texto)
    return en

def calculo_nuevas_frecuencias(freq_let: np.ndarray, freq_digr: np.ndarray, intercambio: np.ndarray) ->[np.ndarray, np.ndarray]:
    """
        Calcula las nuevas frecuencias si se hizo un intercambio\\
        Para esto toma dos frecuencias, una de unigramas y otra de digramas\\
        además de un vector de tamaño dos que trae dos enteros de valor menor\\
        o igual al largo de freq_let-1, que representa las posiciones a intercambiar\\
        Devuelve otro set de frecuencias ad hoc. Se asume que el orden es lexicográfico\\
        Esta función asume que las letras intercambiadas no están en el borde.\\
        ######################\\
        Inputs:\\
        freq_let: Representa las frecuencias de aparición de letras del alfabeto\\
                  Puede ser tanto una lista de dimensiones (base) x 2, representando\\
                  los unigramas (primera coordenada) y su respectiva frecuencia, o bien\\
                  sólo la frecuencia (la segunda coordenada). En el caso del primer caso, \\
                  va a devolver de forma desordenada las letras.\\
        freq_dig: Representa la frecuencia de aparición de digramas. Esto puede ser tanto una lista\\
                  de dimensiones (base ** 2) x2, representando los digramas en la primera coordenada\\
                  y la frecuencia en la segunda, o bien sólo la frecuencia. En ambos casos se asume\\
                  un orden lexicográfico con respecto al orden de las frecuencias relativas.
    """
    assert np.shape(intercambio)[0] == 2
    assert intercambio[0] >= 0 and intercambio[0] <= np.shape(freq_let)[0]
    assert intercambio[1] >= 0 and intercambio[1] <= np.shape(freq_let)[0]
    val_1 = intercambio[0]
    val_2 = intercambio[1]
    tamano = len(freq_let)
    new_freq_let = np.copy(freq_let)
    new_freq_digr = np.copy(freq_digr)
    for i in range(tamano):
        if i == val_1 or i == val_2:
            continue
        indice_1_primera_cifra = tamano*i + val_1
        indice_2_primera_cifra = tamano*i + val_2
        new_freq_digr[[indice_1_primera_cifra, indice_2_primera_cifra]] = \
            new_freq_digr[[indice_2_primera_cifra, indice_1_primera_cifra]]
        indice_1_segunda_cifra = tamano*val_1 + i
        indice_2_segunda_cifra = tamano*val_2 + i
        new_freq_digr[[indice_1_segunda_cifra, indice_2_segunda_cifra]] = \
            new_freq_digr[[indice_2_segunda_cifra, indice_1_segunda_cifra]]
    new_freq_digr[[val_1, val_2]] = new_freq_digr[[val_2, val_1]]
    new_freq_digr[[val_1, val_1]] = new_freq_digr[[val_2, val_2]]
    new_freq_let[[val_1, val_2]] = new_freq_let[[val_2, val_1]]
    return new_freq_let, new_freq_digr


def SimmulatedAnneling(texto_cifrado: str, inicio: str, N_max: int, beta: function,semilla: int, base=alfabeto) -> [str, np.ndarray]:
    """
            NO ESTÁ OPTIMIZADO, USAR BAJO SU PROPIO RIESGO
        Dado un texto cifrado, se hace simmulated anneling\\
        para hallar el cifrado original. Para esto parte con\\
        Un intento de clave (inversa) inicial, un número máximo de pasos,\\
        una función de betas en función del tiempo discreto y una base\\
        de símbolos a permutar\\
        ######################\n
        Inputs:\\
        texto_cifrado: string, es el texto a decifrar\\
        inicio: string, para este proyecto consiste solo en letras minúsculas\\
        N_max: int, es el número de pasos máximo\\
        beta: funcion, es la función de los beta a utilizar durante el algoritmo\\
        semilla: Es una semilla para poder calcular los valores aleatorios del método\\
        base: string del mismo largo que inicio, trae los mismos caracteres. Usualmente sera alfabeto\\
        ######################\n
        Outputs:\\
        clave_inversa_actual: Es un string que representa LA INVERSA de la clave del código\\
                              con la que se codificó el texto\\
        energias: Numpy array con las energías del simmulated anneling en cada paso de la simulación\\
                  tiene largo N_max
    """
    b_act = beta(0)
    intento_decifrado_actual = cifrador(inicio, texto_cifrado, base=base)
    energia_actual = energia(intento_decifrado_actual)
    clave_inversa_actual = inicio
    np.random.seed(semilla)
    lista_aleatorios = np.random.random(N_max)
    tamano = len(base)
    intercambios = np.random.randint(tamano, size=(N_max, 2))
    energias = np.zeros(N_max)
    energias[0] = energia_actual
    for k in range(N_max):
        posible_nueva_clave = vecinos(clave_inversa_actual, intercambios[k,:])
        posible_intento_decifrado = cifrador(posible_nueva_clave,texto_cifrado,base)
        nueva_energia = energia(posible_intento_decifrado)
        dif_ener = nueva_energia - energia_actual
        umbral = 1 + (dif_ener >= 0) * (np.exp((-1) * dif_ener * b_act) - 1)
        if lista_aleatorios[k] < umbral:
            clave_inversa_actual = posible_nueva_clave
            energia_actual = nueva_energia
            intento_decifrado_actual = posible_intento_decifrado
        b_act = beta(k+1)
        energias[k] = energia_actual
        print(k+1,"/",N_max)
    return clave_inversa_actual, energias

def SimmulatedAnnelingOptimizado(texto_cifrado: str, N_max: int, beta: function, semilla: int, base=alfabeto) -> [str, np.ndarray]:
    """
        Dado un texto cifrado, se hace simmulated anneling\\
        para hallar el cifrado original. Para esto parte con\\
        la base como intento de  clave, un número máximo de pasos,\\
        una función de betas en función del tiempo discreto y una base\\
        de símbolos a permutar\\
        ######################\n
        Inputs:\\
        texto_cifrado: string, es el texto a decifrar\\
        N_max: int, es el número de pasos máximo\\
        beta: funcion, es la función de los beta a utilizar durante el algoritmo\\
        semilla: Es una semilla para poder calcular los valores aleatorios del método\\
        base: string del mismo largo que inicio, trae los mismos caracteres. Usualmente sera alfabeto\\
        ######################\n
        Outputs:\\
        clave_actual: string, es la clave con la que se asume que se CODIFICÓ el texto, \\
                      posee el largo de la base\\
        energias: numpy array con las energias para usos de análisis posterior\\
                  es de largo N_max
    """
    b_act = beta(1)
    clave_actual = base
    intento_cifrado_inicial = cifrador(alfabeto, texto_cifrado, base=base)
    energia_actual = energia(intento_cifrado_inicial)
    letras_act = estadisticas_letras(intento_cifrado_inicial,base=base)
    digramas_act = estadistica_digramas(intento_cifrado_inicial,base=base)
    freq_letras_act = np.array([k[1] for k in letras_act])
    freq_digramas_act = np.array([k[1] for k in digramas_act])
    np.random.seed(semilla)
    lista_aleatorios = np.random.random(N_max)
    tamano = len(base)
    intercambios = np.random.randint(tamano, size=(N_max, 2))
    energias = np.zeros(N_max)
    energias[0] = energia_actual
    for k in range(N_max):
        nuevas_freqs_let, nuevas_freqs_dig = calculo_nuevas_frecuencias(freq_letras_act, \
                                                  freq_digramas_act, intercambios[k,:])
        nueva_energia = energia_desde_frecuencias(nuevas_freqs_let, nuevas_freqs_dig)
        dif_ener = nueva_energia - energia_actual
        umbral = 1 + (dif_ener >= 0) * (np.exp((-1) * dif_ener * b_act) - 1)
        if lista_aleatorios[k] < umbral:
            energia_actual = nueva_energia
            freq_letras_act = nuevas_freqs_let
            freq_digramas_act = nuevas_freqs_dig
            nueva_clave = vecinos(clave_actual, intercambios[k,:])
            clave_actual = nueva_clave
        b_act = beta(k+1)
        energias[k] = energia_actual
    return clave_actual, energias

"""
    Se puede probar cuanto se demora el simmulated anneling sin la optimización senalada en la presentación\\
    Es bastante más lenta, pese a llegar exactamente a los mismos resultados.
"""
# ti = time()
# solucion_tentativa, energias = SimmulatedAnneling(prueba_cifrado, alfabeto, N_max, lambda x: x**2, 100, base=alfabeto)
# tf = time()
# intento_de_solucion = cifrador(solucion_tentativa, prueba_cifrado)
# print(intento_de_solucion)
# print("La demora de la función es ", tf-ti, "(s)")
"""
    Aplicado de simmulated anneling, hace todo entre medio, se puede seleccionar una semilla determinista\\
    si se quiere incluso. También hace los gráficos, muestra el texto decodificado (es el print(decodificado) del final)\\
    , y muestra la clave final tambien\\
"""
# alpha = 2
# N_max = 10_000
# def beta(x):
#     return x** alpha
# energia_base = energia(test)

# semilla = np.random.randint(10000)

# ti = time()
# solucion_optimizada, energias_opt = SimmulatedAnnelingOptimizado(prueba_cifrado, N_max, beta, semilla, base=alfabeto)
# tf = time()
# print(prueba_cifrado)
# inversa = crearInversa(solucion_optimizada, alfabeto)
# traducido = cifrador(inversa, prueba_cifrado)
# # print(traducido)
# print("La demora del caso optimizado es ", tf-ti, "(s)")
# print("La energia final es ", energias_opt[-1])
# print("La energia del original es ",energia_base)
# print("Clave obtenida es ", solucion_optimizada)
# print("Clave original es ", clave_de_prueba)

# fig, ax = plt.subplots(1,1, figsize=(6,6))


# ax.plot(energias_opt)
# ax.set_yscale("log")
# ax.axhline(energia_base, color="red", label="Energía caso inicial")
# ax.set_xlim(0,N_max)
# ax.tick_params(labelsize=15)
# ax.set_title("Optimizado", fontsize=30)
# ax.set_xlabel("Número de Iteraciones", fontsize=25)
# ax.set_ylabel("Energía", fontsize=25)
# fig.tight_layout()
# # fig.savefig("Grafico_Energias_Monoalfabetico.png")
# plt.show()