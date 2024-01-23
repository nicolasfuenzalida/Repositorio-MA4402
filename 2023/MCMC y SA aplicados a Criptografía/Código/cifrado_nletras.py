import numpy as np
from CifradoMonoalfabetico import *
import matplotlib.pyplot as plt

"""
    Este documento fue de un intento (fallido) de extensión de la metodología usada\\
    para el cifrado monoalfabético al cifrado polialfabético. Está incompleto y hay funciones que\\
    no se usan del todo\\
"""

def diccionario_caracteres_n(base: str,n: int) -> np.ndarray:
    """
        Dada una base de caracteres y un numero natural, crea\\
        una lista de todas las combinaciones de letras con esos n\\
        caracteres.\\
        ##############################\\
        Inputs:\\
        base: string, es una lista de caracteres\\
        n: número natural, largo de strings que se van a enciclopediar\\
        ##############################\\
        Outputs:\\
        final: Lista de numpy que trae todos los strings de largo n con caracteres en la base\\
               Está ordenada en orden lexicográfico según el orden original en base.
    """
    tamano = len(base)
    final = np.array(["a"*n] * (tamano ** n), dtype=str)
    for i in range(tamano ** n):
        coordenadas = np.array([i//(tamano ** j)%tamano for j in reversed(range(n))])
        string_actual = ["a"]*n
        for k in range(n):
            letra = base[coordenadas[k]]
            string_actual[k] = letra
        string_actual = "".join(string_actual)
        final[i] = string_actual
    return final

# Ahora la idea sería hallar de forma abstracta la inversa de una función biyectiva
# Entonces la forma de hacer una biyección no será con un string, sino con una lista
# Donde cada elemento será una tupla de largo determinado por la funcion anterior

def permutador_general(lista: np.ndarray, semilla: int) -> np.ndarray:
    """
        Dada una lista de elementos que vamos a permutar\\
        Devuelve una permutación aleatoria de esta lista\\
        Es idéntica a la función crear_clave_aleatoria de CifradoMonoalfabetico\\
        Pero devuelve una lista en vez de un string\\
        ##############################\\
        Inputs:\\
        lista: Lista de numpy, es el conjunto de elementos a permutar\\
        semilla: Semilla del random que vamos a usar, para controlar la aleatoriedad\\
        ##############################\\
        Outputs:\\
        permutado: Lista del mismo largo que la original
    """
    largo = len(lista)
    np.random.seed(semilla)
    permutado = np.random.choice(lista, size=largo, replace=False)
    return permutado

# Para el primer intento, veremos si se puede deducir algo de valor mediante
# la estadística con letras y digramas. Sin embargo, primero deberemos armar una funcion
# Que haga cifrados de a bloques de n caracteres.

def cifrado_largo(texto: str, clave: str, base: str) -> str:
    """
        Considerando un texto a codificar con una clave que consistirá\\
        en ir modificando el texto vía permutaciones de conjuntos de n caracteres\\
        Devuelve otro texto con dichas codificaciones.\\
        Para optimizar el proceso, se asume implícitamente que clave es una biyección\\
        Desde el mismo conjunto de elementos pero en orden lexicográfico, esto se usa\\
        para realizar de forma más rápida este cálculo. El orden lexicográfico es con respecto\\
        a base.\\
        ##############################\\
        Inputs:\\
        texto: string, es el texto a codificar\\
        clave: es un reordenamiento del conjunto de todos los strings \\
               de un mismo largo de elementos de base\\
        base: caracteres que se van a usar para cifrar
        ##############################\\
        Outputs:\\
        codificacion_final: string ya codificado
    """
    n = len(clave[0])
    tamano = len(base)
    assert len(clave) == tamano ** n # Consistencia para definir la clave
    codificacion_final = ""
    numero_letras_actual = 0
    a_permutar = ""
    texto_hasta_ahora = ""
    coords_caracteres = np.zeros(n)
    for k in range(len(texto)):
        caracter_actual = texto[k]
        texto_hasta_ahora += caracter_actual
        if caracter_actual in base:
            numero_letras_actual += 1
            a_permutar += caracter_actual
            coord_caract = base.index(caracter_actual)
            coords_caracteres[numero_letras_actual-1] = coord_caract
        if numero_letras_actual == n:
            cual_soy = np.sum([coords_caracteres[i] * tamano ** (n-i-1) for i in range(n)], dtype=int)
            cual_sere = clave[cual_soy]
            nuevo = ""
            i = 0
            for h in texto_hasta_ahora:
                if h in base:
                    nuevo += cual_sere[i]
                    i += 1
                else:
                    nuevo += h
            codificacion_final += nuevo
            texto_hasta_ahora = ""
            numero_letras_actual = 0
            a_permutar = ""
            coords_caracteres = np.zeros(n)
    codificacion_final += texto_hasta_ahora
    return codificacion_final

# texto_prueba = "abbaab baab ba"
# print(texto_prueba)
# alfabeto_prueba = "ab"
# a_codificar = diccionario_caracteres_n(alfabeto_prueba, 3)
# print(a_codificar)
# clave = permutador_general(a_codificar, np.random.randint(1000))
# print(clave)
# print(cifrado_largo(texto_prueba, clave, alfabeto_prueba))

"""
    Ahora, esto dará lugar a la formulación más general de una decodificación,\\
    en contrar una función inversa en un conjunto finito, que se aplica de forma\\
    reiterada sobre una concatenaciaón de caracteres a decodificar.\\
    Tomaremos la función que mide los digrafos y letras en cada caso,
    desde el archivo CifradoMonoalfabetico\\
"""

def estadistica_maestra(string: str, base: str, n: int) -> np.ndarray:
    """
        Toma un string, una base (que tambien es un string), \\
        y un número natural n, y luego devuelve una lista de listas\\
        que trae la estadística de todos los substrings de un largo \\
        determinado. Si se pone base=ab y n=3, entonces se devuelven las\\
        frecuencias de los string aaa, aab, aba, abb, baa, bab, bba, bbb\\
        en el string "string".\\
        Si hay separacion de caracteres, entonces se ignoran y tras esos\\
        caracteres fuera de la base se resetea la cuenta\\
        ##############################\\
        Inputs:\\
        string: string a codificar\\
        base: string con carácteres a contar\\
        n: Tamaño de los substrings\\
        ##############################\\
        Output:\\
        final: numpy array de dimensión (base ** n)x2, en la primera coordenada\\
               trae el string que se está contando, en la segunda, tiene su frecuencia\\
               en "string".
    """
    tamano = len(base)
    endpoint = len(string)
    ti = time()
    diccionario = diccionario_caracteres_n(base, n)
    tf = time()
    print("Crear el diccionario se demora ",tf-ti, "(s)")
    ti = time()
    final = np.array([[elemento,0] for elemento in diccionario], dtype=object)
    tf = time()
    print("Crear la base para la estadistica se demora ", tf-ti, "(s)")
    i = 0
    str_actual = string[0:n]
    coordenadas = np.array([base.find(str_actual[k]) for k in range(n)])
    while -1 in coordenadas:
        i += np.where(coordenadas == -1)[0][0] + 1
        if i > endpoint - n:
            return final
        str_actual = string[i:i+n]
        coordenadas = np.array([base.find(str_actual[k]) for k in range(n)])
    while i < endpoint-n:
        posicion_actual = np.sum([coordenadas[j] * tamano ** (n-j-1) for j in range(n)], dtype=int)
        final[posicion_actual,1] += 1
        i+=1
        str_actual = string[i:i+n]
        siguiente = str_actual[-1]
        if siguiente not in base:
            i+=n
            if i > endpoint - n:
                return final
            str_actual = string[i:i+n]
            coordenadas = np.array([base.find(str_actual[k]) for k in range(n)])
            while -1 in coordenadas:
                i += np.where(coordenadas == -1)[0][0] + 1
                if i > endpoint - n:
                    return final
                str_actual = string[i:i+n]
                coordenadas = np.array([base.find(str_actual[k]) for k in range(n)])
        else:
            coordenadas = [k for k in coordenadas[1:]] + [base.find(siguiente)]
    if -1 not in coordenadas:
        posicion_actual = np.sum([coordenadas[j] * tamano ** (n-j-1) for j in range(n)], dtype=int)
        final[posicion_actual,1] += 1
    return final


# ti = time()
# contador = estadistica_maestra(test, alfabeto, 2)
# tf = time()
# print("La demora total fue de ", tf-ti, "(s)")
# cuando_no_zero = np.array([a for a in contador if a[1] != 0])
# print(len(cuando_no_zero), " son no nulos, entre ",len(contador))

def multi_vecinos(biyeccion: np.ndarray, intercambio: np.ndarray) -> np.ndarray:
    """
        Toma una biyección representada por una lista\\
        y un intercambio, representado por una lista de tamano 2\\
        que representa qué elementos de la biyección se van a intercambiar\\
        Entonces devuelve otra biyeccion con esas posiciones intercambiadas\\
        con respecto a la biyección inicial\\
        Esta función es un salto de un vecino a otro en el grafo armado.
        ##############################\\
        Inputs:\\
        biyeccion: numpy array, idealmente todos los elementos de la lista\\
                   son distintos entre sí, sin embargo la función no lo impone\\
        intercambio: lista o similar, sirve cualquier elemento que se puede llamar como lista\\
                     es de largo dos y representa los elementos que se van a intercambiar\\
                     esa representación es mediante un número entero que es la posición en el array.\\
        ##############################\\
        Outputs:\\
        nuevo: numpy array casi idéntico a biyección, trae sólo los valores en posición "intercambio"\\
               con valores distintos, intercambiados.
    """
    largo = len(biyeccion)
    assert len(intercambio) == 2
    val_1, val_2 = intercambio
    assert 0 <= val_1 and val_1 <= largo-1
    assert 0 <= val_2 and val_2 <= largo-1
    nuevo = np.copy(biyeccion)
    nuevo[val_1], nuevo[val_2] = nuevo[val_2], nuevo[val_1]
    return nuevo

def palabras(string: str, base: str) -> np.ndarray:
    """
        Dado un string (string) y una base (también en formato string)\\
        de caracteres a considerar, esta función devuelve una lista de substrings\\
        compuestos únicamente por elementos de la base de forma contigua.\\
        Si base fuese el alfabeto, esto serían las palabras de toda la vida, separadas\\
        por espacios y comas.\\
        ##############################\\
        Inputs:\\
        string: string a sacarle las palabras\\
        base: string de caracteres válidos para el análisis\\
        ##############################\\
        Outputs:\\
        palabras: numpy array, lista de strings que consiste en las palabras de "string"
    """
    if base == alfabeto:
        new_string = string.lower()
    else:
        new_string = string
    palabras = []
    palabra_actual = []
    for k in new_string:
        letra_actual = k
        if k in base:
            palabra_actual += [k]
        else:
            palabras += ["".join(palabra_actual)]
            palabra_actual = []
    if not palabra_actual == []:
        palabras += ["".join(palabra_actual)]
    return np.array(palabras)


def estadistica_maestra_optimizado(palabras: list, base: str, n: int) -> [np.ndarray, np.ndarray]:
    """
        Dado un listado de palabras, una base de caracteres, y un número n de largo a contar\\
        esta funcion devuelve la estadistica de substrings con caracteres de la base en cada palabra\\
        a lo largo de la lista de palabras, además, devuelve otra lista de palabras de largo menor a n\\
        para asi no perder informacion al respecto\\
        ##############################\\
        Inputs:\\
        palabras: Lista de strings.\\
        base: string con caracteres a considerar\\
        n: largo de caracteres a considerar.
        ##############################\\
        Outpus:\\
        final_contador: numpy array de tamano (base ** n) x 2, que cuenta la aparición de\\
                        substrings de largo n en cada PALABRA. La primera coordenada es el caracter\\
                        la segunda es la frecuencia\\
        contador_palabras_cortas: numpy array con strings correspondientes a palabras de largo menor a n
    """
    assert n >= 1
    diccionario = diccionario_caracteres_n(base, n)
    tamano = len(base)
    final_contador = np.array([[k,0] for k in diccionario], dtype=object)
    contador_palabras_cortas = []
    for pal in palabras:
        largo_palabra = len(pal)
        if largo_palabra > n:
            substr_act = pal[0:n]
            coordenadas = np.array([base.find(substr_act[k]) for k in range(n)])
            for k in range(largo_palabra-n):
                pos_actual_diccionario = np.sum([coordenadas[j] * tamano ** (n-j-1) for j in range(n)], dtype=int)
                final_contador[pos_actual_diccionario][1] += 1
                sig_letra = pal[k+n]
                new_coordenadas = np.append(coordenadas[1:],np.array([base.find(sig_letra)]))
                coordenadas = new_coordenadas
                substr_act = pal[k:k+n]
            pos_actual_diccionario = np.sum([coordenadas[j] * tamano ** (n-j-1) for j in range(n)], dtype=int)
            final_contador[pos_actual_diccionario][1] += 1
        else:
            contador_palabras_cortas += [pal]
    return final_contador, np.array(contador_palabras_cortas)


def estadistica_maestra_n_menor(palabras_cortas: np.ndarray, contador_largas: np.ndarray, base: str, n: int) -> [np.ndarray, np.ndarray]:
    """
        Calcula la estadistica de substrings de caracteres de largo n de una base en una lista de palabras\\
        Sabiendo informacion de esto la estadistica de substrings más grandes que n y teniendo la lista\\
        de palabras pequenas. Para el contador de palabras grandes, lo que se hace es ver cuantas veces\\
        aparece un substring determinado (de largo n-1) al principio y al final de un substring de tamano n\\
        Cada uno de estos substrings aparecerá 2 veces (esta entre medio y tiene una letra antes y otra despues)\\
        o bien aparecerá contado 1 vez, esto quedará claro al tomar el mayor valor de los dos que aparezcan, ya que\\
        tendrá de forma implícita ambos casos.\\
            CUIDADO, ESTO FALLA CUANDO SE TIENE AL MENOS UNA PALABRA QUE PARTE Y TERMINA IGUAL\\
            Por ejemplo, coco parte y termina con "co", según esto, si antes se contó con 3\\
            ahora "co" se contaría una vez, en detrimento del caso que aparezca 2 veces.\\
            No se puede arreglar con juegos de paridad, debido a que si ocurre 2 veces, no habría\\
            manera de distinguirlo.\\
            Esta funcion sera potencialmente util solo para largos muy grandes, se desaconseja su uso.
        ##############################\\
        Inputs:\\
        palabras_cortas: numpy array de palabras de largo menor a n\\
        contador_largas: numpy array de dimensiones (base ** n) x 2, donde la primera coordeada\\
                         es el elemento con su frecuencia contada, y su segunda coordenada la frecuencia contada\\
        base: string que corresponde a caracteres válidos para la estadística\\
        n: largo de los substrings inicial\\
        ##############################\\
        Outputs:\\
        palabras_pequenas: (Idealmente, nótese de por el comentario que esto no servía tan bien) \\
                           Estadística de substrings de largos n-1 en base a estadísticas de largo n\\
                           Corresponde a un numpy array de dimensiones (base ** (n-1))x2, con la primera\\
                           coordenada siendo los elementos a contar su frecuencia, y la segunda la frecuencia\\
        liliputencies: numpy array de palabras de largo menor a n-1\\
    """
    assert n > 1
    tamano = len(base)
    palabras_pequenas, liliputences = estadistica_maestra_optimizado(palabras_cortas, base, n-1)
    for k in range(tamano ** (n-1)):
        contador_anterior = 0
        contador_siguiente = 0
        for j in range(tamano):
            coord_anterior = k * tamano + j
            coord_siguient = j * tamano ** (n-1) + k
            contador_anterior += contador_largas[coord_anterior][1]
            contador_siguiente += contador_largas[coord_siguient][1]
        final = max(contador_anterior, contador_siguiente)
        palabras_pequenas[k,1] += final
    return palabras_pequenas, liliputences
