"""
    La idea de este archivo es que, dado un string\n
    Se codifique dicho string.
"""

from FuncionesAuxiliares import *
import numpy as np

def decididor(letra: str, base: str) -> np.ndarray:
    """
        Dado un string de largo uno, y una base\n
        Devuelve una lista numpy de tamaño 2\n
        Cuya primera coordenada será True si la letra está\n
        en la base, o False de lo contrario, y en caso que sea\n
        cierto, devuelve la primera vez que aparece dicho caracter\n
        en la base\n
        #######################################\n
        Inputs:\n
        letra: String de tamaño 1\n
        base: String (tipicamente el alfabeto), que se verá si esta la letra o no\n
        #######################################\n
        Outputs:\n
        final: Lista de numpy de tamaño 2\n
            final[0]: Booleano, True si letra esta en base, Falso si no\n
            final[1]: Entero, posicion donde está la letra en la base por primera vez
    """
    contador = 0
    for k in base:
        if k == letra:
            final = np.array([True, contador])
            return final
        contador += 1
    final = np.array([False,-1])
    return final

def cifrador(clave, string, base=alfabeto):
    """
        Dada una clave en formato de string de 26 letras\n
        Y un string de tamano cualquiera, devuelve un texto\n
        Haciendo un reemplazo por valores de ese string en particular\n
        Si la clave es bcadefghijklmnopqrstuvwxyz, entonces una letra a\n
        se cifrará como b, una letra b como c, y una letra c como a, el resto\n
        Se mantendrá igual siempre\n
        #######################################\n
        Inputs:\n
        clave: String de 26 letras\n
        string: String, texto a codificar\n
        \n
        Opcionales:\n
        base: Desde donde se hará el cifrado, por defecto se tomara el alfabeto en\n
        minusculas, correspondiente a abcdefghijklmnopqrstuvwxyz\n
        #######################################\n
        Outputs:\n
        cifrado: string pasado por el filtro de cifrado\n
    """
    assert len(clave) == len(base)
    final = ""
    for letra in string:
        presencia, posicion = decididor(letra, base)
        if presencia:
            new_letter = clave[posicion]
            final += new_letter
        else:
            final += letra
    return final


def permutador(clave,string):
    """
    Dada una clave, que es una permutacion, por ejemplo [2,3,4,1]\n
    y un string del mismo largo de la permutacion, se aplica a este la permutacion
    """
    l_s = len(string)
    l_c = len(clave) 
    s = [None] * l_s
    if l_c != l_s:
        return 'tamaños no coinciden'
    else:
        for i in range(l_s):
            s[clave[i]-1] = string[i]
    return ''.join(s)


def p_inverso(clave,string):
    """
    Dada una clave, que es una permutacion, por ejemplo [2,3,4,1]\n
    y un string del mismo largo de la permutacion, que esta permutado por esta clave\n
    se encuentra el texto original
    """
    l_s = len(string)
    l_c = len(clave) 
    clave2 = [None] * l_s
    if l_c != l_s:
        return 'tamaños no coinciden'
    else:
        for i in range(l_c):
            clave2[clave[i]-1] = i+1
    return permutador(clave2,string)


def completador(text,clave):
    """
    dado un texto, se ve si este se puede dividr en bloques de\n
    largo p, donde p es el largo de la permutacion. En caso de no ser\n
    posible i.e, el ultimo bloque no tiene largo p, se agregan espacios\
    para completar
    """
    l_c = len(clave)
    text_final = text
    while len(text_final)%l_c != 0:
        text_final = text_final + ' '
    return text_final


def c_permutacion(clave,string):
    """
    Funcion que cifrara un texto, con el cifrado de permutacion\n
    de acuerdo a la clave
    """
    texto = completador(string,clave)
    l_t = len(texto)
    l_c = len(clave)
    t_final = ''
    cantidad_bloques = int(l_t/l_c)
    for i in range(cantidad_bloques):
        per = texto[l_c*i:l_c*(i+1)]
        t_final = t_final + permutador(clave,per)
    return t_final


def d_permutacion(clave,string):
    """
    funcion que, dado un texto cifrado, t, la posible clave con la\n
    que se cifro, entrega el posible texto original
    """
    texto = string
    l_t = len(texto)
    l_c = len(clave)
    t_final = ''
    cantidad_bloques = int(l_t/l_c)
    for i in range(cantidad_bloques):
        per = texto[l_c*i:l_c*(i+1)]
        t_final = t_final + p_inverso(clave,per)
    return t_final


