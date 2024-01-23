"""
Este archivo crea un cifrado de forma aleatoria, incluye letras minúsculas.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from Lector import *
from frec import fre
from f1_energia import *



alfabeto = "abcdefghijklmnopqrstuvwxyz"
largo_alfabeto = len(alfabeto)

def diferencias(a: str,b: str) -> np.ndarray:
    """
    Dados dos strings a,b, del mismo largo, entrega una lista de numpy\n
    Que tiene el mismo número de elementos que los largos de cada string\n
    Y que en sus elementos es 0 si son iguales los elementos respectivos\n
    Y es 1 si son distintos.\n
    ####################################################################\n
    Inputs:\n
    a: String\n
    b: String\n
    ####################################################################\n
    Output:\n
    dif: Array de Numpy de largo len(a)=len(b)
    """
    assert len(a) == len(b)
    largo = len(a)
    dif = np.zeros(largo, dtype=int)
    for k in range(largo):
        dif[k] = (a[k] != b[k])
    return dif

def vecinos(string: str, cambio: np.ndarray) -> str:
    """
    Dado un string y un numpy array de tamano 2 que codifica que posiciones\n
    Se van a intercambiar, devuelve otro string con dichas posiciones intercambiadas\n
    Los elementos del numpy array estan acotados por el tamaño del string\n
    #####################################################################\n
    Inputs:\n
    string: String a modificar\n
    cambio: Numpy array de tamaño 2 con enteros\n
    #####################################################################\n
    Outputs\n
    final: String modificado
    """
    assert len(cambio) == 2
    assert cambio[0] < len(string) and cambio[1] < len(string)
    pos_1 = cambio[0]
    pos_2 = cambio[1]
    final = list(string)
    final[pos_1], final[pos_2] = final[pos_2], final[pos_1]
    final = "".join(final)
    return final


def alfabetizador(string: str):
    """
    toma un texto, y elimina todos los caracteres no pertenecientes al alfabeto
    y transforma mayusculas a minusculas
    """
    f = ''.join(c for c in string if c.isalpha())
    f_m = f.lower()
    return f_m

def crearInversa(string, base):
    """
        Dado un string que es una codificacion de elementos de la base\\
        Encuentra la permutacion inversa.
    """
    inversa = list(base)
    i = 0
    tamano = len(base)
    while i<tamano:
        im_letra_act = string[i]
        ind_imagen = base.index(im_letra_act)
        inversa[ind_imagen] = base[i]
        i += 1
    string_final = ""
    for k in inversa:
        string_final += k
    return string_final