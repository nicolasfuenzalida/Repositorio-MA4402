"""
    Toma un archivo desde "Textos" y devuelve un String con todo su contenido.
"""

import os

def lector(nombre):
    """
        Toma un nombre de archivo, y luego devuelve un string\n
        con todo el contenido en el texto de nombre.txt\n
        Asimismo, aprovecha de pasar todas las letras a minúscula\n
        ######################\n
        Inputs:\n
        nombre: String, que debe ser del nombre de un texto\n
        ######################\n
        Outputs:\n
        Importado: String, trae todo lo que traia el txt\n
    """
    cwd = os.getcwd() # Directorio actual
    texto = open(cwd+"\\Textos\\"+nombre+".txt","r",encoding="utf8")
    valores = texto.read()
    Importado = ""
    for x in valores:
        Importado += x
    Importado = Importado.lower()
    return Importado

#este es asi, porque la version anterior solo funciona en windows
def lector_mac(nombre):
    """
        Toma un nombre de archivo, y luego devuelve un string\n
        con todo el contenido en el texto de nombre.txt\n
        Asimismo, aprovecha de pasar todas las letras a minúscula\n
        ######################\n
        Inputs:\n
        nombre: String, que debe ser del nombre de un texto\n
        ######################\n
        Outputs:\n
        Importado: String, trae todo lo que traia el txt\n
    """
    cwd = os.getcwd() # Directorio actual
    texto = open(cwd+"/Textos/"+nombre+".txt","r",encoding="utf8")
    valores = texto.read()
    Importado = ""
    for x in valores:
        Importado += x
    Importado = Importado.lower()
    return Importado

