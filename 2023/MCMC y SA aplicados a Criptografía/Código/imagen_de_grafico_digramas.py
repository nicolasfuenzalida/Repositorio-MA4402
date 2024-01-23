"""
    Este archivo fue usado para crear el gráfico de frecuencias cuadrado\\
    de la presentación del 20 de diciembre. No tiene otra finalidad más allá de eso\\
    Es un imshow común y corriente\\
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from CifradoMonoalfabetico import *
from cifrado_nletras import diccionario_caracteres_n

diccionario = diccionario_caracteres_n(alfabeto, 2) # Crea el diccionario

fig, ax = plt.subplots(1,1) # Crea la imagen

color="red" # Determina el tono de los gráficos


eje_x = np.linspace(0, 25, num=26, dtype=int) # Primera letra
eje_y = np.linspace(0, 25, num=26, dtype=int) # Segunda letra

asignacion_color = -1/np.log(freqs_rel_digrama_universo[eje_x*len(alfabeto) + eje_y[:, np.newaxis]]) # Se pone la estadística del inglés
mapa = plt.imshow(asignacion_color, cmap="Reds") # Se hace el gráfico


# Cosas para hacer más agradable a la vista un gráfico.
cb = fig.colorbar(mapa)
cb.set_label(r"$\frac{-1}{\mathrm{log}(\mathrm{Frecuencia\: Relativa})}$", rotation=270, labelpad=25, fontsize=15)
ax.tick_params(axis="both", labelsize=13, length=0, top=True, labeltop=True, bottom=False, labelbottom=False)
ax.xaxis.set_label_position('top')
ax.set_xticks(eje_x)
ax.set_xticklabels(alfabeto)
ax.set_yticks(eje_y)
ax.set_yticklabels(alfabeto)
ax.set_xlabel("Primera letra", fontsize=15)
ax.set_ylabel("Segunda letra", fontsize=15)
ax.set_aspect('equal')
fig.savefig("grafico_final.png")
plt.show()