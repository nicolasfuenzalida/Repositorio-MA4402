Observacion: para ejecutar los codigos correctamente hay que descargar Stockfish en la pagina https://stockfishchess.org/download/, descomprimirlo y moverlo 
a la misma carpeta del codigo. 

# Crimson-Red
Proyecto del curso Simulación Estocástica

Este proyecto consiste en crear un motor de ajedrez basado en una red neuronal como función de evaluación, para luego implementar el algoritmo alpha-beta-prunning.

Los archivos principales del proyecto son:

# Dataset

En este notebook se creó el dataset de entrenamiento, el cual se encuentra comprimido en el archivo Dataset_Entrenamiento.rar.

El dataset de entrenamiento consiste en aproximadamente 400000 posiciones de 5000 partidas descargadas de https://lichess.org/ con evaluaciones de StockFish como etiquetas.
A algunas posiciones se les agrearon jugadas aleatorias para diversificar el dataset.

# Entrenamiento_Modelo_colab
En este notebook se entrenó el modelo en google collab.

# Play
En este notebook se puede probar el modelo, jugando partidas con el o haciéndolo jugar contra StockFish. 

# Crimson_Red_v10_50e
Este es el modelo entrenado por 50 épocas con el dataset, el cual está en formato .h5 y .keras

Mas archivos:

# StockFish
En este archivo está StockFish, descargado de https://stockfishchess.org/

# Partidas_Crimson_Red
En este archivo hay varias partidas del Crimson Red contra StockFish de 1320 de elo en formato pgn.

# CrimsonRed_VS_StockFish
Esta es una partida de Crimson Red (Blancas) contra StockFish de 1320 de elo (Negras) en formato gif.

![](https://github.com/lvillarroel457/Crimson-Red/blob/main/CrimsonRed_VS_StockFish.gif)

# RedSinEntrenar_VS_StockFish
Esta es una partida de una red con la misma arquitercura que Crimson Red, pero sin entrenamiento (Blancas), contra StockFish de 1320 de elo (Negras) en formato gif.

![](https://github.com/lvillarroel457/Crimson-Red/blob/main/RedSinEntrenar_VS_StockFish.gif)

# Referencias
Neural Networks for Chess y Giraffe Using Deep Reinforcement Learning to Play Chess son las principales referencias del proyecto, las cuales están en formato pdf en la carpeta referencias del repositorio.

