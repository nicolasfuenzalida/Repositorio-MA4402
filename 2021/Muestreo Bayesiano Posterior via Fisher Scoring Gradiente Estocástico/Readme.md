# Muestreo Bayesiano Posterior via Fisher Scoring Gradiente Estocástico

## Integrantes:

Branco Paineman.

Vicente Cabezas.

## Tema principal:

Algoritmo de gradiente estocástico (SGD).

## Resumen:

En la investigación presentada, se explora un método bayesiano para la optimización de parámetros en redes neuronales simples utilizando el algoritmo Stochastic Gradient Fisher Scoring (SGFS), comparándolo con el método estándar Stochastic Gradient Descent (SGD). Utilizando la librería de Deep Learning Pytorch, se modeló una red neuronal simple con función de activación sigmoid para clasificar imágenes en blanco y negro de los dígitos 9 y 7 del dataset MNIST. El SGFS, que mejora la implementación del algoritmo estocástico SGLD, busca calcular los parámetros óptimos de un modelo de Machine Learning a partir de una distribución a priori. Se destaca que SGFS es eficiente al utilizar mini-batches en lugar del dataset completo en cada iteración, y dependiendo de la tasa de avance, puede aproximar la distribución posterior mediante una distribución normal o imitar el comportamiento de SGLD. Los datos para el experimento provienen del dataset MNIST, el cual incluye 70.000 imágenes en escala de grises de dígitos escritos a mano, de los cuales se filtraron los dígitos 7 y 9 para realizar una clasificación binaria.

## Referencias:

[1] Sungjin Ahn, Anoop Korattikara, Max Welling (2012). Bayesian Posterior Sampling via Stochastic Gradient Fisher Scoring. https://arxiv.org/ftp/arxiv/papers/1206/1206.6380.pdf
