# Simulated Annealing para modelamiento de líneas de metro

## Integrantes:

Martín Sepúlveda.

Lucas Villagrán.

## Tema principal:

Aplicación de MCMC: simulated annealing.

## Resumen:

El objetivo de este proyecto es simular una red de estaciones de micro (también aplicable para generar líneas de metro) dada una distribución de casas aleatoria en una ciudad simplificada, generada siguiendo el apunte [3]. Inicialmente, las estaciones se generan de manera normal y luego se aplican Voronoi Tilings junto con el algoritmo Simulated Annealing. El problema y la función objetivo se plantean de manera similar al paper [2], basándose en el algoritmo descrito en el paper [1]. La función objetivo es maximizar la utilidad de las estaciones, abarcando el mayor número posible de casas y puntos de interés cercanos en cada región, para luego conectarlas teniendo en cuenta la ruta de metro de menor distancia utilizando el Traveling Salesman Problem estocástico.

## Referencias:

[1] Byers, S., & Raftery, A. E. (2002). Bayesian estimation and segmentation of spatial point processes using Voronoi tilings. En Chapman and Hall/CRC eBooks (pp. 123-138).

[2] Stadler, T., Hofmeister, S., & Dünnweber, J. (2022). A method for the optimized placement of bus stops based on Voronoi diagrams. Proceedings of the . . . Annual Hawaii International Conference on System Sciences. https://doi.org/10.24251/hicss.2022.694

[3] (S/f). U-cursos.cl. Recuperado el 21 de noviembre de 2023, de https://www.u-cursos.cl/ingenieria/2023/2/MA4402/1/material_docente/detalle?id=6924161
