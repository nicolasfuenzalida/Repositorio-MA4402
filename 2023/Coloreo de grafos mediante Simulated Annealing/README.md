# Coloreo de grafos mediante Simulated Annealing

## Integrantes:

Ramiro Hoffens

Benjamin Mitchell

## Tema principal:

Aplicación de MCMC: Simulated Annealing

## Resumen:

El problema de coloreo de grafos, sobre un grafo G = (V, E) y un conjunto de colores C consiste en
encontrar una función x : V → C tal que no hayan dos nodos adyacentes con el mismo color. Esto puede
plantearse como un problema de minimización sobre H(x) = ∑u∼v 1xu=xv , donde la configuración x será
solución al problema de coloreo si y sólo si H(x) = 0.
Es de particular interés encontrar soluciones donde se minimice el tamaño de C. Es posible acotar el
número cromático por ∆(G) + 1, donde ∆(G) es el grado mayor en el grafo.

## Referencias:

[1] ’Simulated Annealing Algorith for Graph Coloring’; A. Köse, B. Aral, M. Balaban, 2017

[2] ’Apuntes Simulación Estocástica’; Joaquín Fontbona, 2023

[3] ’Coloración de Grafos’, María Rosa Murga Díaz, Universidad de Cantabria, 2013
