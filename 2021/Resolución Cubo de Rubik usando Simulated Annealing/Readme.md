# Resolución Cubo de Rubik usando Simulated Annealing

## Integrantes:

Sebastián Cobaise.

Arturo Lazcano.

## Tema principal:

Aplicación de MCMC: simulated annealing.

## Resumen:

El proyecto se centra en la implementación del algoritmo estocástico Simulated Annealing, para resolver eficientemente el cubo de Rubik de 3x3 y 2x2. Basándose en literatura especializada y conocimientos adquiridos en el curso, se desarrolla un código en Python utilizando librerías como numpy y matplotlib. El cubo se modela como una matriz que refleja el color de cada pieza, estableciendo una función objetivo que minimiza el número de piezas mal colocadas. Se define un grafo donde cada vértice representa un estado del cubo y se conecta con los estados resultantes de posibles rotaciones. El Simulated Annealing utiliza una probabilidad para decidir cambios de estado basada en la diferencia de la función objetivo entre estados y una temperatura ajustable. El documento final discute la optimización del algoritmo, problemas encontrados y resultados en términos de tiempo y número de movimientos.

## Referencias:

[1] Shahram Saeidi, 2018. Solving the Rubik’s Cube using Simulated Annealing and Genetic Algorithm.
