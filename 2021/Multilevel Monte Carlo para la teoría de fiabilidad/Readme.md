# Multilevel Monte Carlo para la teoría de fiabilidad

## Integrantes:

Axel Alvarez V.

## Tema principal:

Métodos de Monte Carlo

## Resumen:

El texto describe un sistema de componentes que se evalúa en función de si están operativos (valor 1) o no (valor 0) en un momento dado. La funcionalidad del sistema se determina mediante un grafo, donde un sistema se considera funcional si existe un camino de izquierda a derecha que solo incluya componentes operativos. Además, se introduce el concepto de conjuntos de corte, que son aquellos conjuntos cuya falla de todos sus componentes resulta en la falla del sistema. Este concepto se utilizará en el proyecto junto con el algoritmo de Multilevel Monte Carlo (MLMC).

El objetivo del algoritmo MLMC es estimar la esperanza de un estimador $T_L$, utilizando una secuencia de estimadores $(T_0, T_1, \dots)$ que se aproximan a $T_L$ con precisión y costos crecientes. Se describe cómo el algoritmo utiliza diferencias entre estimaciones sucesivas para mejorar la precisión y reducir el costo en comparación con el método de Monte Carlo estándar.

Finalmente, el texto anticipa que se presentarán los métodos de MLMC aplicados a sistemas de fiabilidad, mostrando resultados numéricos que destacan las ventajas del MLMC sobre el algoritmo de Monte Carlo convencional discutido en clases.

## Referencias:

[1] Louis J.M. Aslett, Tigran Nagapetyan, Sebastian J. Vollmer, Multilevel Monte Carlo for Reliability Theory.
