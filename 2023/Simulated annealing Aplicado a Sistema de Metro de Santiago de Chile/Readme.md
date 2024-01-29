# Simulated annealing Aplicado a Sistema de Metro de Santiago de Chile

## Integrantes:

Allen Arroyo

Isidora Miranda

## Tema principal:

Aplicación de MCMC: simulated annealing

## Resumen:

Para un modelo simplificado de la red de metro de Santiago de Chile se tiene como objetivo minimizar el tiempo total de espera para el transbordo de pasajeros entre líneas de metro ajustando el tiempo de envío del primer tren, el tiempo de circulación entre estaciones y el tiempo de permanencia en la estación, esto en cada línea de metro a través de Simulated Annealing (SA), como es realizado en los papers [1][2] y un algoritmo de generación de tiempos factibles llamado Vector Modifying Algorithm [1] con datos de tiempo generados por v.a. uniformes en rangos adecuados. Se sigue la metodología del paper [1] llamado “A simulated annealing algorithm for first train transfer problem in urban railway networks” donde se especifica que para determinar un horario sincronizado tal que no aumente el tiempo de espera total de los transbordos basta con obtener los tiempos relacionados al primer tren de cada línea debido al supuesto que los siguientes trenes presentan salidas a intervalos constantes de tiempo.

## Referencias:

[1] Kang, L., & Zhu, X. (2015, 7 de diciembre). A simulated annealing algorithm for first train transfer problem in urban railway networks. Applied Mathematical Modelling. https://doi.org/10.1016/j.apm.2015.05.008

[2] Liu, X., Huang, M., Qu, H., & Chien, S. (2018). Minimizing Metro Transfer Waiting Time with AFCS Data Using Simulated Annealing with Parallel Computing. Wiley. https://doi.org/10.1155/2018/4218625
