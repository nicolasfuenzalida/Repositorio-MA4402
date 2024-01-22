# CPP-SA
Simulated Annealing on Chinese Postman Problem

El CPP es un problema $NP$-completo, que dado un grafo $G$ simple, finito y conexo con pesos en sus aristas, buscar un recorrido cerrado (que empiece y termine en el mismo vértice) de peso mínimo. Para esto, buscaremos soluciones a partir de la técnica Simulated Annealing (recocido simulado).
Esta técnica se basa en que a partir de un grafo donde los nodos son posibles soluciones factibles y las aristas existen cuando hay una relación simétrica entre dos posibles soluciones, se va explorando el grafo a partir de la aleatoriedad, donde la posible solución va cambiando a 
partir de ciertos algoritmos estocásticos. Dependiendo de la construcción de este grafo, su relación de vecinos y sus parámetros, se puede probar teóricamente que converge a una solución óptima.

La construcción del grafo y el acercamiento a la resolución del problema, incluído el algoritmo se encuentran en Modelamiento.md, mientras que el el código aplicado se encuentra en un python notebook, llamado algoritmo.ipynb. En este código, se encuentra resuelto un ejemplo de juguete de grafo completo de 7 nodos, donde la representación de la solución se encuentra en scatter.gif.
Además de este ejemplo, también se introdujeron dos grafos más, de 20 y 35 nodos respectivamente, cuyas soluciones (caminos) se encuentran representadas en los archivos scatter20.gif y scatter35.gif. Cabe destacar que en el caso del grafo de 20 nodos, el tiempo de ejecución estuvo debajo del minuto, mientras que con 35 nodos, la ejecución tomó alrededor de 5 minutos para cada CdM.

Finalmente, se utiliza una base de datos online de las ciudades de Chile, dada por: https://simplemaps.com/data/cl-cities. Sobre la base de esta, se obtiene la latitud/longitud de las ciudades, se pasan a sistema de coordenada en X-Y, se calcula la distancia, y con ello se construye un grafo de 30 nodos (tomando 30 ciudades) conexo. Este grafo se construye de manera aleatoria (en términos de cuáles aristas se toman y cuáles no), lo cual puede representar que solo hay ciertas rutas disponibles a seguir a través de las distintas ciudades. Se ejecuta el algoritmo y se obtiene un resultado que está cerca de lo que sería un óptimo euleriano (peso igual a la suma de pesos totales de aristas). El resultado se observa en scatter30_ciudades.gif.

IMPORTANTE: Hay que tener un cuidado especial cuando se ejecuta el código para obtener las animaciones, pues hay que ejecutarlos dos veces, guardando el camino óptimo en una lista manualmente. Esto se indica en algoritmo.ipynb. Recordar que también hay que importar nuevamente las librerías (primera celda).

Referencias:
1. Jing Zhang (2011). Modeling and Solution for Multiple Chinese Postman CSEE.
2. Pareto Software, LLC. (2023). Chile Cities Database. SimpleMaps. https://simplemaps.com/data/cl-cities
3. Nokia Mobile Phones, Nokia 2110 User’s Guide, 1996.
