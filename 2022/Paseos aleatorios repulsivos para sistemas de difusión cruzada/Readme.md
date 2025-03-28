# Simulación del proceso de Markov asociado al sistema SKT conservativo de difusion cruzada.

## Integrantes:

Benjamin Borquez.

Vicente Poblete.

## Tema principal:

Cadenas de markov.

## Resumen:

Buscamos modelar la interaccion entre dos especies animales que conviven en un determinado ecosistema espacio-temporal. Para esto consideramos funciones $u$ y $v$ que representan, para una variable espacial $(R^n)$ y una temporal $(R_+)$, la densidad de población de cada especie en ese lugar y tiempo. Se propone una implementación del proceso de Markov con el objetivo de simular poblaciones sujetas a distintos parámetros de difusion $d$ y difusion cruzada $a$. Esto implica una implementación basada en los vecinos de los estados actuales para reducir la cantidad de tasas calculadas. Además, usando esta implementación, se logra, a través de la función inversa generalizada, modelar los saltos de individuos a través del toro.

## Referencias:

[1] Felipe Muñoz-Hernandez, Ayman Moussa and Vincent Bansaye. Stability of a crossdiffusion system and approximation by repulsive random walks: a duality approach.
