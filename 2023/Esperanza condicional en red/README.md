# Implementación numérica de esperanzas condicionales usando redes neuronales

## Integrantes:

Melanie Sánchez Pfeier

## Tema principal:

Algoritmos estocásticos en aprendizaje de máquinas: Algoritmo de gradiente estocástico (SGD)

## Resumen:

Dadas X, Y variables aleatorias reales en
(Ω, F, P), se sabe que la esperanza condicional 
E(Y |X) es la proyección ortogonal de Y en
(Ω, σ(X), P), por lo tanto, es la función medi-
ble de X más cercana a Y en L2. Es decir,
f = E(Y |X) es la única solución del problema
min E(f(X)-Y)^2 s.a f∈K
donde K es el conjunto de todas las funciones
medibles f : R → R tal que f (X) ∈ L2(Ω, F, P).
Por el Teorema de Hornik sobre la universalidad
de redes neuronales en L2(μ) se propone entrenar una red neuronal
que aproxime la solucion del problema.

## Referencias:

[1] Kurt Hornik, Maxwell Stinchcombe, and
Halber White. MultilayerFeedforward Networksare Universal Approximators. Neural
Networks, Vol2, pp.359-366,1989.

[2] Phillipp Grohs, and Gitta Kutyniok. Mathematical aspects of deep learning. Cambridge University Press, 2023.

[3] Apuntes Curso MA5606-1: Tópicos Matemáticos en Aprendizaje de Máquinas, Redes Neuronales y Aprendizaje Profundo.
Profesores Joaquín Fontbona y Claudio
Muñoz, 2023.
