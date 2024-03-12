# Implementación numérica de esperanzas condicionales usando redes neuronales

## Integrantes:

Melanie Sánchez Pfeiffer.

## Tema principal:

Algoritmos estocásticos en aprendizaje de máquinas: Algoritmo de gradiente estocástico (SGD).

## Resumen:

Dadas X, Y variables aleatorias reales en $(\Omega, \mathcal{F}, \mathbb{P})$, se sabe que la esperanza condicional $\mathbb{E}(Y|X)$ es la proyección ortogonal de $Y$ en $(\Omega, \sigma(X), \mathbb{P})$, por lo tanto, es la función medible de $X$ más cercana a $Y$ en $L^2$. Es decir, $f = \mathbb{E}(Y|X)$ es la única solución del problema

$$\begin{equation}
\begin{aligned}
& \underset{x}{\text{min}}
& & \mathbb{E}(f(X)-Y)^2 \\
& \text{s.a.}
& & f \in K\\
\end{aligned}
\end{equation}$$

donde $K$ es el conjunto de todas las funciones medibles $f : \mathbb{R} \to \mathbb{R}$ tal que $f(X) \in L^2(\Omega, \mathcal{F}, \mathbb{P})$.

Por el Teorema de Hornik sobre la universalidad de redes neuronales en $L^2(\mu)$ se propone entrenar una red neuronal que aproxime la solución del problema.

## Referencias:

[1] Kurt Hornik, Maxwell Stinchcombe, and Halber White. MultilayerFeedforward Networksare Universal Approximators. Neural Networks, Vol2, pp.359-366,1989.

[2] Phillipp Grohs, and Gitta Kutyniok. Mathematical aspects of deep learning. Cambridge University Press, 2023.

[3] Apuntes Curso MA5606-1: Tópicos Matemáticos en Aprendizaje de Máquinas, Redes Neuronales y Aprendizaje Profundo. Profesores Joaquín Fontbona y Claudio Muñoz, 2023.
