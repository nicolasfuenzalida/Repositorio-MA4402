# Flujos Normalizantes y Autorregresivos

## Integrantes:

C. Palma

A. Wortsman

## Tema principal:

Algoritmos estocasticos/otros

## Resumen:

Los Flujos Normalizantes (Normalizing Flows, en inglés) [1] son una forma de construir 
densidades de probabilidad bastante flexibles sobre variables aleatorias continuas. En concreto, dadas
una variable aleatoria y ∈ Rn y x una variable aleatoria ”simple”, buscamos encontrar una 
transformación T : Rn → Rn tal que: T(x) ∼ py, x ∼ px.

Tres casos particulares de flujos autorregresivos son considerados: Afines, Splines (polinomiales
o racionales por trozos) y Splines con acoplamiento [3].

Se estudiarán implementaciones de flujos normalizantes con estas últimas tres funciones para
distintas distribuciones con el objetivo de lograr samplear de estas, sin conocerlas explícitamente,
y partiendo de una normal estándar.

## Referencias:

[1] E. G. Tabak and E. Vanden-Eijnden. Density estimation by dual ascent of the log-likelihood.
Communications in Mathematical Sciences, 8(1):217–233, 2010.

[2] D. P. Kingma, T. Salimans, R. Jozefowicz, X. Chen, I. Sutskever, and M. Welling. Improved
variational inference with inverse autoregressive flow. Advances in neural information processing
systems, 29:4743–4751, 2016.

[3] L. Dinh, D. Krueger, and Y. Bengio. Nice: Non-linear independent components estimation.
arXiv preprint arXiv:1410.8516, 2014.
