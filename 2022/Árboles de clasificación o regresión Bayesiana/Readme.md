# Árboles de clasificación o regresión Bayesiana

## Integrantes:

Jorge Sossa.

Gary Vidal.

## Tema principal:

MCMC y estadística Bayesiana.

## Resumen:

Este proyecto se centra en la creación de modelos de clasificación, ajustando de manera estocástica un árbol de decisión. El problema a resolver es encontrar un árbol de decisión que se ajuste al modelo, de tal manera que las predicciones sean lo más precisas posible.

A diferencia de otros maneras para ajustar árboles, este método utiliza dos ejes principales:
- Se especifica una distribución a priori.
- Se realiza una búsqueda estocástica para encontrar el posteriori.

La distribución a priori de un árbol se obtiene identificando los nodos terminales y los nodos interiores, a los cuales se les asigna una probabilidad dependiendo de su altura y si estos fueron divididos o no, la probabilidad es la multiplicación de todos estos términos.

## Referencias:

[1] Chipman, H. A., George, E. I., and McCulloch, R. E. (1998). Bayesian CART Model Search. Journal of the American
Statistical Association, 93 (443), 935-948. http://dx.doi.org/10.1080/01621459.1998.10473750

[2] A Practical Markov Chain Monte Carlo Approach to Decision Problems. Timothy Huang. Yurity Nevmyvaka. https://aaai.org/papers/flairs-2001-100/
