# Simulación y Estimación Bayesiana de Procesos Gaussianos

## Integrantes:

M. Cabezas.

A. Kolm.

## Tema principal:

4.4. MCMC y estadística Bayesiana.

## Resumen:

Este trabajo explora los procesos Gaussianos ($\mathcal{GP}$), examinando su comportamiento, simulación y aplicaciones en Inferencia Bayesiana y Aprendizaje de Máquinas. Un GP se define como una distribución de probabilidad en un espacio de funciones, donde se puede generar una función como un vector normal multivariado usando una función de media y un kernel. Los kernels controlan la regularidad de las funciones en un GP, como se ejemplifica con el movimiento browniano y el kernel SE. La simulación de un GP requiere discretización debido a la incapacidad de los ordenadores para representar funciones continuas. La Inferencia Bayesiana ajusta GP a datos reales mediante el teorema de Bayes, definiendo priors sobre la función de media, el kernel y los hiperparámetros, y optimizando la verosimilitud del modelo. Se muestran dos resultados de inferencia bayesiana: regresión sobre datos simulados y regresión sobre datos reales.

## Referencias:

[1] Apunte del curso MA5204-1 Aprendizaje de máquinas: https://github.com/GAMES-UChile/Curso-Aprendizaje-de-Maquinas/blob/master/notas_de_clase.pdf

[2] C. E. Rasmussen & C. K. I. Williams, Gaussian Processes for Machine Learning, the MIT Press, 2006, Massachusetts Institute of Technology. www.GaussianProcess.org/gpml

[3] Duvenaud, D. (2014). Automatic model construction with Gaussian processes (Doctoral dissertation, University of Cambridge). https://web.cs.toronto.edu/
