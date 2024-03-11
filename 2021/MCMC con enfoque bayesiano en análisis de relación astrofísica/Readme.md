# Markov Chain Monte-Carlo con enfoque bayesiano en análisis de relación astrofísica

## Integrantes:

Kurt Walsen.

## Tema principal:

MCMC y estadística Bayesiana, Metropolis-Hasting.

## Resumen:

El objetivo de este proyecto consiste en estudiar la relación Masa - Dispersión de velocidades, la cual corresponde a una relación astrofísica que cumplen galaxias, donde existe una relación lineal entre la masa del agujero negro central M una galaxia y su dispersión de velocidades σ.

Se plantea el siguiente modelo lineal para describir la relación:

$$\log(\frac{M}{M_\odot}) = m \cdot \log(\frac{\sigma}{\sigma_0}) + b + \mathcal{N}(0,\omega^2)$$

Para deducir la relación lineal entre log(M) y log(σ) se estudiará la distribución de los parámetros a través de un modelo  generativo basado en un análisis bayesiano con sampleos a través de Markov Chain Monte-Carlo (MCMC).

El modelo bayesiano implementado considera el hecho de que los datos presentan outliers, incertezas bi-dimensionales y dispersión interna, efectos que se traducen en la cantidad de parámetros utilizados y la función a posteriori considerada (ver [2]).

## Referencias:

[1] Harris (2013) A Catalog of Globular Cluster Systems: What Determines the Size of a Galaxy’s Globular Cluster Population?

[2] Hogg (2010) Data analysis recipes: Fitting a model to data

[3] Foreman-Mackey et al. (2013) emcee: The MCMC Hammer
