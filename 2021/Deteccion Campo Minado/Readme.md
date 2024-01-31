# Detección Campo Minado

## Integrantes:

Cynthia Vega

Daniel Minaya

## Tema principal:

Simulación de cadenas de Markov/Metropolis-Hasting

## Resumen:

En una imagen aérea de un área potencialmente minada, se
pueden detectar distintos tipos de objetos que no necesariamente
son minas, por lo que es de gran interés para la población tener
un método que sea capaz de detectar espacios con una alta
probabilidad de contener minas, de manera objetiva.
Para poder determinar las áreas con mayor probabilidad de
contener minas, se particiona el espacio de estudio en regiones de
Voronoi, cada una con intensidad constante, y se propone mover
estas regiones junto a las intensidades asignadas a la presencia
de minas a través de iteraciones MCMC con pasos de Gibbs y
Metropolis–Hastings.


## Referencias:

[1] Simon D. Byers ; Adrian E. Raftery
Bayesian Estimation and Segmentation of Spatial Point Processes using Vorono¨ı Tilings..

[2] Daniel Walsh and Adrian E. Raftery
Detecting Mines in Minefields with Linear Characteristics.
