# Estudio del metodo híbrido de gradiente estocástico y ecuación de Langenvin

## Integrantes:

Victoria Andaur

Matias Vera

## Tema principal:

Algoritmo de gradiente estocástico (SGD) y MCMC y estadística Bayesiana

## Resumen:

Como sociedad, en casi todas las áreas de la vida, nos
interesa realizar predicciones con datos, pero en los
últimos años ha habido un incremento masivo en la
cantidad de datos disponible lo que ha conllevado que
se manejen constantemente conjuntos de datos que
superan el millón de casos y es precisamente en estos
conjuntos en los que los métodos de Cadenas de
Markov de Monte Carlo (MCMC) fallan, ya que para
implementarse requieren actuar sobre la totaliad de
los datos. Por otro lado, el método de gradiente 
descendiente estocástico (SGD) permite trabajar con más
información de manera rápida, pero no permite cuantificar
la incerteza de los datos. Es así que [1] propone
un modelo híbrido, en donde se pueda calcular la 
incerteza de los datos a un bajo costo computacional.

## Referencias:

[1] M. Welling and Y. W. Teh, “Bayesian learning via
stochastic gradient Langevin dynamics,” in Pro-ceedings of the 28th international conference on machine learning (ICML-11), 2011, pp. 681–688
