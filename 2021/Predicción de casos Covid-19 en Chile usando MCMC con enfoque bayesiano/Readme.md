# Predicción de casos Covid-19 en Chile usando MCMC con enfoque bayesiano

## Integrantes:

Tomás Laengle.

Sebastián Tapia.

## Tema principal:

Metropolis-Hasting.

## Resumen:

En este trabajo, se empleará la inferencia de parámetros bayesianos mediante métodos de Markov Chain Monte Carlo (MCMC) en el modelo epidemiológico Susceptible-Infectado-Recuperado (SIR), con el propósito de comparar las predicciones con los datos originales en Chile. El modelo SIR divide la población en tres conjuntos disjuntos y se representa como una cadena de Markov con tres estados, con tasas de contagio ($\lambda$) y de recuperación ($\mu$). El objetivo es aproximar estas tasas y predecir el comportamiento futuro de la pandemia. Se utiliza un enfoque bayesiano, donde la función de verosimilitud y las distribuciones a priori de los parámetros son clave. Se emplea el método MCMC para aproximar las distribuciones de los parámetros y obtener estimaciones. Se utilizan datos del Ministerio de Salud, incluyendo la población total, contagios diarios y casos recuperados, durante seis meses desde abril de 2020. Se realizarán comparaciones entre las aproximaciones y datos reales para validar el modelo y sacar conclusiones.

## Referencias:

[1] A discrete stochastic model of the COVID-19 outbreak: Forecast and control https://www.researchgate.net/publication/339950496_A_discrete_stochastic_model_of_the_COVID-19_outbreak_Forecast_and_control

[2] Bayesian inference of COVID-19 spreading rates in South Africa https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237126

[3] Datos COVID19 Chile
