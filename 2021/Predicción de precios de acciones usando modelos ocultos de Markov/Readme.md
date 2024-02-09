# Predicción de precios de acciones usando modelos ocultos de Markov

## Integrantes:

Diego Dominguez

Sebastian Rojas

## Tema principal:

MCMC y estadística Bayesiana

## Resumen:

El texto aborda el desafío de predecir el precio de mercado, destacando su volatilidad y dificultad para modelarlo. Se propone utilizar técnicas basadas en modelos ocultos de Markov para mejorar la predicción del precio de mercado. Se explica que un modelo oculto de Markov consta de una cadena de Markov con estados, matriz de transición y distribución inicial, donde se generan símbolos pertenecientes a un conjunto y se observan sin conocer el estado de la cadena. Se describe el proceso de entrenamiento utilizando algoritmos como Forward-Backward, Viterbi y Baum-Welch.

En este proyecto, se emplea un modelo de Markov de orden máximo (MOM) con símbolos que representan el precio de apertura, cierre, máximo y mínimo de cada día. El objetivo es estimar estos precios para el día siguiente basándose en días hábiles consecutivos. Se detalla el proceso de predicción y se menciona que se evalúa el desempeño del modelo utilizando datos de 100 días hábiles consecutivos. Finalmente, se anticipa que la presentación incluirá una revisión del concepto de modelo oculto de Markov, la explicación de los algoritmos utilizados y la presentación de resultados de las predicciones.

## Referencias:

[1] Rafiul Hassan, Md. y Nath B. (2005). Stock Market Forecasting Using Hidden Markov Model: A New Approach.

[2] Marjanovic, B. (2017). Huge Stock Market Dataset.
