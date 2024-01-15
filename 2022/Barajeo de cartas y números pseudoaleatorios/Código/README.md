# Fenómenos de Cutoff en Cadenas de Markov
Bajo ciertas condiciones, toda cadena de Markov tiene una distribución invariante a la que converge a medida que la cadena evoluciona. Muchos modelos, entre ellos MCMC, dependen de la rapidez de convergencia de la cadena y de la seguridad con que esta está cerca de su medida invariante.

Bajos ciertas condiciones, una cadena de Markov experimenta lo que se conoce un fenómeno de cutoff. Este consiste en que la cadena converge abruptamente al equilibrio. De esta forma es posible cuantificar desde que momento la cadena se puede considerar como una buena aproximación de la distribución invariante.

Un ejemplo de esto es al revolver cartas. Se puede probar que al revolver un mazo de 52 cartas, bastan 7 iteraciones del método *riffle shuffle* para que el mazo tenga una distribución uniforme.

En este proyecto, exploramos dos modelos que presentan fénomeno de cutoff: la urna de Ehrenfest y el paseo aleatorio sesgado. 
