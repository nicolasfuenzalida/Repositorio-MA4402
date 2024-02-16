# Modelación de trayectorias de dispersión meteorológicas

## Integrantes:

Felipe Latorre Naranjo.

Jorge Sepúlveda Ruz.

## Tema principal:

Ecuaciones Diferenciales Estocásticas.

## Resumen:

Inspirados en la tesis de Nurul Huda [1] se estudiará numéricametne la ecuación diferencial estocástica "Random Flight Model” (RFM) en el eje vertical modificada con el lema de Ito y normalizando algunos parámetros con tal de tener una implementación sencilla. Considerando el contexto meteorológico de esta EDE se estudian 3 perfiles distintos, estos derivados estadísticamente de las corrientes atmosféricas en la capa límite de la atmósfera (ABL). Dichos modelos son "inestable” que es cuando la superficie se calienta durante el día donde ocurre un efecto termal de corrientes convectivas; el segundo, "estable”, que ocurre durante las noches y ocurre que el nivel de turbulencia decrece con la altura, el tercero "neutral” que ocurre cuando la capa esta nublada y con bastante viento. Respecto a la ecuación esta depende de funciones elementales, estas son τw y σw, que son el tiempo de decorrelación lagrangiano y la desviación estándar del turbulente de velocidad, estas funciones cambian según el perfil a estudiar. En este proyecto se estudiará el rendimiento de 4 esquemas de resolución de EDEs, refiérase a los de Euler, Milstein, HON-SKRII[1] y Leggraup[1], luego se devolverá el cambio de variables de Ito para obtener la velocidad y altura de estos y analizar el comportamiento de 10 partículas para cada modelo.

## Referencias:

[1] Nurul Huda, M. (2016). Stochastic trajectory modelling of atmospheric dispersion. University College London.
