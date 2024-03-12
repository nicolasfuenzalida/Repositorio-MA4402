# Proyecto importante

Este proyecto fue escogido para que cualquier usuario pueda realizar modificaciones facilmente. En la carpeta, los códigos que comiencen con "Ejecutable" son una versión destinada a que se le hagan modificaciones de parámetros. Para esto, ir a las celdas que tengan comentado lo siguiente:

"# Se puede cambiar"

# Abstract

Inspirados en la tesis de Nurul Huda [1] se estudiará numéricametne la ecuación diferencial estocástica "Random Flight Model" (RFM) en el eje vertical modificada con el lema de Ito y normalizando algunos parámetros con tal de tener una implementación sencilla.

Considerando el contexto meteorológico de esta EDE se estudian 3 perfiles distintos, estos derivados estadísticamente de las corrientes atmosféricas en la capa límite de la atmósfera (ABL). Dichos modelos son "inestable" que es cuando la superficie se calienta durante el día donde ocurre un efecto termal de corrientes convectivas; el segundo, "estable", que ocurre durante las noches y ocurre que el nivel de turbulencia decrece con la altura, el tercero "neutral" que ocurre cuando la capa está nublada y con bastante viento.

Respecto a la ecuación esta depende de funciones elementales, estas son $\tau_w$ y $\sigma_w$, que son el tiempo de decorrelación lagrangiano y la desviación estandar del turbulente de velocidad, cumpliendo este que $\sigma_w ^2 = E[W^2_t] = E[W^2_0]$, estas funciones cambian según el perfil a estudiar.

En este proyecto se estudiará el rendimiento de 4 esquemas de resolución de EDEs, refiérase a los de Euler, Milstein, HON-SKRII[1] y Leggraup[1], luego se devolverá el cambio de variables de Ito para obtener la velocidad y altura de estos y análizar el comportamiento de 10 partículas para cada modelo.

# Modelo
Luego del análisis explicado en el archivo jupyter, se estudia el siguiente modelo
$$d\Omega_t = \left( -\frac{\Omega_t}{\tau_w} + \frac{\partial \sigma_w}{\partial z} \right)dt +\left( \frac{2}{\tau_w} \right)^\frac{1}{2}dB_t\space,\space\space \Omega_0 \sim N(0,1)$$

$$dZ_t = \Omega_t\sigma_wdt \space,\space\space Z_0 \sim N(z_0,\sigma^2_z)$$

# Datos importantes
Estos ya se han mencionado antes, pero acá se guardan todos.

$W_t$ y $Z_t$ son la velocidad vertical y altura respectivamente de una partícula en un tiempo dado.

$\Omega_t = \frac{W_t}{\tau_t}$ es el cambio de variables que simplifica el modelo.

$\tau_t$ es el tiempo de decorrelación lagrangiano, dígase de el tiempo que toman las partículas en perder la correlación respecto a condiciones iniciales.

$\sigma_t$ es la desviación estandar del turbulente de velocidad.


# Alcance ético

El modelo tiene un impacto ético dependiendo del uso dado, por ejemplo, este modelo puede servir para predicción (con cierto grado de error) del movimiento de una nube de piroclastos de un volcán activo o emisiones polutantes derivadas de procesos manufactureros, luego con esta información se pueden tomar decisiones de prevención para la población afectada y el mismo medio ambiente.

# Referencia 

[1] Nurul Huda, M. (2016). Stochastic trajectory modelling of atmospheric dispersion. University College London.
