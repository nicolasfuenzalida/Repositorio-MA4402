# Monte Carlo for chance constraint. Multidiciplinary design optimization

## Integrantes:

Juan Cuevas.

Manuel Torres.

## Tema principal:

Métodos de Monte Carlo.

## Resumen:

Resumen: Estudiaremos problemas de optimización del siguiente estilo:

$$\begin{equation}
\begin{aligned}
& \underset{x}{\text{min}}
& & -x_2 \\
& \text{s.a.}
& & h_1(x, \xi, u, v) = \xi_2 x_1 + 2x_2 - u + v = 0, \\
&&& h_2(x, \xi, u, v) = 3x_1 - u - v = 0, \\
&&& \mathbb{P}(-\xi_1 + u(x, \xi) - \frac{1}{2} (\xi_2 + 1)x_1 \leq 0) \geq \alpha, \\
&&& \mathbb{P}(-v(x, \xi) \leq 0) \geq \alpha, \\
&&& x_1, x_2 \geq 0.
\end{aligned}
\end{equation}$$

dDonde $h_1$, $h_2$ son ecuaciones modeladas por diferentes disciplinas y $\xi = (\xi_1,\xi_2)$ corresponde a una distribución Gaussiana multivariada tal que $\mathbb{E}(\xi)=(1,1)$ y $\mathbb{V}\text{ar}(\xi) = I$ (la matriz identidad). Las cantidades $u(x,\xi)$ y $v(x,\xi)$ son llamadas variables de estados. El nivel de probabilidad es $\alpha = 0.9987$. El objetivo es obtener un óptimo implementando los métodos de Monte-Carlo para trabajar las restricciones probabilísticas.

## Referencias:

[1] Chiralaksanakul, A., & Mahadevan, S. (2007). Decoupled approach to multidisciplinary design optimization under uncertainty. Optimization and Engineering, 8(1).

[2] Cools, R. Advances in multidimensional integration. Journal of computational and applied mathematics 149, 1 (2002).
