# Monte Carlo for chance constraint. Multidiciplinary design optimization

## Integrantes:

Juan Cuevas

Manuel Torres

## Tema principal:

Metodos de Monte Carlo

## Resumen:

Resumen: Estudiaremos problemas de optimización del siguiente estilo:

(1) min −x2

s.a. h1(x, ξ, u, v) = ξ2x1 + 2x2 − u + v = 0

h2(x, ξ, u, v) = 3x1 − u − v = 0

P (−ξ1 + u(x, ξ) − 0.5(ξ2 + 1)x1 ≤ 0) ≥ α

P (−v(x, ξ) ≤ 0) ≥ α

x1, x2 ≥ 0,

donde h1, h2 son ecuaciones modeladas por diferentes disciplinas y ξ = (ξ1, ξ2) corresponde a una distribución
Gaussiana multivariada tal que E(ξ) = (1, 1) y Var(ξ) = I (la matriz identidad). Las cantidades u(x, ξ) y v(x, ξ)
son llamadas variables de estados. El nivel de probabilidad es α = 0.9987. El objetivo es obtener un óptimo
implementando los métodos de Monte-Carlo para trabajar las restricciones probabilísticas.
## Referencias:

[1]

[2] 

[3]
