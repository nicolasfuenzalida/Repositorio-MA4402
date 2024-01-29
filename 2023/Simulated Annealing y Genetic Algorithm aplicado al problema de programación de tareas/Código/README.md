# Proyecto-Final-Simulacion-Estocastica
El problema de programación de trabajos, o JSP, es un problema muy estudiado en optimización combinatorial. Se abordará JSP con simulated annealing (SA) y el algoritmo genético (GA). 

plot_gantt_chart(lista_instancias, maquinas): función para poder visualizar los horarios, dando de input la lista de instancias y las máquinas, es llamado Gantt Chart.

costos(orden,tiempos,filas,columnas): costo para un horario. El costo se tomará como el tiempo de finalización de la última operación.

generar_instancias(maquinas,tareas): función que genera instancias aleatorias.

lista_instancias(orden,tiempos,maquinas,tareas): función que dado un orden de operaciones,la cantidad de máquinas, tareas y una lista de tiempos, retorne la lista de instancias.

neighborhood(orden_operaciones_original,lista_de_tiempos_original,filas,columnas): definir las vecindades de un horario, para ello vamos a alternar el orden del horario, intercambiando dos operaciones entre sí, lo cual define un vecino. Dos horarios con un orden son vecinas si se obtiene permutando exactamente 2 operaciones.

sim_ann(N,funcion,orden_i,tiempos,filas,columnas): implementación del método simulated annealing, lo ajustamos para el JSP.

Funciones beta: fn_beta, beta_raiz(n,C), beta_cuad(n,C), beta_pol(n,C), beta_exp(n,C)

adjacent_swapping(orden,tiempo,filas,columnas): obtener todos los vecinos de un horario. Nos basamos en la vecindad neighborhood ya definida.

DG_distance(s1,s2,filas,columnas): función que representa el número de operaciones diferentes entre dos horarios sigma_1 y sigma_2.

crossover(p1,p2,t1,t2,filas,columnas): definir cruce entre dos individuos/horarios.

mutation(p1,t1,filas,columnas): definir mutación de un individuo/horario.

genetic_algorithm(N,valor,filas,columnas): se crea el algoritmo genético, N es la cantidad de generaciones, valor es un parámetro predeterminado para ver si se va a elegir crossover o mutation.
