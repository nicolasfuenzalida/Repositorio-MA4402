# Simulated annealing para planificación de trayectorias aéreas

## Integrantes:

Catalina Lizana.

Fabián A. Ulloa.

## Tema principal:

Aplicación de MCMC: simulated annealing.

## Resumen:

La idea de este proyecto es estudiar el uso de Simulated Annealing en la optimización de las trayectorias de aviones, con el fin de evitar choques entre ellos. Para ello se tomará el modelo presentado en el paper "Large Scale 4D Trajectory Planning"[1]. Para efectos del proyecto se elegirán valores apropiados para los diversos parámetros asociados y se realizarán algunas simplificaciones del problema, esto pues, el caso real es de una complejidad computacional mayor.

El archivo Simulated_Annealing_2_airports.ipynb contiene la implementación del algoritmo simulando trayectorias aleatorias con punto de partida y de llegada iguales (i.e. 2 aeropuertos). El archivo Simulated_Annealing_More_airports.ipynb contiene la implementación del algoritmo pero con trayectorias que parten desde un mismo aeropuerto y que tienen distintos puntos de llegada.

En la carpeta Animaciones se encuentras las animaciones (gifs) asociadas al código Simulated_Annealing_2_airports.ipynb, el archivo ej_trays.gif es el ejemplo de como se ven las trayectorias, trays_.gif son las trayectorias usadas en el algoritmo SA, y trays_SA son las trayectorias obtenidas a partir de trays_.gif una vez corrido el algoritmo con la sucesion beta_n exponencial.

## Referencias:

[1] Arianit Islami, Supatcha Chaimatanan, Daniel Delahaye. Large Scale 4D Trajectory Planning. Air Traffic Management and Systems – II , 420, Springer, pp 27-47, 2016, Lecture Notes in Electrical Engineering.
