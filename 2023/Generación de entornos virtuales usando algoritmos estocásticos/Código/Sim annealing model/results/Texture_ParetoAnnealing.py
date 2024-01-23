import numpy as np
import matplotlib.pyplot as plt
import noise
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage
from PIL import Image
# from arrows3dplot import * # python_file in project with class
import matplotlib.cm as cm
import math
import random


# Función de evaluación basada en la pendiente del terreno
def evaluate_terrain(terrain):
    # Calcular la pendiente del terreno
    gradient_x, gradient_y = np.gradient(terrain)
    val = gradient_x**2 + gradient_y**2
    slope = np.sqrt(gradient_x**2 + gradient_y**2)
    # En este ejemplo, la función de evaluación es la suma de las pendientes
    return np.sum(slope)


def generate_neighbor(state, temperature):
    # Genera un vecino cambiando aleatoriamente algunos puntos del terreno.
    neighbor = state.copy()
    num_changes = int(np.shape(state)[0])
    for _ in range(num_changes):
        x, y = np.random.randint(
            0, state.shape[0]), np.random.randint(0, state.shape[1])
        neighbor[x, y] += np.random.normal(0, temperature)
    return neighbor

# Simulated Annealing para generar terreno realista


def simulated_annealing(initial_state, max_iterations, cooling_rate):
    current_state = initial_state
    current_energy = evaluate_terrain(current_state)
    iteration_arr = []
    energy_arr = []
    for iteration in range(max_iterations):
        temperature = initial_temperature / (1 + cooling_rate * iteration)

        # Perturb the current state
        new_state = generate_neighbor(current_state, temperature)
        new_energy = evaluate_terrain(new_state)

        # Calculate the change in energy
        delta_energy = new_energy - current_energy

        # Accept the new state with a probability based on the temperature and energy change
        if delta_energy < 0 or random.uniform(0, 1) < math.exp(-delta_energy / temperature):
            current_state = new_state
            current_energy = new_energy

        # Print or log the energy at regular intervals
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Energy: {current_energy}")

        iteration_arr.append(iteration)
        energy_arr.append(current_energy)

        # Add other termination conditions if needed

    return current_state, iteration_arr, energy_arr


if __name__ == "__main__":
    # Parámetros
    terrain_size = 200
    initial_temperature = 1
    max_iterations = 5000
    cooling_rate = 0.01
    seed = 43

    # Genera estado inicial y aplica Simulated Annealing
    initial_state = np.random.pareto(a=1.8, size=(terrain_size, terrain_size))
    final_state, iteration_arr, energy_arr = simulated_annealing(
        initial_state, max_iterations, cooling_rate)

    plt.plot(iteration_arr, energy_arr)
    plt.xlabel('Iteración')
    plt.ylabel('Energía')
    plt.title('Pareto / Simulated Annealing - Resultados de Energía')
    plt.yscale('log')
    plt.show()

    # Visualización del terreno inicial
    plt.subplot(1, 2, 1)
    plt.title('Textura Pareto')
    plt.imshow(initial_state, cmap='terrain', interpolation='bilinear')

    # Visualización del terreno final
    plt.subplot(1, 2, 2)
    plt.title('Textura Simulated Annealing / Pareto Texture')
    plt.imshow(final_state, cmap='terrain', interpolation='bilinear')

    plt.show()
