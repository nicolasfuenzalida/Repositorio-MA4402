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


def initialize_terrain(size):
    return np.random.rand(size, size)

# iterations: 100
# transition_std: 0.1


def markov_chain_terrain(size, iterations, transition_std):
    terrain = initialize_terrain(size)

    for _ in range(iterations):
        new_terrain = np.copy(terrain)

        for x in range(1, size-1):
            for y in range(1, size-1):
                neighborhood = terrain[x-1:x+2, y-1:y+2]
                new_terrain[x, y] = np.mean(
                    neighborhood) + np.random.normal(0, transition_std)
        terrain = new_terrain

    return terrain


def calculate_energy(matrix):
    # Puedes definir tu propia función de energía aquí.
    # como input considera un terreno, ie una matriz de numeros entre -1 y 1
    rows, cols = matrix.shape
    all_cells = [(i, j) for i in range(rows) for j in range(cols)]
    inner_cells = [(i, j) for i in range(1, rows - 1)
                   for j in range(1, cols - 1)]

    # se implementa una penalización de elevación para evitar tener picos muy altos
    elevation_penalty = sum(abs(matrix[i, j] - matrix[i + 1, j])
                            + abs(matrix[i, j] - matrix[i, j + 1]) /
                            + abs(matrix[i, j] - matrix[i, j - 1]) /
                            + abs(matrix[i, j] - matrix[i+1, j + 1]) /
                            + abs(matrix[i, j] - matrix[i+1, j - 1]) /
                            + abs(matrix[i, j] - matrix[i-1, j - 1]) /
                            + abs(matrix[i, j] - matrix[i-1, j]) /
                            + abs(matrix[i, j] - matrix[i-1, j + 1]) for i, j in inner_cells)

    gradient_x, gradient_y = np.gradient(matrix)
    total_gradient = np.sqrt(gradient_x**2 + gradient_y**2)

    # se implementa una recompensa a la suavidad del terreno:
    # smoothness_penalty = sum(abs(matrix[i, j] - 2 * matrix[i + 1, j] + matrix[i + 1, j]) + abs(matrix[i, j] - 2 * matrix[i, j + 1] + matrix[i, j + 1]) for i, j in inner_cells)

    # se implementa una cohesión con respecto a los vecinos de la matrix:
    # cohesion_penalty = sum(abs(matrix[i, j] - matrix[i + 1, j + 1]) + abs(matrix[i, j + 1] - matrix[i + 1, j]) for i, j in inner_cells)

    # incentiva máximizar los rangos de elevación posible
    elevation_range_penalty = max(matrix.flatten()) - min(matrix.flatten())
    energy = elevation_penalty - \
        elevation_range_penalty + np.mean(total_gradient)
    return energy


def generate_neighbor(state, temperature):
    # Genera un vecino cambiando aleatoriamente algunos puntos del terreno.
    neighbor = state.copy()
    num_changes = int(np.shape(state)[0]/10)
    for _ in range(num_changes):
        x, y = np.random.randint(
            0, state.shape[0]), np.random.randint(0, state.shape[1])
        neighbor[x, y] += np.random.normal(0, temperature)
    return neighbor


def simulated_annealing(initial_state, max_iterations, cooling_rate):
    current_state = initial_state
    current_energy = calculate_energy(current_state)
    iteration_arr = []
    energy_arr = []
    for iteration in range(max_iterations):
        temperature = initial_temperature / (1 + cooling_rate * iteration)

        # Perturb the current state
        new_state = generate_neighbor(current_state, temperature)
        new_energy = calculate_energy(new_state)

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


def ploteo3d(terrain, size, title="3D Plot"):
    # get aspect ratio of tif file for late plot box-plot-ratio
    y_ratio, x_ratio = size, size

    # create arrays and declare x, y, z variables
    lin_x = np.linspace(0, 500, terrain.shape[0], endpoint=False)
    lin_y = np.linspace(0, 500, terrain.shape[1], endpoint=False)
    y, x = np.meshgrid(lin_y, lin_x)

    z = terrain

    # Apply gaussian filter, with sigmas as variables. Higher sigma = more smoothing and more calculations.
    sigma_y = 1
    sigma_x = 1
    sigma = [sigma_y, sigma_x]
    z_smoothed = sp.ndimage.gaussian_filter(z, sigma)

    # Some min and max and range values coming from gaussian_filter calculations
    z_smoothed_min = np.amin(z_smoothed)
    z_smoothed_max = np.amax(z_smoothed)

    # Creating figure with the specified title
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    ax = plt.axes(projection='3d')
    ax.azim = -30
    ax.elev = 0
    ax.set_box_aspect((x_ratio, y_ratio, ((x_ratio + y_ratio) / 8)))

    surf = ax.plot_surface(x, y, z_smoothed, cmap='terrain', edgecolor='none')

    m = cm.ScalarMappable(cmap=surf.cmap, norm=surf.norm)
    m.set_array(z_smoothed)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Parámetros
    terrain_size = 30
    initial_temperature = 1.0
    max_iterations = 7000
    cooling_rate = 0.01
    seed = 42

    # Genera estado inicial y aplica Simulated Annealing
    initial_state = markov_chain_terrain(terrain_size, 100, 0.1)
    final_state, iteration_arr, energy_arr = simulated_annealing(
        initial_state, max_iterations, cooling_rate)

    plt.plot(iteration_arr, energy_arr)
    plt.xlabel('Iteración')
    plt.ylabel('Energía')
    plt.title('Simulated Annealing/Markov Chain - Resultados de Energía')
    plt.yscale('log')
    plt.show()

    # Visualización del terreno inicial
    plt.subplot(1, 2, 1)
    plt.title('Terreno Inicial Markov Chain')
    plt.imshow(initial_state, cmap='terrain', interpolation='bilinear')
    ploteo3d(initial_state, terrain_size, title="Terreno Inicial Markov Chain")

    # Visualización del terreno final
    plt.subplot(1, 2, 2)
    plt.title('Terreno Simulated Annealing / Markov Chain ')
    plt.imshow(final_state, cmap='terrain', interpolation='bilinear')
    ploteo3d(final_state, terrain_size,
             title="Terreno Simulated Annealing / Markov Chain")

    plt.show()
