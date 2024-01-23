import numpy as np
import matplotlib.pyplot as plt
import noise


def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)

    # Avoid division by zero
    if min_val == max_val:
        return np.zeros_like(arr)

    normalized_array = -1 + 2 * (arr - min_val) / (max_val - min_val)
    return normalized_array


def initialize_terrain(size):
    return np.random.rand(size, size)


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


# Parámetros
terrain_size = 100

# Generar terreno inicial usando Markov Chain
initial_terrain = markov_chain_terrain(terrain_size, 100, 0.1)

plt.imshow(initial_terrain, cmap='terrain', origin='lower')
plt.colorbar()
plt.title('Terreno Generado con Markov Chain')

plt.tight_layout()
plt.show()
