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


def generate_random_matrix(size):
    return np.random.uniform(low=-1, high=1, size=(size, size))


# Función para generar un terreno inicial aleatorio usando ruido Perlin
def generate_perlin_terrain(size, scale, octaves, persistence, lacunarity, seed):
    # genera un Array 2D de elementos de rango [-1,1] usando perlin noise:
    # Size : las dimensiones del terreno generado, en este caso solo puede ser un cuadrado
    # Scale : el grado de zoom que tendrá el terreno
    # Octave : agrega detalles a las superficie, por ejemplo octave 1 pueden ser las montañas,
    # octave 2 pueden ser las rocas, son como multiples pasadas al terreno para agregarle detalle
    # Lacuranity : ajusta la frequencia en la que se agrega detalle en octave,
    # un valor deseable suele ser 2
    # Persistence : determina la influencia que tiene cada octave
    terrain = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            terrain[i][j] = noise.pnoise2(i/scale, j/scale, octaves=octaves, persistence=persistence,
                                          lacunarity=lacunarity, repeatx=size, repeaty=size, base=seed)
    return terrain


def costo_terreno(terreno):
    """
    Función de costo para suavizar un terreno en Simulated Annealing.

    Parámetros:
    - terreno: numpy array, matriz que representa el terreno con alturas.

    Retorna:
    - float, valor del costo.
    """
    # Ajusta los pesos según sea necesario
    w_homogeneidad = 1.0
    w_pendiente = 1.0
    w_extremos = 1.0

    # Parámetro para evitar divisiones por cero
    epsilon = 1e-8

    # Calcula la homogeneidad local (puedes ajustar la ventana según tus necesidades)
    homogeneidad_local = np.std(terreno, axis=(0, 1))

    # Calcula la penalización de homogeneidad
    penalizacion_homogeneidad = 1.0 / (np.mean(homogeneidad_local) + epsilon)

    # Calcula la penalización de pendiente en ambas direcciones (horizontal y vertical)
    penalizacion_pendiente = np.sum(
        np.abs(np.diff(terreno, axis=0))) + np.sum(np.abs(np.diff(terreno, axis=1)))

    # Calcula la recompensa de extremos
    extremos = np.array([terreno[0, 0], terreno[0, -1],
                        terreno[-1, 0], terreno[-1, -1]])
    recompensa_extremos = np.sum(
        np.exp(-np.abs(np.tile(extremos, len(terreno.flatten()) // 4) - terreno.flatten())**2))

    # Calcula el costo total
    costo_total = w_homogeneidad * penalizacion_homogeneidad + w_pendiente * \
        penalizacion_pendiente + w_extremos * recompensa_extremos

    return costo_total

# Simulated Annealing para generar terreno realista


def simulated_annealing(initial_terrain, iterations, initial_temperature, cooling_rate):
    current_terrain = initial_terrain.copy()
    current_energy = costo_terreno(current_terrain)

    for iteration in range(iterations):
        # Generar un nuevo terreno vecino
        new_terrain = current_terrain + \
            generate_random_matrix(size=current_terrain.shape)
        new_energy = costo_terreno(new_terrain)

        # Calcular la diferencia de energía
        energy_difference = new_energy - current_energy

        # Decidir si aceptar el nuevo terreno
        if energy_difference < 0 or np.random.rand() < np.exp(-energy_difference / (initial_temperature - 1e-15)):
            current_terrain = new_terrain
            current_energy = new_energy

        # Enfriar la temperatura
        initial_temperature *= cooling_rate

    return current_terrain


# Parámetros
terrain_size = 500
scale = 20.0
octaves = 6
persistence = 0.5
lacunarity = 2.0
seed = 42
iterations = 1000
initial_temperature = 1.0
cooling_rate = 0.7

# Generar terreno inicial usando ruido Perlin
initial_terrain = generate_random_matrix(terrain_size)

# Aplicar Simulated Annealing
final_terrain = simulated_annealing(
    initial_terrain, iterations, initial_temperature, cooling_rate)
final_terrain = normalize_array(final_terrain)
plt.figure(figsize=(12, 6))

plt.imshow(initial_terrain, cmap='terrain', origin='lower')
plt.colorbar()
plt.title('Terreno Generado con Ruido Perlin')

# Visualizar terreno generado
plt.figure()
plt.imshow(final_terrain, cmap='terrain', origin='lower')
plt.colorbar()
plt.title('Terreno Generado con Simulated Annealing y Ruido Perlin')

plt.tight_layout()
plt.show()
