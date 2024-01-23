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


def generate_worley_noise(size, num_points):
    # Inicializar la matriz con valores grandes
    noise_matrix = np.ones((size, size)) * float('inf')

    # Generar puntos aleatorios
    points = np.random.rand(num_points, 2) * size

    for i in range(size):
        for j in range(size):
            for p in points:
                # Calcular la distancia euclidiana
                distance = np.sqrt((p[0] - i)**2 + (p[1] - j)**2)

                # Actualizar el valor si la distancia es menor
                if distance < noise_matrix[i, j]:
                    noise_matrix[i, j] = distance

    return noise_matrix


# Tamaño de la matriz y número de puntos
size = 100
num_points = np.random.randint(10, 20)

# Generar Worley Noise
worley_noise = generate_worley_noise(size, num_points)

# Visualizar Worley Noise
plt.imshow(worley_noise, cmap='terrain', origin='lower')
plt.colorbar()
plt.title(f"Worley Noise with {num_points} points")
plt.show()


# Función de evaluación basada en la pendiente del terreno
def evaluate_terrain(terrain):
    # Calcular la pendiente del terreno

    gradient_x, gradient_y = np.gradient(terrain)
    val = gradient_x**2 + gradient_y**2

    slope = np.sqrt(gradient_x**2 + gradient_y**2)
    # En este ejemplo, la función de evaluación es la suma de las pendientes
    return np.sum(slope)


def terrain_cost_proximity(terrain, threshold=0.3, penalty_factor=10, proximity_factor=10):
    # Compute the gradient of the terrain
    gradient_x, gradient_y = np.gradient(terrain)

    # Calculate the magnitude of the gradient vector at each point
    slope = np.sqrt(gradient_x**2 + gradient_y**2)

    # Apply a penalty for slopes above the threshold
    threshold_reward = np.maximum(0, slope - threshold)

    # Calculate penalties based on proximity of values
    proximity_penalty = proximity_factor * \
        np.sum(np.abs(np.diff(terrain, axis=0)))
    proximity_penalty += proximity_factor * \
        np.sum(np.abs(np.diff(terrain, axis=1)))

    # Calculate the overall cost as the sum of penalties
    cost = np.sum(threshold_reward) * penalty_factor - proximity_penalty

    return cost

# Simulated Annealing para generar terreno realista


def simulated_annealing(initial_terrain, iterations, initial_temperature, cooling_rate):
    current_terrain = initial_terrain.copy()
    current_energy = evaluate_terrain(current_terrain)
    a, b = np.shape(initial_terrain)

    for iteration in range(iterations):
        # Generar un nuevo terreno vecino
        new_terrain = current_terrain + generate_worley_noise(a, 1)
        new_energy = evaluate_terrain(new_terrain)

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
terrain_size = 200

scale = 20.0
octaves = 6
persistence = 0.5
lacunarity = 2.0
seed = 42
iterations = 1000
initial_temperature = 1.0
cooling_rate = 0.9

# Generar terreno inicial usando ruido Perlin
initial_terrain = generate_random_matrix(size)

# Aplicar Simulated Annealing
final_terrain1 = simulated_annealing(
    initial_terrain, iterations, initial_temperature, cooling_rate)
final_terrain1 = normalize_array(final_terrain1)

plt.figure(figsize=(12, 6))

plt.imshow(initial_terrain, cmap='terrain', origin='lower')
plt.colorbar()
plt.title('Terreno Inicial (in tratamiento)')

# Visualizar terreno generado

# Visualizar Worley Noise con puntos
plt.scatter(*np.where(final_terrain1 == 0),
            c='red', marker='o', label='Points')
plt.imshow(final_terrain1, cmap='terrain', origin='lower')
plt.colorbar(label='Distance to nearest point')
plt.title(f"Worley Noise with {num_points} points")
plt.legend()

plt.tight_layout()
plt.show()
