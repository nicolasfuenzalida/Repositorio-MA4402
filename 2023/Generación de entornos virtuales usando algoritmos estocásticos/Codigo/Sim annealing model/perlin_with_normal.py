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


def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)

    # Avoid division by zero
    if min_val == max_val:
        return np.zeros_like(arr)

    normalized_array = -1 + 2 * (arr - min_val) / (max_val - min_val)
    return normalized_array


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

# Función de evaluación basada en la pendiente del terreno


def evaluate_terrain(terrain):
    # Calcular la pendiente del terreno

    gradient_x, gradient_y = np.gradient(terrain)
    val = gradient_x**2 + gradient_y**2

    slope = np.sqrt(gradient_x**2 + gradient_y**2)
    # En este ejemplo, la función de evaluación es la suma de las pendientes
    return np.sum(slope)

# Simulated Annealing para generar terreno realista


def simulated_annealing(initial_terrain, iterations, initial_temperature, cooling_rate):
    current_terrain = initial_terrain.copy()
    current_energy = evaluate_terrain(current_terrain)

    for iteration in range(iterations):
        # Generar un nuevo terreno vecino
        new_terrain = current_terrain + \
            np.random.normal(loc=0.3, scale=0.01, size=current_terrain.shape)
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


def ploteo3d(terrain, size):

    # Set max number of pixel to: 'None' to prevent errors. Its not nice, but works for that case. Big images will load RAM+CPU heavily (like DecompressionBomb)
    # Image.MAX_IMAGE_PIXELS = None # first we set no limit to open
    # img = Image.open(source_file_dem)

    # get aspect ratio of tif file for late plot box-plot-ratio
    y_ratio, x_ratio = size, size

    # open georeference TIF file

    # create arrays and declare x,y,z variables
    lin_x = np.linspace(0, 500, terrain.shape[0], endpoint=False)
    lin_y = np.linspace(0, 500, terrain.shape[1], endpoint=False)
    y, x = np.meshgrid(lin_y, lin_x)

    z = terrain

    # Apply gaussian filter, with sigmas as variables. Higher sigma = more smoothing and more calculations. Downside: min and max values do change due to smoothing
    sigma_y = 1
    sigma_x = 1
    sigma = [sigma_y, sigma_x]
    z_smoothed = sp.ndimage.gaussian_filter(z, sigma)

    # Some min and max and range values coming from gaussian_filter calculations
    z_smoothed_min = np.amin(z_smoothed)
    z_smoothed_max = np.amax(z_smoothed)
    z_range = z_smoothed_max - z_smoothed_min

    # Creating figure
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection='3d')
    ax.azim = -30
    ax.elev = 0
    ax.set_box_aspect((x_ratio, y_ratio, ((x_ratio+y_ratio)/8)))
    # ax.arrow3D(1,1,z_smoothed_max, -1,0,1, mutation_scale=20, ec ='black', fc='red') #draw arrow to "north" which is not correct north. But with georeferenced sources it should work
    surf = ax.plot_surface(x, y, z_smoothed, cmap='terrain', edgecolor='none')
    # setting colors for colorbar range
    m = cm.ScalarMappable(cmap=surf.cmap, norm=surf.norm)
    m.set_array(z_smoothed)
    # cbar = fig.colorbar(m, shrink=0.5, aspect=20, ticks=[z_smoothed_min, 0, (z_range*0.25+z_smoothed_min), (z_range*0.5+z_smoothed_min), (z_range*0.75+z_smoothed_min), z_smoothed_max])
    # cbar.ax.set_yticklabels([f'{z_smoothed_min}', ' ',  f'{(z_range*0.25+z_smoothed_min)}', f'{(z_range*0.5+z_smoothed_min)}', f'{(z_range*0.75+z_smoothed_min)}', f'{z_smoothed_max}'])
    # plt.xticks([])  # disabling xticks by Setting xticks to an empty list
    # plt.yticks([])  # disabling yticks by setting yticks to an empty list
    # draw flat rectangle at z = 0 to indicate where mean sea level is in 3d
    # x_rectangle = [0,1,1,0]
    # y_rectangle = [0,0,1,1]
    # z_rectangle = [0,0,0,0]
    # verts = [list(zip(x_rectangle,y_rectangle,z_rectangle))]
    # ax.add_collection3d(Poly3DCollection(verts, alpha=0.5))
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    fig.tight_layout()


plt.show()

# Parámetros
terrain_size = 50
scale = 20.0
octaves = 6
persistence = 0.5
lacunarity = 2.0
seed = 42
iterations = 1000
initial_temperature = 1.0
cooling_rate = 0.7

# Generar terreno inicial usando ruido Perlin
initial_terrain = generate_perlin_terrain(
    terrain_size, scale, octaves, persistence, lacunarity, seed)

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

ploteo3d(initial_terrain, terrain_size)
ploteo3d(final_terrain, terrain_size)
plt.tight_layout()
plt.show()
