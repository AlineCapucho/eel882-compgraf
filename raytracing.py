import numpy as np
import matplotlib.pyplot as plt

# Normaliza um dado vetor
def normalize(vector):
    return vector / np.linalg.norm(vector)

# Calcula a reflexão de um vetor em torno de um eixo
def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

# Calcular a interseção entre um raio e uma esfera
def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c

    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

# Encontrar o objeto mais próximo que intersecta com o raio
def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    
    # Itera sobre todas as distâncias e encontra a menor distância positiva.
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    
    return nearest_object, min_distance

# Responsável por criar a câmera dada uma posição
def create_camera(position):
    return np.array(position)

# Responsável por criar a fonte de luz, dadas suas propriedades.
def create_light(position, ambient, diffuse, specular):
    return {
        'position': np.array(position),
        'ambient': np.array(ambient),
        'diffuse': np.array(diffuse),
        'specular': np.array(specular)
    }

# Responsável por criar um objeto para a cena, dada as propriedades do objeto
def create_object(center, radius, ambient, diffuse, specular, shininess, reflection):
    return {
        'center': np.array(center),
        'radius': radius,
        'ambient': np.array(ambient),
        'diffuse': np.array(diffuse),
        'specular': np.array(specular),
        'shininess': shininess,
        'reflection': reflection
    }

def ray_tracing(height, width, max_depth, objects, light, camera):
    image = np.zeros((height, width, 3))  # Inicializa a imagem com pixels pretos
    ratio = float(width) / height  # Calcula a proporção da tela.
    screen = (-1, 1/ratio, 1, -1/ratio)  # Define as coordenadas da tela

    # Loop sobre cada pixel da imagem
    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            pixel = np.array([x, y, 0])  # Define a posição do pixel na tela
            origin = camera  # A origem do raio é a posição da câmera
            direction = normalize(pixel - origin)  # Normaliza a direção do raio

            color = np.zeros((3))  # Cor inicial do pixel
            reflection = 1 

            # Loop para calcular as reflexões até a profundidade máxima
            for _ in range(max_depth):
                nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
                
                # Se não houver interseção, sai do loop
                if nearest_object is None:
                    break

                intersection = origin + min_distance * direction  # Calcula o ponto de interseção
                normal_to_surface = normalize(intersection - nearest_object['center'])  # Normaliza o ponto de interseção
                shifted_point = intersection + 1e-5 * normal_to_surface  # Evita auto-interseção alterando (de forma pouco significativa) a posição do ponto
                intersection_to_light = normalize(light['position'] - shifted_point)  # Direção do ponto de interseção para a luz

                # Verifica se o ponto de interseção está na sombra de outro objeto
                _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
                intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
                is_shadowed = min_distance < intersection_to_light_distance

                # Se estiver na sombra, sai do loop
                if is_shadowed:
                    break

                illumination = np.zeros((3))  # Inicializa a iluminação

                illumination += nearest_object['ambient'] * light['ambient']
                illumination += nearest_object['diffuse'] * light['diffuse'] * max(np.dot(intersection_to_light, normal_to_surface), 0)
                intersection_to_camera = normalize(camera - intersection)
                H = normalize(intersection_to_light + intersection_to_camera)
                illumination += nearest_object['specular'] * light['specular'] * max(np.dot(normal_to_surface, H), 0) ** (nearest_object['shininess'] / 4)

                color += reflection * illumination
                reflection *= nearest_object['reflection']

                # Atualiza a origem e a direção do raio para calcular a reflexão
                origin = shifted_point
                direction = reflected(direction, normal_to_surface)

            image[i, j] = np.clip(color, 0, 1)

    return image

def main():
    camera_position = [0, 0, 1]

    light_position = [4, 4, 4]
    light_ambient = [1, 1, 1]
    light_diffuse = [1, 1, 1]
    light_specular = [1, 1, 1]

    object1_center = [-0.3, 0, -0.9]
    object1_radius = 0.85
    object1_ambient = [0, 0, 0.1]
    object1_diffuse = [0.2, 0.6, 0.1]
    object1_specular = [0.5, 1, 0.5]
    object1_shininess = 75
    object1_reflection = 0.1

    object2_center = [0.1, -0.3, 0]
    object2_radius = 0.2
    object2_ambient = [0.1, 0, 0.3]
    object2_diffuse = [0.7, 0, 0.7]
    object2_specular = [1, 1, 1]
    object2_shininess = 100
    object2_reflection = 0.5

    object3_center = [0, -7000, 0]
    object3_radius = 7000 - 0.85
    object3_ambient = [0.1, 0.1, 0.1]
    object3_diffuse = [0.6, 0.6, 0.6]
    object3_specular = [1, 1, 1]
    object3_shininess = 100
    object3_reflection = 0.5

    camera = create_camera(camera_position)

    light = create_light(light_position, light_ambient, light_diffuse, light_specular)

    objects = [
        create_object(object1_center, object1_radius, object1_ambient, object1_diffuse, object1_specular, object1_shininess, object1_reflection),
        create_object(object2_center, object2_radius, object2_ambient, object2_diffuse, object2_specular, object2_shininess, object2_reflection),
        create_object(object3_center, object3_radius, object3_ambient, object3_diffuse, object3_specular, object3_shininess, object3_reflection),
    ]

    width = 500
    height = 500
    max_depth = 5

    image = ray_tracing(height, width, max_depth, objects, light, camera)

    plt.imsave('imagem.jpg', image)

if __name__ == "__main__":
    main()
