import pygame
from stable_baselines3 import DQN
from hunters import HunterPreyEnv

# Inicializar Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))

# Cargar el modelo entrenado
model = DQN.load("hunter_prey_dqn")  # Cambia el nombre del archivo si es diferente

# Crear el entorno de juego
env = HunterPreyEnv()

# Reiniciar el entorno para obtener la primera observación
obs, info = env.reset()

# Crear un reloj para controlar los FPS
clock = pygame.time.Clock()

# Bucle principal del juego real
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Obtener la acción predicha por el modelo entrenado
    action, _states = model.predict(obs, deterministic=True)

    # Ejecutar la acción en el entorno
    obs, reward, terminated, truncated, info = env.step(action)

    # Renderizar el entorno para visualizarlo
    env.render()

    # Limitar la velocidad del bucle a 30 FPS
    clock.tick(60)

    # Reiniciar el entorno si el episodio termina
    if terminated or truncated:
        obs, info = env.reset()

# Cerrar Pygame cuando se termine el juego
pygame.quit()