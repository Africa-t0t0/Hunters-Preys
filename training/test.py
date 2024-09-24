import pygame

from stable_baselines3 import DQN

from hunters import HunterPreyEnv

# Cargar el modelo entrenado
model = DQN.load("hunter_prey_dqn")

# Inicializar Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))

# Crear el entorno
env = HunterPreyEnv()

# Reiniciar el entorno para obtener la primera observación
obs, info = env.reset()

clock = pygame.time.Clock()

# En el bucle principal, agrega control de FPS:
for _ in range(1000):
    # Lógica del juego
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # Renderizar el entorno para visualizarlo
    env.render()

    # Limitar la velocidad del bucle a 30 FPS
    clock.tick(30)

    if terminated or truncated:
        obs, info = env.reset()

# Cerrar Pygame después de la prueba
pygame.quit()