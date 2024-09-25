import time
import pygame

from stable_baselines3 import DQN
from preys import PreyEnv

# Crear el entorno de entrenamiento para los preys
env = PreyEnv()

# Entrenar el modelo DQN para que los preys aprendan a mantenerse alejados
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

# Guardar el modelo entrenado
model.save("prey_dqn_model")


clock = pygame.time.Clock()

# En tu bucle principal, asegúrate de limitar los FPS
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    # Limitar la velocidad de renderizado a 30 FPS
    clock.tick(30)
    done = terminated or truncated

    if done:
        env.reset()
