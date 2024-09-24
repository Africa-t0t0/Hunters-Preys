import time
import pygame

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


from hunters import HunterPreyEnv

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))

# Create the environment
env = HunterPreyEnv()
env = Monitor(env)
# Check that the environment follows the Gym API
check_env(env, warn=True)

# Initialize the DQN model
model = DQN("MlpPolicy", env, verbose=1)

# Train the model for 10000 steps
model.learn(total_timesteps=10000)

# Save the trained model
model.save("hunter_prey_dqn")
clock = pygame.time.Clock()

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Test the trained agent (optional)
obs, info = env.reset()  # env.reset() devuelve una tupla (obs, info), pero solo necesitamos `obs`
for _ in range(100000):
    action, _states = model.predict(obs)  # Pasa solo `obs` a model.predict()
    obs, rewards, terminated, truncated, info = env.step(action)  # Paso en el entorno
    env.render()  # Visualizar el progreso
    clock.tick(30)
    time.sleep(0.02)  # Puedes ajustar este valor para pausar más o menos tiempo

    if terminated or truncated:
        obs, info = env.reset()  # Al reiniciar, asegúrate de extraer solo `obs`

pygame.quit()  # Quit pygame once done