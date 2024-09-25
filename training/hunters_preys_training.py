from stable_baselines3.common.callbacks import BaseCallback
from hunter_prey import PreyHunterEnv
from stable_baselines3 import A2C

# Callback personalizado para renderizar el entorno durante el entrenamiento
class RenderCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        # Renderizar el entorno en cada paso
        self.env.render()
        return True

# Inicializar el entorno
env = PreyHunterEnv()

# Crear un modelo A2C
model = A2C("MlpPolicy", env, verbose=1)

# Entrenar el modelo y renderizar el entorno periódicamente
model.learn(total_timesteps=1000000, callback=RenderCallback(env))