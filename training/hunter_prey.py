import pygame
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PreyHunterEnv(gym.Env):
    def __init__(self):
        super(PreyHunterEnv, self).__init__()

        self.action_space = spaces.MultiDiscrete([4] * 6)  # 3 preys + 3 hunters

        # Espacio de observación: 2 para cada prey y cada hunter (posiciones x, y)
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)

        # Inicializar las posiciones de 3 preys y 3 hunters
        self.num_preys = 3
        self.num_hunters = 3
        self.prey_positions = [np.random.rand(2).astype(np.float32) for _ in range(self.num_preys)]
        self.hunter_positions = [np.random.rand(2).astype(np.float32) for _ in range(self.num_hunters)]

        # Velocidades
        self.prey_speed = 0.005
        self.hunter_speed = 0.01

    def reset(self, seed=None, options=None):
        """Reiniciar el entorno."""
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Reiniciar las posiciones de los preys y hunters
        self.prey_positions = [self.np_random.random(2).astype(np.float32) for _ in range(self.num_preys)]
        self.hunter_positions = [self.np_random.random(2).astype(np.float32) for _ in range(self.num_hunters)]

        return self._get_observation(), {}

    def _get_observation(self):
        """Obtener la observación de todos los preys y hunters."""
        return np.concatenate(self.prey_positions + self.hunter_positions).astype(np.float32)

    def step(self, action):
        """Ejecutar un paso en el entorno para todos los preys y hunters."""
        # Separar las acciones de los preys y los hunters
        prey_actions = action[:self.num_preys]
        hunter_actions = action[self.num_preys:]

        # Movimiento de los preys basado en las acciones
        for i, prey_action in enumerate(prey_actions):
            if prey_action == 0:  # Mover hacia arriba
                self.prey_positions[i][1] += self.prey_speed
            elif prey_action == 1:  # Mover hacia abajo
                self.prey_positions[i][1] -= self.prey_speed
            elif prey_action == 2:  # Mover hacia la izquierda
                self.prey_positions[i][0] -= self.prey_speed
            elif prey_action == 3:  # Mover hacia la derecha
                self.prey_positions[i][0] += self.prey_speed

        # Movimiento de los hunters basado en las acciones
        for i, hunter_action in enumerate(hunter_actions):
            if hunter_action == 0:  # Mover hacia arriba
                self.hunter_positions[i][1] += self.hunter_speed
            elif hunter_action == 1:  # Mover hacia abajo
                self.hunter_positions[i][1] -= self.hunter_speed
            elif hunter_action == 2:  # Mover hacia la izquierda
                self.hunter_positions[i][0] -= self.hunter_speed
            elif hunter_action == 3:  # Mover hacia la derecha
                self.hunter_positions[i][0] += self.hunter_speed

        # Limitar las posiciones al rango [0, 1]
        self.prey_positions = [np.clip(pos, 0, 1) for pos in self.prey_positions]
        self.hunter_positions = [np.clip(pos, 0, 1) for pos in self.hunter_positions]

        # Recompensas
        total_prey_reward = 0
        total_hunter_reward = 0

        for prey_pos in self.prey_positions:
            for hunter_pos in self.hunter_positions:
                distance = np.linalg.norm(prey_pos - hunter_pos)
                total_prey_reward += distance  # Preys son recompensados por estar lejos de los hunters
                total_hunter_reward -= distance  # Hunters son recompensados por acercarse a los preys

        # Unificar las recompensas sumando las de preys y hunters
        total_reward = total_prey_reward + total_hunter_reward

        # Condiciones de terminación
        terminated = any(
            np.linalg.norm(prey_pos - hunter_pos) < 0.05 for prey_pos in self.prey_positions for hunter_pos in
            self.hunter_positions)
        truncated = False  # Puedes cambiar esto si implementas un límite de tiempo
        info = {}

        # Devolver las observaciones, recompensa unificada, estado de terminación, truncamiento y info
        return self._get_observation(), total_reward, terminated, truncated, info

    def render(self, mode='human'):
        """Renderizar el entorno usando Pygame."""

        # Inicializar Pygame si no lo has hecho antes
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))  # Tamaño de la ventana
            pygame.display.set_caption("Prey vs Hunters")

        # Manejo de eventos para evitar que la ventana se congele
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Limpiar la pantalla con un fondo blanco
        self.screen.fill((255, 255, 255))

        # Dibujar los preys (cuadrados verdes)
        for prey_pos in self.prey_positions:
            prey_x = int(prey_pos[0] * 800)
            prey_y = int(prey_pos[1] * 600)
            pygame.draw.rect(self.screen, (0, 255, 0), (prey_x, prey_y, 20, 20))

        # Dibujar los hunters (cuadrados rojos)
        for hunter_pos in self.hunter_positions:
            hunter_x = int(hunter_pos[0] * 800)
            hunter_y = int(hunter_pos[1] * 600)
            pygame.draw.rect(self.screen, (255, 0, 0), (hunter_x, hunter_y, 20, 20))

        # Actualizar la pantalla
        pygame.display.flip()