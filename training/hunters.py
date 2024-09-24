import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np

WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))


class HunterPreyEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(HunterPreyEnv, self).__init__()
        self.capture_count = 0

        # Define action and observation space
        # The actions could be: move up, down, left, right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # The observation space is the relative position of the hunter and prey
        # (relative x, relative y)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Prey variables
        self.num_preys = 5
        self.prey_positions = [np.random.rand(2).astype(np.float32) for _ in range(self.num_preys)]
        self.prey_speed = 0.1  # Prey speed (you can make this dynamic)

        # Hunter variables
        self.hunter_pos = np.array([0.0, 0.0])  # Hunter starts at the center
        self.hunter_speed = 0.1  # Hunter speed
        self.hunter_size = 25

        # Obstacles
        self.num_obstacles = 5
        self.obstacle_size = 40
        self.obstacle_positions = [self._generate_obstacle_position() for _ in range(self.num_obstacles)]

    def _generate_obstacle_position(self):
        """Genera una posición para un obstáculo que esté dentro de los límites de la pantalla."""
        # Asegurar que el obstáculo no esté demasiado cerca de los bordes
        margin = self.obstacle_size / WIDTH  # Ajusta según el tamaño de la ventana (por ejemplo, 800x600)
        return np.random.rand(2) * (1 - 2 * margin) + margin

    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation."""
        super().reset(seed=seed)

        # Reset positions
        self.hunter_pos = np.array([0.0, 0.0])
        self.prey_positions = [np.random.rand(2).astype(np.float32) for _ in range(self.num_preys)]

        # Reset hunter speed and size
        self.hunter_speed = 0.1  # Restablecer velocidad inicial del hunter
        self.hunter_size = 25
        # Return the initial observation
        return self._get_observation(), {}

    def step(self, action):
        """Execute one time step within the environment."""

        # Calcular nueva posición del hunter antes de moverlo
        new_hunter_pos = np.copy(self.hunter_pos)
        if action == 0:  # Mover hacia arriba
            new_hunter_pos[1] += self.hunter_speed
        elif action == 1:  # Mover hacia abajo
            new_hunter_pos[1] -= self.hunter_speed
        elif action == 2:  # Mover hacia la izquierda
            new_hunter_pos[0] -= self.hunter_speed
        elif action == 3:  # Mover hacia la derecha
            new_hunter_pos[0] += self.hunter_speed

        # Verificar colisión con los obstáculos
        if not self._check_collision(new_hunter_pos, self.obstacle_positions):
            self.hunter_pos = new_hunter_pos  # Mover solo si no hay colisión

        # Limitar la posición del hunter dentro de los límites
        self.hunter_pos[0] = np.clip(self.hunter_pos[0], 0, 1)
        self.hunter_pos[1] = np.clip(self.hunter_pos[1], 0, 1)

        # Mover cada prey de forma independiente
        for i in range(len(self.prey_positions)):
            new_prey_pos = self.prey_positions[i] + (np.random.rand(2) - 0.5) * self.prey_speed

            # Verificar colisión con los obstáculos
            if not self._check_collision(new_prey_pos, self.obstacle_positions):
                self.prey_positions[i] = new_prey_pos  # Mover solo si no hay colisión
            self.prey_positions[i] = np.clip(self.prey_positions[i], 0, 1)
        # Check if there are preys
        if not self.prey_positions:
            return self._get_observation(), 0, True, False, {}  # Si no hay preys, termina el episodio

        # Get distance to closest prey
        closest_prey_index, closest_distance = self._get_closest_prey()

        if closest_prey_index is None:  # If not preys, we end the step.
            return self._get_observation(), 0, True, False, {}

        reward = -closest_distance
        terminated = bool(closest_distance < 0.1)

        if terminated:
            # Pop captured prey
            self.capture_count += 1
            print(f"Total count: {self.capture_count}")
            self.hunter_speed = min(self.hunter_speed + 0.001, 0.1)
            self.prey_positions.pop(closest_prey_index)
            # If not preys, end step.
            if not self.prey_positions:
                return self._get_observation(), reward, True, False, {}

        return self._get_observation(), reward, False, False, {}

    def _get_closest_prey(self):
        """Find the nearest prey to the hunter."""
        if not self.prey_positions:  # Check if list is not empty
            return None, float('inf')  # If not preys, we return inf distance.
        distances = [np.linalg.norm(self.hunter_pos - prey_pos) for prey_pos in self.prey_positions]
        closest_index = np.argmin(distances)
        return closest_index, distances[closest_index]

    def _get_observation(self):
        """Obtener la posición relativa al prey más cercano."""
        closest_prey_index, _ = self._get_closest_prey()

        if closest_prey_index is None:  # Si no hay preys, devolver una observación por defecto
            return np.zeros(2, dtype=np.float32)  # Devolver un array de ceros como observación

        # Si hay preys, devolver la posición relativa al más cercano
        return (self.prey_positions[closest_prey_index] - self.hunter_pos).astype(np.float32)

    def _check_collision(self, pos, obstacle_positions):
        """Verifica si la posición 'pos' colisiona con alguno de los obstáculos."""
        for obstacle_pos in obstacle_positions:
            if np.linalg.norm(pos - obstacle_pos) < 0.05:  # Ajusta el umbral según el tamaño del obstáculo
                return True  # Colisión detectada
        return False

    def render(self):
        """Render the environment (for visualization)."""

        # Limpiar la pantalla con un fondo blanco
        screen.fill((255, 255, 255))

        # Dibujar al hunter (cuadrado rojo)
        hunter_x = int(self.hunter_pos[0] * WIDTH)
        hunter_y = int(self.hunter_pos[1] * HEIGHT)
        pygame.draw.rect(screen, (255, 0, 0), (hunter_x, hunter_y, 20, 20))

        # Dibujar cada prey (cuadrado verde)
        for prey_pos in self.prey_positions:
            prey_x = int(prey_pos[0] * WIDTH)
            prey_y = int(prey_pos[1] * HEIGHT)
            pygame.draw.rect(screen, (0, 255, 0), (prey_x, prey_y, 20, 20))

        # Dibujar cada obstáculo (cuadrado gris)
        for obstacle_pos in self.obstacle_positions:
            obstacle_x = int(obstacle_pos[0] * WIDTH)
            obstacle_y = int(obstacle_pos[1] * HEIGHT)
            pygame.draw.rect(screen, (128, 128, 128), (obstacle_x, obstacle_y, self.obstacle_size, self.obstacle_size))

        # Mostrar el contador de capturas
        font = pygame.font.SysFont(None, 36)
        text = font.render(f"Capturas: {self.capture_count}", True, (0, 0, 0))
        screen.blit(text, (10, 10))

        # Actualizar la pantalla para mostrar los cambios
        pygame.display.flip()

    def close(self):
        pass
