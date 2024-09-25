import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np

WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))


class PreyEnv(gym.Env):
    """Entorno para entrenar a los preys a alejarse del cazador."""

    def __init__(self):
        super(PreyEnv, self).__init__()

        self.action_space = spaces.Discrete(4)  # Acciones del prey
        self.observation_space = spaces.Box(low=-1, high=1, shape=(12,),
                                            dtype=np.float32)  # 2 para el prey y 2*3 para los hunters

        self.prey_pos = np.random.rand(2).astype(np.float32)
        self.num_hunters = 5  # Número de hunters
        self.hunter_positions = [np.random.rand(2).astype(np.float32) for _ in range(self.num_hunters)]

        self.num_obstacles = 2
        self.obstacle_positions = [np.random.rand(2).astype(np.float32) for _ in range(self.num_obstacles)]
        self.prey_speed = 0.03
        self.hunter_speed = 0.01
        self.hunter_directions = [np.random.choice([0, 1, 2, 3]) for _ in
                                  range(self.num_hunters)]  # Direcciones iniciales aleatorias
        self.steps_until_change = [50 for _ in
                                   range(self.num_hunters)]  # Cada hunter cambiará su dirección cada 50 pasos

    def _check_collision(self, pos):
        """Verificar si la nueva posición colisiona con algún obstáculo."""
        for obstacle_pos in self.obstacle_positions:
            if np.linalg.norm(pos - obstacle_pos) < 0.05:
                return True
        return False

    def _move_hunters_randomly_or_towards_prey(self):
        """Mover a cada hunter aleatoriamente o hacia el prey si está cerca."""
        for i in range(self.num_hunters):
            direction_to_prey = self.prey_pos - self.hunter_positions[i]
            distance_to_prey = np.linalg.norm(direction_to_prey)

            # Si el prey está dentro de un rango cercano, mover hacia él
            if distance_to_prey < 0.4:
                direction = direction_to_prey / distance_to_prey  # Normalizar la dirección
                new_hunter_pos = self.hunter_positions[i] + direction * self.hunter_speed
            else:
                # Movimiento aleatorio si el prey está lejos
                self.steps_until_change[i] -= 1
                if self.steps_until_change[i] <= 0:
                    self.hunter_directions[i] = np.random.choice([0, 1, 2, 3])
                    self.steps_until_change[i] = 50  # Reiniciar el número de pasos

                new_hunter_pos = np.copy(self.hunter_positions[i])
                if self.hunter_directions[i] == 0:  # Mover hacia arriba
                    new_hunter_pos[1] += self.hunter_speed
                elif self.hunter_directions[i] == 1:  # Mover hacia abajo
                    new_hunter_pos[1] -= self.hunter_speed
                elif self.hunter_directions[i] == 2:  # Mover hacia la izquierda
                    new_hunter_pos[0] -= self.hunter_speed
                elif self.hunter_directions[i] == 3:  # Mover hacia la derecha
                    new_hunter_pos[0] += self.hunter_speed

            # Verificar colisión con los obstáculos
            if not self._check_collision(new_hunter_pos):
                self.hunter_positions[i] = new_hunter_pos  # Mover solo si no hay colisión

            # Asegurarse de que el hunter se mantenga dentro de los límites del mapa
            self.hunter_positions[i] = np.clip(self.hunter_positions[i], 0, 1)

    def reset(self, seed=None, options=None):
        """Reiniciar el entorno."""
        super().reset(seed=seed)

        self.prey_pos = np.random.rand(2).astype(np.float32)
        self.hunter_positions = [np.random.rand(2).astype(np.float32) for _ in range(self.num_hunters)]
        self.obstacle_positions = [np.random.rand(2).astype(np.float32) for _ in range(self.num_obstacles)]
        return self._get_observation(), {}

    def step(self, action):
        """Ejecutar un paso del entorno."""

        # Movimiento del prey basado en la acción (si no hay cazadores cerca)
        move_x, move_y = 0, 0
        if action == 0:  # Mover hacia arriba
            move_y = self.prey_speed
        elif action == 1:  # Mover hacia abajo
            move_y = -self.prey_speed
        elif action == 2:  # Mover hacia la izquierda
            move_x = -self.prey_speed
        elif action == 3:  # Mover hacia la derecha
            move_x = self.prey_speed

        # Detectar proximidad de los cazadores
        escape_vector = np.zeros(2, dtype=np.float32)  # Vector de escape

        for hunter_pos in self.hunter_positions:
            direction_to_prey = self.prey_pos - hunter_pos
            distance_to_hunter = np.linalg.norm(direction_to_prey)

            # Si algún cazador está cerca (por ejemplo, a menos de 0.2 de distancia), calcular vector de escape
            if distance_to_hunter < 0.2:
                escape_vector += direction_to_prey / (distance_to_hunter + 1e-6)  # Normalizar y sumar

        # Si existe un vector de escape (es decir, hay hunters cerca), moverse en esa dirección
        if np.linalg.norm(escape_vector) > 0:
            # Normalizar el vector de escape para mover el prey
            escape_vector /= np.linalg.norm(escape_vector)
            self.prey_pos += escape_vector * self.prey_speed
        else:
            # Movimiento normal si no hay cazadores cerca
            self.prey_pos[0] += move_x
            self.prey_pos[1] += move_y

        # Limitar la posición del prey a un rango más limitado (márgenes de seguridad)
        self.prey_pos = np.clip(self.prey_pos, 0, 1)

        # Mover a los hunters aleatoriamente o hacia el prey si están cerca
        self._move_hunters_randomly_or_towards_prey()

        # Limitar la posición de los hunters a un rango más limitado
        for i in range(self.num_hunters):
            self.hunter_positions[i] = np.clip(self.hunter_positions[i], 0, 1)

        # Recompensa por alejarse de los hunters (promedio de las distancias a los hunters)
        total_distance_to_hunters = sum(
            [np.linalg.norm(self.prey_pos - hunter_pos) for hunter_pos in self.hunter_positions])
        reward = total_distance_to_hunters / self.num_hunters  # Más lejos de los hunters es mejor para el prey

        # Recompensa adicional por mantenerse cerca de los obstáculos
        for obstacle_pos in self.obstacle_positions:
            distance_to_obstacle = np.linalg.norm(self.prey_pos - obstacle_pos)
            if distance_to_obstacle < 0.05:
                reward += 0.1

        # Condición de terminación: si algún cazador atrapa al prey
        terminated = any(np.linalg.norm(self.prey_pos - hunter_pos) < 0.05 for hunter_pos in self.hunter_positions)

        if terminated:
            reward -= 1.0  # Penalización significativa si el prey es atrapado

        truncated = False  # No estamos implementando truncamiento basado en tiempo

        # Obtener la observación actualizada
        obs = self._get_observation()

        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        """Obtener la observación (posición del prey y de todos los hunters)."""
        return np.concatenate([self.prey_pos] + self.hunter_positions).astype(np.float32)

    def render(self):
        """Render the environment for visualization using Pygame."""

        # Inicializar Pygame si no lo has hecho antes
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))  # Tamaño de la ventana
            pygame.display.set_caption("Prey vs Hunter")

        # Limpiar la pantalla con un fondo blanco
        self.screen.fill((255, 255, 255))

        # Escalar las posiciones (convertir de [0.1, 0.9] a píxeles en la pantalla, con márgenes)
        def scale_position(pos):
            x = int(0.8 * 800 * (pos[0] - 0.1)) + 80  # Mantener un margen de 10% en los bordes
            y = int(0.8 * 600 * (pos[1] - 0.1)) + 60
            return x, y

        # Dibujar al prey (cuadrado verde)
        prey_x, prey_y = scale_position(self.prey_pos)
        pygame.draw.rect(self.screen, (0, 255, 0), (prey_x, prey_y, 20, 20))

        # Dibujar a todos los cazadores (cuadrados rojos)
        for hunter_pos in self.hunter_positions:
            hunter_x, hunter_y = scale_position(hunter_pos)
            pygame.draw.rect(self.screen, (255, 0, 0), (hunter_x, hunter_y, 20, 20))

        # Dibujar los obstáculos (cuadrados grises)
        for obstacle_pos in self.obstacle_positions:
            obstacle_x, obstacle_y = scale_position(obstacle_pos)
            pygame.draw.rect(self.screen, (128, 128, 128), (obstacle_x, obstacle_y, 20, 20))

        # Actualizar la pantalla
        pygame.display.flip()