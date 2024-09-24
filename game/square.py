import pygame
import random


WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Define window size
WIDTH = 800
HEIGHT = 600

HUNTERS = 3
PREYS = 15


class Square(pygame.sprite.Sprite):
    def __init__(self, color, x, y):
        super().__init__()
        self.size = 20
        self.image = pygame.Surface([20, 20])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed_x = random.randint(-3, 3)
        self.speed_y = random.randint(-3, 3)
        self.color = color  # Store the color for later

    def update(self):
        # Movement in any direction
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        # Bounce off the edges
        if self.rect.right > WIDTH or self.rect.left < 0:
            self.speed_x *= -1
        if self.rect.bottom > HEIGHT or self.rect.top < 0:
            self.speed_y *= -1

    def grow_and_speed_up(self):
        self.size += 2
        self.image = pygame.Surface([self.size, self.size])
        self.image.fill(self.color)
        self.rect = self.image.get_rect(center=self.rect.center)

        # Increase speed
        if self.speed_x > 0:
            self.speed_x += 1
        else:
            self.speed_x -= 1

        if self.speed_y > 0:
            self.speed_y += 1
        else:
            self.speed_y -= 1

    def speed_up_slightly(self):
        # Slightly increase the speed (small increment)
        if self.speed_x > 0:
            self.speed_x += 0.2
        else:
            self.speed_x -= 0.2

        if self.speed_y > 0:
            self.speed_y += 0.2
        else:
            self.speed_y -= 0.2