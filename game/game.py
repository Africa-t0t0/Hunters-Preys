import time
import pygame
import random

# Initialize pygame
pygame.init()

# Define colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Define window size
WIDTH = 800
HEIGHT = 600

HUNTERS = 3
PREYS = 15

# Create the window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hunters vs Prey")


# Define the class for the characters (squares)
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


# Create sprite groups
all_sprites = pygame.sprite.Group()
hunters = pygame.sprite.Group()
preys = pygame.sprite.Group()

# Create hunters (red)
for i in range(HUNTERS):  # Adjust the number of hunters here
    x = random.randint(0, WIDTH - 20)
    y = random.randint(0, HEIGHT - 20)
    hunter = Square(RED, x, y)
    all_sprites.add(hunter)
    hunters.add(hunter)

# Create prey (green)
for i in range(PREYS):  # Adjust the number of prey here
    x = random.randint(0, WIDTH - 20)
    y = random.randint(0, HEIGHT - 20)
    prey_item = Square(GREEN, x, y)
    all_sprites.add(prey_item)
    preys.add(prey_item)

# Main loop
running = True
clock = pygame.time.Clock()
start_time = time.time()

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    current_time = time.time()
    # Check if 5 seconds have passed
    if current_time - start_time >= 5:
        # Increase prey speed slightly
        for prey in preys:
            prey.speed_up_slightly()
        # Reset timer
        start_time = current_time

    # Update sprites
    all_sprites.update()

    # Detect collisions between hunters and prey
    for hunter in hunters:
        collisions = pygame.sprite.spritecollide(hunter, preys, True)  # True removes the prey after collision
        if collisions:
            print(f"A hunter caught {len(collisions)} prey!")
            hunter.grow_and_speed_up()
            # You can add more logic here, such as increasing points or changing states

    # Draw white background and the sprites
    screen.fill(WHITE)
    all_sprites.draw(screen)

    # Update display
    pygame.display.flip()

    # Control update speed (FPS)
    clock.tick(60)

# Quit pygame
pygame.quit()