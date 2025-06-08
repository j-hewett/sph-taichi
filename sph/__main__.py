import pygame
import numpy as np
from sph.core import Simulator

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
PARTICLE_RADIUS = 5
BACKGROUND_COLOR = (30, 30, 30)
PARTICLE_COLOR = (100, 200, 255)
SIM_FLOOR_Y = -SCREEN_HEIGHT/2

GRID_SIZE = 20


def sim_to_screen(positions):
    """
    Convert simulation coordinates to screen coordinates (flip Y) for a (N, 2) array.
    """
    screen_x = positions[:, 0] + SCREEN_WIDTH / 2
    screen_y = SCREEN_HEIGHT - (positions[:, 1] + SCREEN_HEIGHT / 2)
    return np.stack((screen_x, screen_y), axis=1).astype(np.int32)

def main():

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    n_particles = 300
    sim = Simulator(n_particles, SCREEN_WIDTH, SCREEN_HEIGHT, dt=1/60)
    sim_started = False
    positions = sim.positions.copy()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                sim_started = True

        if sim_started:
            positions = sim.step()
            print(len(positions))
        screen.fill(BACKGROUND_COLOR)
        screen_positions = sim_to_screen(positions)
        for pos in screen_positions:
            pygame.draw.circle(screen, PARTICLE_COLOR, pos, PARTICLE_RADIUS)
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
                    

if __name__ == "__main__":
    main()
