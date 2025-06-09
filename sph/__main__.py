import pygame
import numpy as np
from sph.core import Simulator

SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 600
PARTICLE_RADIUS = 5
BACKGROUND_COLOR = (30, 30, 30)
PARTICLE_COLOR = (100, 200, 255)
SIM_FLOOR_Y = -SCREEN_HEIGHT/2


def sim_to_screen(positions):
    screen_x = positions[:, 0] + SCREEN_WIDTH / 2
    screen_y = SCREEN_HEIGHT - (positions[:, 1] + SCREEN_HEIGHT / 2)
    return np.stack((screen_x, screen_y), axis=1).astype(np.int32)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    n_particles = 600
    
    start_color = np.array([100, 200, 255])
    end_color = np.array([255, 100, 100])
    max_cap = 100.0

    sim = Simulator(n_particles, SCREEN_WIDTH, SCREEN_HEIGHT, dt=0.025)
    sim_started = False
    positions = sim.positions.copy()
    velocities = sim.velocities.copy()
    running = True
    while running:
        userForce = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                sim_started = True

        if sim_started:
            positions, velocities = sim.step()
        speeds = np.linalg.norm(velocities, axis=1)
        capped_speeds = np.clip(speeds, 0, max_cap)
        norm_speeds = capped_speeds / max_cap

        ## Interpolate colors
        t = norm_speeds[:, None]  # shape (N,1)
        colors = ((1 - t) * start_color + t * end_color).astype(np.uint8)

        ## Draw particles
        screen.fill(BACKGROUND_COLOR)
        screen_positions = sim_to_screen(positions)
        for pos, color in zip(screen_positions, colors):
            pygame.draw.circle(screen, color, pos, PARTICLE_RADIUS)
        pygame.display.flip()
        clock.tick()
        print(clock.get_fps(), end='\r')
    
    pygame.quit()
                    

if __name__ == "__main__":
    main()
