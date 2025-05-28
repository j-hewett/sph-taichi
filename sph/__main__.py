import pygame
import numpy as np
from sph.core import Particle, Simulator
from .utils import calc_density

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
PARTICLE_RADIUS = 5
BACKGROUND_COLOR = (30, 30, 30)
PARTICLE_COLOR = (100, 200, 255)
SIM_FLOOR_Y = -SCREEN_HEIGHT/2

R_SMOOTHING = 100

GRID_SIZE = 20
MAX_DENSITY = 20  # tune this to adjust brightness scaling


def sim_to_screen(pos):
    """Convert simulation coordinates to screen coordinates (flip Y)."""
    return int(pos[0] + SCREEN_WIDTH // 2), int(SCREEN_HEIGHT - (pos[1] + SCREEN_HEIGHT // 2))

def main():

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    n_particles = 100
    df = 0.8

    positions = (np.random.random((n_particles,2)) - 0.55) * np.array([800, 600])

    particles = [Particle(position=pos, velocity=[0,0]) for pos in positions]

    sim = Simulator(particles, dt=0.15)
    sim_started = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                sim_started = True

        if sim_started:
            sim.step()
            for p in particles:
                if p.position[1] <= SIM_FLOOR_Y + PARTICLE_RADIUS:
                    p.position[1] = SIM_FLOOR_Y + PARTICLE_RADIUS
                    p.velocity[1] *= -df
                elif p.position[1] >= SIM_FLOOR_Y + SCREEN_HEIGHT - PARTICLE_RADIUS:
                    p.position[1] = SIM_FLOOR_Y + SCREEN_HEIGHT - PARTICLE_RADIUS
                    p.velocity[1] *= -df
                if p.position[0] <= -SCREEN_WIDTH / 2 + PARTICLE_RADIUS:
                    p.position[0] = -SCREEN_WIDTH / 2 + PARTICLE_RADIUS
                    p.velocity[0] *= -df
                elif p.position[0] >= SCREEN_WIDTH / 2 - PARTICLE_RADIUS:
                    p.position[0] = SCREEN_WIDTH / 2 - PARTICLE_RADIUS
                    p.velocity[0] *= -df

        screen.fill(BACKGROUND_COLOR)

        for p in particles:           
            x,y = sim_to_screen(p.position)
            pygame.draw.circle(screen, PARTICLE_COLOR, (x,y), PARTICLE_RADIUS)
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
                    

if __name__ == "__main__":
    main()
