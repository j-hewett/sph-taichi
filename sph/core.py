import numpy as np
from .utils import *


class Particle:
    def __init__(self, position, velocity=None, mass=1.0):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity or [0.0, 0.0], dtype=np.float32)
        self.force = np.zeros(2, dtype=np.float32)
        self.mass = mass
        self.density = 0
    
    def update(self, dt):
        self.position += self.velocity*dt

class Simulator:
    def __init__(self, particles, dt=0.5):
        self.particles = particles
        self.dt = dt
        self.smoothingradius = 60

    def step(self):
        spatial_lookup, start_indices = update_spatial_lookup(self.particles,self.smoothingradius)
        particle_neighbours = {
            p: for_each_point_within_radius(p, self.particles, self.smoothingradius, spatial_lookup, start_indices)
            for p in self.particles
        }

        for particle in self.particles:

            apply_gravity(particle, self.dt)
            future_particle = particle
            future_particle.position = particle.position + particle.velocity*self.dt
            neighbours = particle_neighbours[future_particle]

            particle.density = calc_density(future_particle, neighbours, self.smoothingradius)

            P_force = calc_pressure_force(future_particle, neighbours, self.smoothingradius)
            P_accel = P_force / particle.density
            particle.velocity += P_accel * self.dt

            particle.update(self.dt)
            
