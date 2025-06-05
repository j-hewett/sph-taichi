import numpy as np
from .utils import *
import copy


class Particle:
    def __init__(self, position, velocity=None, mass=1.0):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity or [0.0, 0.0], dtype=np.float32)
        self.force = np.zeros(2, dtype=np.float32)
        self.mass = mass
        self.density = 0.01
    
    def update(self, dt):
        self.position += self.velocity*dt

class Simulator:
    def __init__(self, particles, dt=0.05):
        self.particles = particles
        self.dt = dt
        self.smoothingradius = 100
        self.positions = np.array([particle.position for particle in self.particles])
        self.densities = np.array([particle.density for particle in self.particles])
        self.g = np.array([0, 0], dtype=np.float32)

    def step(self):
        spatial_lookup, start_indices = update_spatial_lookup(self.particles,self.smoothingradius)
        particle_neighbours = {
            p: for_each_point_within_radius(p, self.particles, self.smoothingradius, spatial_lookup, start_indices)
            for p in self.particles
        }
        self.positions = np.array([particle.position for particle in self.particles])
        self.densities = np.array([particle.density for particle in self.particles])

        for i, particle in enumerate(self.particles):

            neighbours = particle_neighbours[particle]
            neighbour_densities = np.array([nbr.density for nbr in neighbours])
            neighbour_positions = np.array([nbr.position for nbr in neighbours])

            future_particle = copy.deepcopy(particle)
            future_particle.position = particle.position + particle.velocity * self.dt

            future_particle.density = calc_density2(future_particle, neighbour_positions, self.smoothingradius)
            print(future_particle.density)
            P_force = calc_pressure_force2(future_particle, neighbour_densities, neighbour_positions, self.smoothingradius)
            particle.density = future_particle.density

            P_accel = 0
            if particle.density>1e-6:
                P_accel = P_force / particle.density
            particle.velocity += (P_accel + self.g) * self.dt
            self.positions[i] += particle.velocity*self.dt
            particle.position = self.positions[i]
            self.particles[i] = particle
            
