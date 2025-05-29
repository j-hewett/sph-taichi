import numpy as np
from .utils import *

class Particle:
    def __init__(self, position, velocity=None, mass=1.0):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity or [0.0, 0.0], dtype=np.float32)
        self.force = np.zeros(2, dtype=np.float32)
        self.mass = mass
        self.density = 0

    def apply_force(self, force):
        self.force += force
    
    def update(self, dt):
        acceleration = self.force / self.mass
        self.velocity += acceleration*dt
        self.position += self.velocity*dt
        self.force[:] = 0

class Simulator:
    def __init__(self, particles, dt=0.01):
        self.particles = particles
        self.dt = dt

    def step(self):
        for particle in self.particles:

            apply_gravity(particle)

            particle.density = calc_density(particle, self.particles)

            P_force = calc_pressure_force(particle, self.particles)
            P_accel = P_force / particle.density
            particle.apply_force(P_accel * particle.mass)

            particle.update(self.dt)
            
