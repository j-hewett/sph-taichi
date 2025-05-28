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
        #self.densities = np.zeros(len(particles), dtype=np.float32)

    def step(self):
        for i, p in enumerate(self.particles):

            apply_gravity(p)

            self.densities[i] = calc_density(p.position, self.particles)

            P_force = calc_pressure_force(p.position, self.particles, self.densities)
            P_accel = P_force / self.densities[i]
            p.apply_force(P_accel * p.mass)

            p.update(self.dt)
            
