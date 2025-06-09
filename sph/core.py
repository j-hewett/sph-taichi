import numpy as np
from .utils import *

class Simulator:
    def __init__(self, N_particles, s_width, s_height, g=-250, dt=0.05):

        self.N = N_particles
        self.s_height, self.s_width = s_height, s_width
        self.dt = dt
        self.g = np.array([0, g], dtype=np.float32)
        self.df = 0.6 ## Damping factor

        self.particle_radius = 5
        self.particle_mass = 1
        self.smoothingradius = 30

        self.positions = self.generate_positions(min_dist = (0.5 * self.particle_radius))
        self.velocities = np.zeros((self.N,2), dtype=np.float32)
        self.densities = np.ones(self.N) * 0.5

    def step(self):

        ## Apply gravity and predict future positions
        self.velocities += self.g * self.dt
        future_positions = self.positions + self.velocities * self.dt

        ## Update spatial lookup with fut pos
        cell_dict = update_spatial_lookup(future_positions,self.smoothingradius)
        
        ## Find neighbours, i = target, j = neighbours
        neighbours_i, neighbours_j = find_neighbours(future_positions, self.smoothingradius, cell_dict)

        ## Compute all distances at once
        pi = future_positions[neighbours_i]
        pj = future_positions[neighbours_j]
        distances = np.linalg.norm(pi - pj, axis=1)

        ## Compute kernel values and accumulate
        influences = SmoothingKernel(self.smoothingradius, distances) * self.particle_mass
        self.densities = np.zeros(self.N)
        np.add.at(self.densities, neighbours_i, influences)

        ## Add self contribution
        self_contrib = SmoothingKernel(self.smoothingradius, 0.0) * self.particle_mass
        self.densities += self_contrib

        P_force = calc_pressure_forces(future_positions, self.densities, neighbours_i, neighbours_j, self.smoothingradius, self.particle_mass)

        ## Update velocities (only where density is valid)
        valid = np.isfinite(self.densities) & (self.densities > 1e-6)
        P_accel = np.zeros_like(P_force)
        P_accel[valid] = P_force[valid] / self.densities[valid][:, np.newaxis]

        self.velocities += P_accel * self.dt

        ## Update positions
        self.positions += self.velocities * self.dt

        self.resolve_collisions()
        return self.positions, self.velocities

    def resolve_collisions(self):

        floor_y = -self.s_height/2 + self.particle_radius
        ceiling_y = self.s_height/2 - self.particle_radius
        left_x = -self.s_width/2 + self.particle_radius
        right_x = self.s_width/2 - self.particle_radius

        floor_mask = self.positions[:,1] <= floor_y
        ceiling_mask = self.positions[:,1] >= ceiling_y
        left_mask = self.positions[:,0] <= left_x
        right_mask = self.positions[:,0] >= right_x

        self.positions[floor_mask, 1] = floor_y
        self.velocities[floor_mask, 1] *= -self.df

        self.positions[ceiling_mask, 1] = ceiling_y
        self.velocities[ceiling_mask, 1] *= -self.df

        self.positions[left_mask, 0] = left_x
        self.velocities[left_mask, 0] *= -self.df

        self.positions[right_mask, 0] = right_x
        self.velocities[right_mask, 0] *= -self.df

    def generate_positions(self, min_dist=1e-2):
        positions = np.empty((self.N, 2))
        
        ## Generate first point
        positions[0] = [0,0]
        
        ## Generate remaining points with distance checking
        for i in range(1, self.N):
            while True:
                new_pos = np.random.uniform(
                    [-0.5 * self.s_width*0.65, -0.5 * self.s_height*0.65],
                    [0.5 * self.s_width*0.65, 0.5 * self.s_height*0.65]
                )
                
                ## Check distance to all existing points
                distances = np.sqrt(np.sum((positions[:i] - new_pos)**2, axis=1))
                if np.all(distances >= min_dist):
                    positions[i] = new_pos
                    break
                    
        return positions


            
