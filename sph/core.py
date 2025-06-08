import numpy as np
from .utils import *

class Simulator:
    def __init__(self, N_particles, s_width, s_height, g=0, dt=1/60):

        self.N = N_particles
        self.s_height, self.s_width = s_height, s_width
        self.dt = dt
        self.g = np.array([0, g], dtype=np.float32)
        self.df = 0.8  ## Damping factor

        self.particle_radius = 5
        self.particle_mass = 1
        self.smoothingradius = 100

        self.positions = self.generate_positions(min_dist = (0.5 * self.particle_radius))
        self.velocities = np.zeros((self.N,2), dtype=np.float32)
        self.densities = np.ones(self.N)
        #self.densities *= (self.s_width*self.s_height)/(self.N * self.particle_mass)
    def step(self):

        ## Apply gravity and predict future positions
        self.velocities += self.g * self.dt
        future_positions = self.positions + self.velocities * self.dt

        ## Update spatial lookup with fut pos
        cell_dict = update_spatial_lookup(future_positions,self.smoothingradius)
        
        all_neighbours = find_neighbours(future_positions, self.smoothingradius, cell_dict)

        for particle_idx in range(self.N):

            neighbour_idxs = all_neighbours[particle_idx]
            neighbour_positions = future_positions[neighbour_idxs]

            self.densities[particle_idx] = calc_density2(particle_idx, future_positions, 
                                                        neighbour_positions, self.smoothingradius, self.particle_mass
                                                        )
            P_force = calc_pressure_force2(particle_idx, self.densities[neighbour_idxs],
                                            neighbour_positions, self.densities, self.positions, self.smoothingradius,
                                            self.particle_mass)
            if np.isfinite(self.densities[particle_idx]) and self.densities[particle_idx] > 1e-6:
                P_accel = P_force / self.densities[particle_idx]
            else:
                P_accel = 0  # or maybe reset velocity/position
            self.velocities[particle_idx] += P_accel * self.dt

            self.positions[particle_idx] += self.velocities[particle_idx]*self.dt

        self.resolve_collisions()
        return self.positions

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
        
        # Generate first point randomly
        positions[0] = np.random.uniform(
            [-0.5 * self.s_width, -0.5 * self.s_height],
            [0.5 * self.s_width, 0.5 * self.s_height]
        )
        
        # Generate remaining points with distance checking
        for i in range(1, self.N):
            while True:
                new_pos = np.random.uniform(
                    [-0.5 * self.s_width, -0.5 * self.s_height],
                    [0.5 * self.s_width, 0.5 * self.s_height]
                )
                
                # Check distance to all existing points
                distances = np.sqrt(np.sum((positions[:i] - new_pos)**2, axis=1))
                if np.all(distances >= min_dist):
                    positions[i] = new_pos
                    break
                    
        return positions


            
