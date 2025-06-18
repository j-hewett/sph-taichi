import numpy as np
from .utils import *

class Simulator:
    def __init__(self, N_particles, s_width, s_height, particle_radius, dt, g=-100):

        self.N = N_particles
        self.s_height, self.s_width = s_height, s_width
        self.dt = dt
        self.g = np.array([0, g], dtype=np.float32)
        self.df = 0.6 ## Damping factor
        self.viscosity = 0.5

        self.particle_radius = particle_radius
        self.mass = np.float32(1)
        self.sradius = np.float32(2 * 2 * particle_radius)

        self.positions = self.generate_positions(min_dist = (0.5 * self.particle_radius))
        self.velocities = np.zeros((self.N,2), dtype=np.float32)
        self.densities = np.ones(self.N) * 0.5

        self.floor_y = -s_height/2 + particle_radius
        self.ceiling_y = s_height/2 - particle_radius
        self.left_x = -s_width/2 + particle_radius
        self.right_x = s_width/2 - particle_radius

    def step(self):

        ## Apply gravity and predict future positions
        self.velocities += self.g * self.dt
        future_positions = self.positions + self.velocities * self.dt ## const lookahead time

        ## Update spatial lookup with fut pos
        cell_dict = update_spatial_lookup(future_positions,self.sradius)
        
        ## Find neighbours, i = target, j = neighbours
        neighbours_i = []
        neighbours_j = []

        neighbor_offsets = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]

        for i in range(self.N):
            x, y = future_positions[i]
            cx, cy = pos_to_cell_coord(np.array([[x, y]]), self.sradius)[0]

            for dx, dy in neighbor_offsets:
                cell_key = (cx + dx, cy + dy)

                if cell_key not in cell_dict:
                    continue

                for j in cell_dict[cell_key]:
                    if j == i:
                        continue
                    delta = future_positions[i] - future_positions[j]
                    if delta @ delta <= (self.sradius ** 2):
                        neighbours_i.append(i)
                        neighbours_j.append(j)

        neighbours_i = np.array(neighbours_i)
        neighbours_j = np.array(neighbours_j)

        ## Compute all distances at once
        pi = future_positions[neighbours_i]
        pj = future_positions[neighbours_j]
        distances = np.linalg.norm(pi - pj, axis=1)

        ## Compute kernel values and accumulate
        influences = SmoothingKernel(self.sradius, distances) * self.mass
        self.densities = np.zeros(self.N)
        np.add.at(self.densities, neighbours_i, influences)

        ## Add self contribution
        self_contrib = SmoothingKernel(self.sradius, 0.0) * self.mass
        self.densities += self_contrib

        ## Calculate pressure forces
        rho_i = self.densities[neighbours_i]
        rho_j = self.densities[neighbours_j]
        delta = pi - pj
        r_particles = normalized(delta, axis=1)
        slopes = d_SmoothingKernel(self.sradius, np.linalg.norm(delta, axis=1))

        shared_P = calc_shared_pressure(rho_j, rho_i)

        pressure_contribs = np.where(
                rho_j[:, None] > 1e-6,
                (shared_P * self.mass * slopes)[:, None] * r_particles / rho_j[:, None],
                0.0
            )

        P_force = np.zeros((self.densities.shape[0], 2))
        np.add.at(P_force, neighbours_i, pressure_contribs)

        ## Update velocities (only where density is valid)
        valid = np.isfinite(self.densities) & (self.densities > 1e-6)
        P_accel = np.zeros_like(P_force)
        P_accel[valid] = P_force[valid] / self.densities[valid][:, np.newaxis]

        ## Calculate viscosity forces
        viscosity_contribs = (self.velocities[neighbours_j] - self.velocities[neighbours_i]) * slopes[:, None]
        V_force = np.zeros((self.velocities.shape[0], 2))
        np.add.at(V_force, neighbours_i, viscosity_contribs)
        
        V_force *= self.viscosity
        V_accel = np.zeros_like(V_force)
        V_accel[valid] = V_force[valid] / self.densities[valid][:, np.newaxis]

        ## Update velocities
        self.velocities += (P_accel + V_accel) * self.dt

        ## Update positions
        self.positions += self.velocities * self.dt
        
        ## Resolve collisions
        floor_mask = self.positions[:,1] <= self.floor_y
        ceiling_mask = self.positions[:,1] >= self.ceiling_y
        left_mask = self.positions[:,0] <= self.left_x
        right_mask = self.positions[:,0] >= self.right_x

        self.positions[floor_mask, 1] = self.floor_y
        self.velocities[floor_mask, 1] *= -self.df

        self.positions[ceiling_mask, 1] = self.ceiling_y
        self.velocities[ceiling_mask, 1] *= -self.df

        self.positions[left_mask, 0] = self.left_x
        self.velocities[left_mask, 0] *= -self.df

        self.positions[right_mask, 0] = self.right_x
        self.velocities[right_mask, 0] *= -self.df

        return self.positions, self.velocities

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