import numpy as np
from collections import defaultdict

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def resolve_collisions(positions, velocities, particle_radius, s_width, s_height, df):
    floor_y = -s_height/2 + particle_radius
    ceiling_y = s_height/2 - particle_radius
    left_x = -s_width/2 + particle_radius
    right_x = s_width/2 - particle_radius

    floor_mask = positions[:,1] <= floor_y
    ceiling_mask = positions[:,1] >= ceiling_y
    left_mask = positions[:,0] <= left_x
    right_mask = positions[:,0] >= right_x

    positions[floor_mask, 1] = floor_y
    velocities[floor_mask, 1] *= -df

    positions[ceiling_mask, 1] = ceiling_y
    velocities[ceiling_mask, 1] *= -df

    positions[left_mask, 0] = left_x
    velocities[left_mask, 0] *= -df

    positions[right_mask, 0] = right_x
    velocities[right_mask, 0] *= -df

    return positions, velocities

def SmoothingKernel(radius, d):
    vol = (np.pi * np.power(radius,4))/6
    return np.where(d <= radius, ((radius - d)**2) / vol, 0.0)

def d_SmoothingKernel(radius, d):
    scale = 12 / (np.pi * np.power(radius,4))
    return np.where(d <= radius, (d - radius) * scale, 0.0)

def calc_density2(p_idx, positions, neighbour_positions, radius, mass): #Vectorized
    self_contribution = SmoothingKernel(radius, 0.0) * mass  # Self contribution
    distances = np.linalg.norm(neighbour_positions - positions[p_idx], axis=1)
    influences = SmoothingKernel(radius, distances)
    return self_contribution + np.sum(mass*influences) #sum of densities

def calc_shared_pressure(density_A, density_B):
    pressure_A = density_to_pressure(density_A)
    pressure_B = density_to_pressure(density_B)
    return (pressure_A + pressure_B) / 2

def calc_pressure_forces(future_positions, densities, flat_i, flat_j, radius, mass):
    pi = future_positions[flat_i]
    pj = future_positions[flat_j]
    rho_i = densities[flat_i]
    rho_j = densities[flat_j]
    mass = np.float32(mass)
    radius = np.float32(radius)

    delta = pi - pj
    r_particles = normalized(delta, axis=1)
    slopes = d_SmoothingKernel(radius, np.linalg.norm(delta, axis=1))

    shared_P = calc_shared_pressure(rho_j, rho_i)

    pressure_contribs = np.where(
            rho_j[:, None] > 1e-6,
            (shared_P * mass * slopes)[:, None] * r_particles / rho_j[:, None],
            0.0
        )

    P_force = np.zeros((densities.shape[0], 2))
    np.add.at(P_force, flat_i, pressure_contribs)

    return P_force

def calc_viscosity_forces(future_positions, velocities, flat_i, flat_j, radius):
    pi = future_positions[flat_i]
    pj = future_positions[flat_j]

    radius = np.float32(radius)

    delta = pi - pj
    slopes = d_SmoothingKernel(radius, np.linalg.norm(delta, axis=1))

    viscosity_contribs = (velocities[flat_j] - velocities[flat_i]) * slopes[:, None]

    V_force = np.zeros((velocities.shape[0], 2))
    np.add.at(V_force, flat_i, viscosity_contribs)

    return V_force



def density_to_pressure(density):
    t_density = np.float32(0.5)
    p_multiplier = np.float32(3000)
    e_density = density - t_density
    pressure = e_density * p_multiplier
    return pressure


def update_spatial_lookup(positions, radius):
    cell_coords = pos_to_cell_coord(positions, radius)
    
    cell_keys = [tuple(cell) for cell in cell_coords]

    cell_dict = defaultdict(list)
    for idx, key in enumerate(cell_keys):
        cell_dict[key].append(idx)
    
    return dict(cell_dict)

def pos_to_cell_coord(pos_array, radius):
    return np.floor_divide(pos_array, radius)

def hash_cell(cell_x, cell_y):
    return (cell_x * 15823) + (cell_y * 9737333)

def get_key_from_hash(h, length):
    return (h % length)

def find_neighbours(positions, radius, cell_dict):
    n_particles = positions.shape[0]
    radius_sq = radius ** 2
    flat_i = []
    flat_j = []

    neighbor_offsets = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]

    for i in range(n_particles):
        x, y = positions[i]
        cx, cy = pos_to_cell_coord(np.array([[x, y]]), radius)[0]

        for dx, dy in neighbor_offsets:
            cell_key = (cx + dx, cy + dy)

            if cell_key not in cell_dict:
                continue

            for j in cell_dict[cell_key]:
                if j == i:
                    continue
                delta = positions[i] - positions[j]
                if delta @ delta <= radius_sq:
                    flat_i.append(i)
                    flat_j.append(j)

    return np.array(flat_i), np.array(flat_j)