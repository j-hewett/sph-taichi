import numpy as np

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

def calc_density_gradient2(p_idx, positions, neighbour_positions, radius, mass): #Vectorized
    distances = np.linalg.norm(neighbour_positions - positions[p_idx], axis=1)
    r_particles = np.where(distances[:, None] > 1e-6, (positions[p_idx] - neighbour_positions) / distances[:, None], 0.0)
    slopes = d_SmoothingKernel(radius, distances)
    return np.sum(mass * slopes[:, None] * r_particles, axis=0)

def calc_shared_pressure(density_A, density_B):
    pressure_A = density_to_pressure(density_A)
    pressure_B = density_to_pressure(density_B)
    return (pressure_A + pressure_B) / 2

def calc_pressure_forces(future_positions, densities, flat_i, flat_j, radius, mass):
    pi = future_positions[flat_i]
    pj = future_positions[flat_j]
    rho_i = densities[flat_i]
    rho_j = densities[flat_j]

    delta = pi - pj
    distances = np.linalg.norm(delta, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        r_particles = np.where(distances[:, None] > 1e-6, delta / distances[:, None], 0.0)

    slopes = d_SmoothingKernel(radius, distances)
    shared_P = calc_shared_pressure(rho_j, rho_i)

    with np.errstate(divide='ignore', invalid='ignore'):
        pressure_contribs = np.where(
            rho_j[:, None] > 1e-6,
            (shared_P * mass * slopes)[:, None] * r_particles / rho_j[:, None],
            0.0
        )

    P_force = np.zeros((densities.shape[0], 2))
    np.add.at(P_force, flat_i, pressure_contribs)

    return P_force

def density_to_pressure(density):
    t_density = 0.5
    p_multiplier = 1000
    e_density = density - t_density
    pressure = e_density * p_multiplier
    return pressure
    
def update_spatial_lookup(positions, radius):
    cell_coords = pos_to_cell_coord(positions, radius)
    
    # Build a dictionary: { (cell_x, cell_y): [particle_indices] }
    cell_dict = {}
    for idx, (x, y) in enumerate(cell_coords):
        key = (x, y)
        if key not in cell_dict:
            cell_dict[key] = []
        cell_dict[key].append(idx)
    
    return cell_dict

def pos_to_cell_coord(pos_array, radius):
    return np.floor(pos_array / radius).astype(int)

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
