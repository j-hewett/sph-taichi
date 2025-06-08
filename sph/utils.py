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

def calc_pressure_force2(p_idx, neighbour_densities, neighbour_positions, densities, positions, radius, mass):
    distances = np.linalg.norm(neighbour_positions - positions[p_idx], axis=1)
    r_particles = np.where(distances[:, None] > 1e-6, (positions[p_idx] - neighbour_positions) / distances[:, None], 0.0)
    slopes = d_SmoothingKernel(radius, distances)
    shared_pressures = calc_shared_pressure(neighbour_densities, densities[p_idx])

    contributions = np.where(
    neighbour_densities[:, None] > 1e-6,  
    (shared_pressures * mass * slopes)[:, None] * r_particles / neighbour_densities[:, None],  # Safe division
    0.0  
    )   

    return np.sum(contributions, axis=0) 

def density_to_pressure(density):
    t_density = 0.5
    p_multiplier = 100
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
    all_neighbours = [[] for _ in range(n_particles)]
    radius_sq = radius ** 2

    for i in range(n_particles):
        x, y = positions[i]
        cx, cy = pos_to_cell_coord(np.array([[x, y]]), radius)[0]

        # Check all 3x3 neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell_key = (cx + dx, cy + dy)
                
                if cell_key not in cell_dict:
                    continue  # No particles in this cell
                
                # Check all particles in this cell
                for j in cell_dict[cell_key]:
                    if j == i:
                        continue  # Skip self
                    
                    delta = positions[i] - positions[j]
                    if delta @ delta <= radius_sq:
                        all_neighbours[i].append(j)
    
    return all_neighbours
