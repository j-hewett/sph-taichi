import taichi as ti


ti.init(arch=ti.gpu)  ## Use GPU acceleration


@ti.data_oriented
class Simulator:
    def __init__(self, N_particles, s_width, s_height, particle_radius, dt, g=-500):
        self.N = N_particles
        self.s_height, self.s_width = s_height, s_width
        self.dt = dt
        self.lookahead = 1/120
        self.g = ti.Vector([0, g], dt=ti.f32)
        self.df = 0.3 ## Damping factor
        self.viscosity = 0.4

        self.t_density = 10
        self.p_multiplier = 5000

        self.particle_radius = particle_radius
        self.mass = 10
        self.sradius = 2.5 * 2 * particle_radius

        ## Taichi fields for particle data
        self.positions = ti.Vector.field(2, dtype=ti.f32, shape=N_particles)
        self.velocities = ti.Vector.field(2, dtype=ti.f32, shape=N_particles)
        self.densities = ti.field(dtype=ti.f32, shape=N_particles)
        self.pressure_forces = ti.Vector.field(2, dtype=ti.f32, shape=N_particles)
        self.viscosity_forces = ti.Vector.field(2, dtype=ti.f32, shape=N_particles)

        self.future_positions = ti.Vector.field(2, dtype=ti.f32, shape=self.N)

        ## Spatial hashing
        self.max_particles_per_cell = 128
        self.grid_size = 256
        self.cell_list = ti.field(dtype=ti.i32, shape=(self.grid_size, self.grid_size, self.max_particles_per_cell))
        self.cell_count = ti.field(dtype=ti.i32, shape=(self.grid_size, self.grid_size))
        self.cell_coords = ti.Vector.field(2, dtype=ti.i32, shape=self.N)

        ## Boundary conditions
        self.floor_y = 0 + particle_radius
        self.ceiling_y = s_height - particle_radius
        self.left_x = 0 + particle_radius
        self.right_x = s_width - particle_radius

        self.screen_positions = ti.Vector.field(2, ti.f32, shape=self.N)

        self.initialize_particles()

    @ti.kernel
    def initialize_particles(self):
        particles_per_row = int(ti.sqrt(self.N))  # or use floor division for full rows
        spacing_x = self.s_width * 0.25 / particles_per_row
        spacing_y = self.s_height * 0.25 / particles_per_row

        for i in range(self.N):
            row = i // particles_per_row
            col = i % particles_per_row
            self.positions[i] = ti.Vector([
                col * spacing_x + self.s_width * 0.40,  # offset from left
                row * spacing_y + self.s_height * 0.40  # offset from bottom
            ])
            self.velocities[i] = ti.Vector([0.0, 0.0])
            self.densities[i] = 1

    @ti.func
    def smoothing_kernel(self, radius, d):
        vol = (ti.math.pi * ti.pow(radius, 4)) / 6
        result = 0.0
        if d <= radius:
            result = ti.pow(radius - d, 2) / vol
        return result

    @ti.func
    def d_smoothing_kernel(self, radius, d):
        scale = 12 / (ti.math.pi * ti.pow(radius, 4))
        result = 0.0
        if d <= radius:
            result = (d - radius) * scale
        return result

    @ti.func
    def density_to_pressure(self, density):
        e_density = density - self.t_density
        return e_density * self.p_multiplier

    @ti.func
    def calc_shared_pressure(self, density_A, density_B):
        pressure_A = self.density_to_pressure(density_A)
        pressure_B = self.density_to_pressure(density_B)
        return (pressure_A + pressure_B) / 2

    @ti.func
    def pos_to_cell_coord(self, pos):
        cell_x = int(ti.floor(pos.x / self.sradius))
        cell_y = int(ti.floor(pos.y / self.sradius))
        cell_x = ti.max(0, ti.min(self.grid_size - 1, cell_x))
        cell_y = ti.max(0, ti.min(self.grid_size - 1, cell_y))
        return cell_x, cell_y
    
    @ti.kernel
    def predict_positions_and_cells(self):
        for i in range(self.N):
            future_pos = self.positions[i] + self.velocities[i] * self.lookahead
            self.future_positions[i] = future_pos

            cell_x, cell_y = self.pos_to_cell_coord(future_pos)
            self.cell_coords[i] = ti.Vector([cell_x, cell_y])

    @ti.kernel
    def update_spatial_lookup(self):
        for i, j in ti.ndrange(self.grid_size, self.grid_size):
            self.cell_count[i, j] = 0

        for p in range(self.N):
            cell = self.cell_coords[p]
            old_count = ti.atomic_add(self.cell_count[cell.x, cell.y], 1)
            if old_count < self.max_particles_per_cell:
                self.cell_list[cell.x, cell.y, old_count] = p

    @ti.kernel
    def compute_densities_and_forces(self):
        ## Reset values
        for i in range(self.N):
            self.densities[i] = 0.0
            self.pressure_forces[i] = ti.Vector([0.0, 0.0])
            self.viscosity_forces[i] = ti.Vector([0.0, 0.0])

        ## Main computation
        for i in range(self.N):
            pos_i = self.future_positions[i]
            vel_i = self.velocities[i]
            cell = self.cell_coords[i]
            
            ## Self-contribution to density
            self.densities[i] += self.smoothing_kernel(self.sradius, 0.0) * self.mass
            
            ## Neighbor processing
            for dx, dy in ti.static(ti.ndrange(3, 3)):
                nx, ny = cell.x + dx - 1, cell.y + dy - 1 
                
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    for k in range(self.cell_count[nx, ny]):
                        j = self.cell_list[nx, ny, k]
                        if i == j:
                            continue
                            
                        pos_j = self.future_positions[j]
                        delta = pos_i - pos_j
                        distance = delta.norm()
                        
                        if distance <= self.sradius and distance > 1e-6:
                            kernel_val = self.smoothing_kernel(self.sradius, distance)
                            self.densities[i] += kernel_val * self.mass

                            r_normalized = delta / distance
                            slope = self.d_smoothing_kernel(self.sradius, distance)
                            
                            if self.densities[j] > 1e-6:
                                shared_p = self.calc_shared_pressure(self.densities[j], self.densities[i])
                                pforce_ij = (shared_p * self.mass * slope) * r_normalized / self.densities[j]
                                self.pressure_forces[i] += pforce_ij
                                self.pressure_forces[j] -= pforce_ij

                            
                            vel_diff = self.velocities[j] - vel_i
                            vforce_ij = vel_diff * (self.viscosity * slope)
                            self.viscosity_forces[i] += vforce_ij
                            self.viscosity_forces[j] -= vforce_ij

    @ti.kernel
    def integrate(self):
        for i in range(self.N):
            ## Apply gravity
            self.velocities[i] += self.g * self.dt
            
            ## Apply pressure and viscosity forces
            if self.densities[i] > 1e-6:
                pressure_accel = self.pressure_forces[i] / self.densities[i]
                viscosity_accel = self.viscosity_forces[i] / self.densities[i]
                self.velocities[i] += (pressure_accel + viscosity_accel) * self.dt
            
            ## Update positions
            self.positions[i] += self.velocities[i] * self.dt
            
            ## Handle boundary collisions
            if self.positions[i].y <= self.floor_y:
                self.positions[i].y = self.floor_y
                self.velocities[i].y *= -self.df
            
            if self.positions[i].y >= self.ceiling_y:
                self.positions[i].y = self.ceiling_y
                self.velocities[i].y *= -self.df
            
            if self.positions[i].x <= self.left_x:
                self.positions[i].x = self.left_x
                self.velocities[i].x *= -self.df
            
            if self.positions[i].x >= self.right_x:
                self.positions[i].x = self.right_x
                self.velocities[i].x *= -self.df

    @ti.kernel
    def compute_screen_positions(self):
        for i in range(self.N):
            pos = self.positions[i]
            self.screen_positions[i] = pos / ti.Vector([self.s_width, self.s_height])

    @ti.kernel
    def log_avg_density(self):
        total = 0.0
        for i in range(self.N):
            total += self.densities[i]
        print("Average density:", total / self.N, end='\r')

    def step(self, dt):
        self.dt = dt
        self.predict_positions_and_cells()
        self.update_spatial_lookup()
        self.compute_densities_and_forces()
        self.integrate()
        self.log_avg_density()
        self.compute_screen_positions()
        return self.screen_positions