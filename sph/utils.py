import numpy as np

GRAVITY = np.array([0, -9.81], dtype=np.float32)
RADIUS = 100

def apply_gravity(p):
    p.apply_force(p.mass * GRAVITY)

def SmoothingKernel(radius, d):
    vol = (np.pi * np.power(radius,4))/6
    return ((radius - d)**2)/vol

def d_SmoothingKernel(radius, d):
    if (d >= radius):
        return 0
    scale = 12 / (np.pi * np.power(radius,4))
    return (d - radius) * scale

def calc_density(p, particles): #Density at point q
    density = 0
    mass = 1

    for particle in particles:
        if (d := np.linalg.norm(particle.position - p.position)) < RADIUS:
            influence = SmoothingKernel(RADIUS, d)
            density += mass * influence
    return density

def calc_density_gradient(p, particles):
    d_density = 0
    mass = 1

    for particle in particles:
        if (d := np.linalg.norm(particle.position - p.position)) < RADIUS:
            r_particle = (p.position - particle.position)/d
            slope = d_SmoothingKernel(RADIUS, d)
            d_density += mass * slope * r_particle
    return d_density

#todo - replace 'q' with directly passing the particle itself
# That way I can access its density (p.density) easily later
# Also need to store density in particle instead of as array
# loop would then be for p in particles: rho = p.density etc
def calc_property(p, particles):
    A = 0
    mass = 1

    for particle in particles:
        if (d := np.linalg.norm(particle.position - p.position)) < RADIUS:
            influence = SmoothingKernel(RADIUS, d)
            A += mass * influence / particle.density
    return A

def calc_shared_pressure(rho_A, rho_B):
    pass

def calc_pressure_force(p, particles):
    d_P = 0
    mass = 1

    for particle in particles:
        if (d := np.linalg.norm(particle.position - p.position)) < RADIUS:
            if d > 1e-5:
                r_particle = (p.position - particle.position)/d
                slope = d_SmoothingKernel(RADIUS, d)
                if particle.density > 1e-5:
                    d_P += density_to_pressure(particle.density) * mass * slope * r_particle / particle.density
    return d_P

# def update_densities(particles):
#     for particle in particles:
#         particle.density = calc_density(particle.position, particles)
#     return particles

def density_to_pressure(density):
    t_density = 1
    p_multiplier = 5
    e_density = density - t_density
    pressure = e_density * p_multiplier
    return pressure
    