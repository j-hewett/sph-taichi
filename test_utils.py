import unittest
import numpy as np
from sph.core import Particle
from sph.utils import *

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.radius = 50
        self.p0 = Particle(position  = np.array([0.0, 0.0]))

        self.particles = np.array([
            Particle(position=np.array([25.0, 0.0])),
            Particle(position=np.array([14.0, 45.0])),
            Particle(position=np.array([-52.0, 23.0])),
        ])

        self.full_particles = np.array([self.p0] + list(self.particles))

        self.positions = np.array([p.position for p in self.particles])

        for particle in self.full_particles:
            particle.density = calc_density(particle, self.full_particles, self.radius)
        
        self.densities = np.array([p.density for p in self.particles])

    def test_density_gradient_equivalence(self):
        grad1 = calc_density_gradient(self.p0, self.particles, self.radius)
        grad2 = calc_density_gradient2(self.p0, self.positions, self.radius)

        np.testing.assert_allclose(grad1, grad2, rtol=1e-5, atol=1e-8)

    def test_density_equivalence(self):
        rho1 = calc_density(self.p0, self.particles, self.radius)
        rho2 = calc_density2(self.p0, self.positions, self.radius)

        np.testing.assert_equal(rho1, rho2)

    def test_pressure_force_equivalence(self):
        PF1 = calc_pressure_force(self.p0, self.particles, self.radius)
        PF2 = calc_pressure_force2(self.p0, self.densities, self.positions, self.radius)

        np.testing.assert_allclose(PF1, PF2, rtol=1e-5, atol=1e-8)
if __name__ == '__main__':
    unittest.main()