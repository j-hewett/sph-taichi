import taichi as ti
from sph.core import Simulator

SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720

@ti.data_oriented
class Renderer:
    def __init__(self, n_particles):
        self.canvas = ti.Vector.field(3, dtype=ti.f32, shape=(SCREEN_WIDTH, SCREEN_HEIGHT))
        self.screen_pos = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
        self.radius = 3
        self.radius_sq = self.radius * self.radius
        self.bg_color = ti.Vector([0.07, 0.18, 0.25])  ## Dark blue
        self.particle_color = ti.Vector([0.1, 0.8, 0.5])  ## White

    @ti.kernel
    def clear_canvas(self):
        for i, j in self.canvas:
            self.canvas[i, j] = self.bg_color

    @ti.kernel
    def draw_particles(self):
        for p in range(self.screen_pos.shape[0]):
            x = ti.cast(self.screen_pos[p][0] * SCREEN_WIDTH, ti.i32)
            y = ti.cast(self.screen_pos[p][1] * SCREEN_HEIGHT, ti.i32)
            
            ## Draw circle using bounding box
            for dy in range(-self.radius, self.radius + 1):
                for dx in range(-self.radius, self.radius + 1):
                    px = x + dx
                    py = y + dy
                    if 0 <= px < SCREEN_WIDTH and 0 <= py < SCREEN_HEIGHT:
                        if dx*dx + dy*dy <= self.radius_sq:
                            self.canvas[px, py] = self.particle_color

def main():
    n_particles = 6000
    gui = ti.GUI("SPH Fluid Simulation", res=(SCREEN_WIDTH, SCREEN_HEIGHT), fast_gui=True)
    renderer = Renderer(n_particles)

    dt = 1/120

    sim = Simulator(n_particles, SCREEN_WIDTH, SCREEN_HEIGHT, 3, dt)

    while gui.running:
        ## Update renderer
        renderer.screen_pos = sim.step(dt)

        renderer.clear_canvas()
        renderer.draw_particles()
        
        ## Display
        gui.set_image(renderer.canvas)
        gui.show()

if __name__ == "__main__":
    main()