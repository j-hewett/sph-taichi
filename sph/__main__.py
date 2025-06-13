import glfw
import numpy as np
import moderngl
from sph.core import Simulator

SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 600

def sim_to_ndc(positions):
    x = positions[:, 0] / (SCREEN_WIDTH / 2)
    y = positions[:, 1] / (SCREEN_HEIGHT / 2)
    return np.stack((x, y), axis=1).astype('f4')

def main():
    try:

        #Initialise GLFW
        if not glfw.init():
            raise Exception("GLFW could not be initialised.")
        
        window = glfw.create_window(SCREEN_WIDTH, SCREEN_HEIGHT, "SPH Fluid Simulation with modernGL", None, None)

        if not window:
            glfw.terminate()
            raise Exception("GLFW window could not be initialised.")

        # Make the context current
        glfw.make_context_current(window)

        ctx = moderngl.create_context()
        
        ctx.gc_mode = 'context_gc'
        ctx.enable_only(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)

        vertex_shader= '''
        #version 330
        in vec2 in_pos;
        void main() {

            gl_Position = vec4(in_pos, 0.0, 1.0);
            gl_PointSize = 8; // Size of the point
        }
        '''

        fragment_shader='''
        #version 330
        out vec4 fragColor;
        void main() {
            fragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
        '''

        prog = ctx.program(vertex_shader = vertex_shader, fragment_shader = fragment_shader)


        n_particles = 1200
        sim = Simulator(n_particles, SCREEN_WIDTH, SCREEN_HEIGHT, 4, dt=1/50)
        sim_started = False
        positions = sim_to_ndc(sim.positions.copy())

        vbo = ctx.buffer(positions.tobytes())
        vao = ctx.vertex_array(prog, vbo, 'in_pos')

        while not glfw.window_should_close(window):
            glfw.poll_events()

            positions, _ = sim.step()

            NDC_positions = sim_to_ndc(positions)
            vbo.write(NDC_positions.tobytes())

            ctx.clear(0.1, 0.2, 0.3, 1.0)

            vao.render(mode=moderngl.POINTS)
            ctx.finish()

            glfw.swap_buffers(window)

    finally:
        prog.release()
        vbo.release()
        vao.release()
        ctx.release()
        glfw.terminate()




        
                    

if __name__ == "__main__":
    main()