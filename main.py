import moderngl_window as mglw
import moderngl
import numpy as np
import pynbody


def get_colormap_texture(name='twilight_shifted', num_points=1000):
    import matplotlib.cm as cm
    cmap = cm.get_cmap(name)
    rgba = cmap(np.linspace(0.001, 0.999, num_points)).astype(np.float32)
    return rgba

def load_shader(name):
    with open(name, 'r') as f:
        return f.read()

class Test(mglw.WindowConfig):
    gl_version = (4, 1)
    aspect_ratio = None
    title = "pynbody-vis"
    # clear_color = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.particleRenderer = self.ctx.program(
            vertex_shader=load_shader("sph_vertex_shader.glsl"),
            fragment_shader=load_shader("sph_fragment_shader.glsl"),
            geometry_shader=load_shader("sph_geometry_shader.glsl")
        )

        self.render_resolution = (500, 500)
        self.texture_size = (int(np.ceil(self.render_resolution[0]*1.5)), self.render_resolution[0])

        self.model = np.eye(4, dtype=np.float32)

        pos, smooth, mass = self.get_data()
        self.points = self.ctx.buffer(pos)
        self.smooth = self.ctx.buffer(smooth)
        self.mass = self.ctx.buffer(mass)

        self.render_texture = self.ctx.texture(self.texture_size, 4, dtype='f4')
        # self.accumulation_buffer = self.ctx.texture(self.)

        self.colormap_texture = self.ctx.texture((1000, 1), 4, get_colormap_texture(), dtype='f4')

        self.render_buffer = self.ctx.framebuffer(color_attachments=[self.render_texture])

        self.scale = 100.0
        self.downsample_factor = 1
        self.particleRenderer['scale'] = self.scale
        self.particleRenderer['outputResolution'] = self.render_resolution[0]
        self.particleRenderer['downsampleFactor'] = self.downsample_factor
        self.particleRenderer['smoothScale'] = 1.0
        #self.particleRenderer['randomNumbers'] = np.random.uniform(0, 1, 1000).astype(np.float32)


        self.vertex_array = self.ctx.vertex_array(
            self.particleRenderer, [(self.points, '3f', 'in_pos'),
                                    (self.smooth, '1f', 'in_smooth'),
                                    (self.mass, '1f', 'in_mass')])

        self.model_matr = np.eye(4)

        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)

        self.vmin_vmax_is_set = False
        self.invalidate()
        self.last_render = 0

        self.logMapper = self.ctx.program(vertex_shader=load_shader("colormap_vertex_shader.glsl"),
                                          fragment_shader=load_shader("colormap_fragment_shader.glsl"))

        self.accumulator = self.ctx.program(vertex_shader=load_shader("accumulator_vertex_shader.glsl"),
                                            fragment_shader=load_shader("accumulator_fragment_shader.glsl"))

        self.log_mapper_vertex_array = self.ctx.simple_vertex_array(
            self.logMapper,
            self.ctx.buffer(np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0],
                                      [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]], dtype=np.float32)),
            'in_vert'
        )

        def geometric_series(n):
            """Return sum_{m=1}^n 2**-m = 1-2**-n"""
            return 1.-2.**-n
        def corners_of_nth_framebuffer(n):
            return [2./3, geometric_series(n-1)], \
                   [2./3 + 2./3*(2**-n), geometric_series(n-1)], \
                   [2./3, geometric_series(n)], \
                   [2./3, geometric_series(n)], \
                   [2./3 + 2./3*(2**-n), geometric_series(n-1)], \
                   [2./3 + 2./3*(2**-n), geometric_series(n)]


        self.accumulator_vertex_array = self.ctx.vertex_array(
            self.accumulator,
            [
                (self.ctx.buffer(np.array([corners_of_nth_framebuffer(n) for n in range(1,6)], dtype=np.float32)),
                 '2f', 'from_position'),
                (self.ctx.buffer(np.array([[[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], \
                                            [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]] for n in range(1,6)], dtype=np.float32)),
                 '2f', 'to_position'),
            ]
        )

        self.logMapper['inputImage'] = 0
        self.logMapper['inputColorMap'] = 1
        self.render_texture.use(0)
        self.colormap_texture.use(1)

        self.setup_kernel_texture()


    def get_data(self):
        f = pynbody.load("/Users/app/Science/tangos/test_tutorial_build/tutorial_changa/pioneer50h128.1536gst1.bwK1.000960")
        f.physical_units()

        f_region = f.dm

        import pickle

        try:
            f_region['smooth'] = pickle.load(open('smooth.pkl', 'rb'))
            f_region['rho'] = pickle.load(open('rho.pkl', 'rb'))
        except:
            pickle.dump(f_region['smooth'], open('smooth.pkl', 'wb'))
            pickle.dump(f_region['rho'], open('rho.pkl', 'wb'))

        pynbody.analysis.halo.center(f.gas,cen_size="1 kpc")
        return f_region['pos'].astype(np.float32), \
            f_region['smooth'].astype(np.float32), \
            f_region['mass'].astype(np.float32)

    def setup_kernel_texture(self, n_samples=100):
        pynbody_sph_kernel = pynbody.sph.Kernel2D()
        x, y = np.meshgrid(np.linspace(-2, 2, n_samples), np.linspace(-2, 2, n_samples))
        distance = np.sqrt(x ** 2 + y ** 2)
        kernel_im = np.array([pynbody_sph_kernel.get_value(d) for d in distance.flatten()]).reshape(n_samples, n_samples)

        self.kernel_texture = self.ctx.texture((n_samples, n_samples), 1, kernel_im.astype(np.float32), dtype='f4')
        self.kernel_texture.use(2)
        self.particleRenderer['kernel'] = 2


    def render(self, time, frametime):



        self.model = self.model_matr.astype(np.float32)
        self.particleRenderer['model'].write(self.model)

        screenbuffer = self.ctx.fbo

        w, h = self.wnd.width, self.wnd.height

        if self.needs_render:
            query = self.ctx.query(time=True)
            with query:
                self.render_buffer.use()

                self.setup_multires_viewport()

                self.render_sph()
                screenbuffer.use()

            time_taken = float(query.elapsed)*1e-9

            #print(f"Render took {time_taken} seconds with downsampling factor {self.downsample_factor}")
            if time_taken>0.02:
                self.downsample_factor = int(np.ceil(self.downsample_factor*time_taken/0.02))
                self.particleRenderer['downsampleFactor'] = self.downsample_factor

            self.needs_render = False
            self.last_render = time

        self.display_render_buffer()

        if not self.vmin_vmax_is_set:
            self.set_vmin_vmax()

        if time-self.last_render>0.3 and self.allow_autorender:
            self.downsample_factor = 1
            self.particleRenderer['downsampleFactor'] = 1
            self.particleRenderer['smoothScale'] = 1.0
            self.needs_render = True
            self.allow_autorender = False




        #self.render_sph()

    def setup_multires_viewport(self):
        import OpenGL.GL
        OpenGL.GL.glViewportIndexedf(0, 0, 0, self.render_resolution[0], self.render_resolution[1])
        y_offset = 0
        for i in range(1, 5):
            res_factor = 2 ** i
            OpenGL.GL.glViewportIndexedf(i, self.render_resolution[0], y_offset,
                                         self.render_resolution[0] // res_factor,
                                         self.render_resolution[1] // res_factor)
            y_offset += self.render_resolution[1] // res_factor

    def perform_multires_accumulation(self):
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.accumulator_vertex_array.render(moderngl.TRIANGLES)

    def set_vmin_vmax(self):
        vals = np.log(self.get_sph_result()).ravel()
        vals = vals[np.isfinite(vals)]

        vmin, vmax = np.percentile(vals, [1.0,99.9])
        print(len(vals), vals.min(), vals.max(), vmin, vmax)
        self.logMapper['vmin'] = vmin
        self.logMapper['vmax'] = vmax
        self.save()
        self.vmin_vmax_is_set = True

    def display_render_buffer(self):
        self.ctx.clear(1.0, 0.0, 0.0, 1.0)
        # self.vertex_array.render(moderngl.POINTS)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.log_mapper_vertex_array.render(moderngl.TRIANGLES)

    def render_sph(self):
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vertex_array.render(moderngl.POINTS)
        self.perform_multires_accumulation()

    def get_sph_result(self):
        mybuffer = np.empty(self.texture_size, dtype=np.float32)
        self.render_buffer.read_into(mybuffer, components=1, dtype='f4')
        return mybuffer

    def save(self):
        mybuffer = self.get_sph_result()
        import pylab as p
        p.imshow(np.log(mybuffer))
        p.colorbar()
        p.savefig("output.png")
        #exit(0)


    def mouse_drag_event(self, x, y, dx, dy):
        dx_rotation_matrix = np.array([[np.cos(dx*0.01), 0, np.sin(dx*0.01), 0],
                                        [0, 1, 0, 0],
                                        [-np.sin(dx*0.01), 0, np.cos(dx*0.01), 0],
                                        [0, 0, 0, 1]])
        dy_rotation_matrix = np.array([[1, 0, 0, 0],
                                        [0, np.cos(dy*0.01), -np.sin(dy*0.01), 0],
                                        [0, np.sin(dy*0.01), np.cos(dy*0.01), 0],
                                        [0, 0, 0, 1]])
        self.model_matr = self.model_matr @ dx_rotation_matrix @ dy_rotation_matrix
        super().mouse_drag_event(x, y, dx, dy)
        self.invalidate()



    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        self.scale*=np.exp(y_offset*0.05)
        self.particleRenderer['scale'] = self.scale
        if abs(y_offset)>0.001:
            self.vmin_vmax_is_set = True

        self.invalidate()

    def invalidate(self):
        self.needs_render = True
        self.allow_autorender = True


if __name__=="__main__":
    Test.run()

