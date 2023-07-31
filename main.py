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
    gl_version = (3, 3)
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

        self.render_resolution = (50, 50)

        self.model = np.eye(4, dtype=np.float32)

        pos, smooth, mass = self.get_data()
        self.points = self.ctx.buffer(pos)
        self.smooth = self.ctx.buffer(smooth)
        self.mass = self.ctx.buffer(mass)

        self.texture = self.ctx.texture(self.render_resolution, 4,  dtype='f4')

        self.colormap_texture = self.ctx.texture((1000, 1), 4, get_colormap_texture(), dtype='f4')

        print(np.frombuffer(self.colormap_texture.read(),dtype='f4').reshape(1000,4))

        self.render_buffer = self.ctx.framebuffer(color_attachments=[self.texture])


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

        self.log_mapper_vertex_array = self.ctx.simple_vertex_array(
            self.logMapper,
            self.ctx.buffer(np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0],
                                      [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]], dtype=np.float32)),
            'in_vert'
        )
        self.logMapper['inputImage'] = 0
        self.logMapper['inputColorMap'] = 1
        self.texture.use(0)
        self.colormap_texture.use(1)

        self.setup_kernel_texture()


    def get_data(self):
        f = pynbody.load("/Volumes/oaktree/EDGE/Halo339_DMO/output_00101")
        f.physical_units()
        import pickle

        try:
            f.dm['smooth'] = pickle.load(open('smooth.pkl', 'rb'))
            f.dm['rho'] = pickle.load(open('rho.pkl', 'rb'))
        except OSError:
            pickle.dump(f.dm['smooth'], open('smooth.pkl', 'wb'))
            pickle.dump(f.dm['rho'], open('rho.pkl', 'wb'))

        pynbody.analysis.halo.center(f.dm[f.dm['mass']<f.dm['mass'].min()*1.01])
        return f.dm['pos'].astype(np.float32), f.dm['smooth'].astype(np.float32), f.dm['mass'].astype(np.float32)

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



        if self.needs_render:
            query = self.ctx.query(time=True)
            with query:
                self.render_buffer.use()
                self.render_sph()
                screenbuffer.use()

            time_taken = float(query.elapsed)*1e-9

            print(f"Render took {time_taken} seconds with downsampling factor {self.downsample_factor}")
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

    def get_sph_result(self):
        mybuffer = np.empty(self.render_resolution, dtype=np.float32)
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

