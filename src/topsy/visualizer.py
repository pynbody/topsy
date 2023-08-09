import moderngl_window as mglw
import moderngl
import numpy as np
import pynbody
import matplotlib
import logging
import pickle
import OpenGL.GL

from . import config, multiresolution_geometry, scalebar

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_colormap_texture(name, context, num_points=config.COLORMAP_NUM_SAMPLES):
    cmap = matplotlib.colormaps[name]
    rgba = cmap(np.linspace(0.001, 0.999, num_points)).astype(np.float32)
    return context.texture((1000, 1), 4, rgba, dtype='f4')

def load_shader(name):
    from importlib import resources
    with open(resources.files("topsy.shaders") / name, "r") as f:
        return f.read()

class Visualizer(mglw.WindowConfig, scalebar.Scalebar):
    gl_version = (4, 1)
    aspect_ratio = None
    title = "pynbody visualizer"

    colorbar_label = r"$\mathrm{log}_{10}$ density / $M_{\odot} / \mathrm{kpc}^2$"
    colormap_name = config.DEFAULT_COLORMAP
    colorbar_aspect_ratio = config.COLORBAR_ASPECT_RATIO
    clear_color = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.particleRenderer = self.ctx.program(
            vertex_shader=load_shader("sph_vertex_shader.glsl"),
            fragment_shader=load_shader("sph_fragment_shader.glsl"),
            geometry_shader=load_shader("sph_geometry_shader.glsl")
        )

        self.render_resolution = (self.args['resolution'], self.args['resolution'])

        self._mrg = multiresolution_geometry.MultiresolutionGeometry(self.render_resolution[0])

        self.texture_size = (int(np.ceil(self.render_resolution[0]*1.5)), self.render_resolution[0])

        self.render_texture = self.ctx.texture(self.texture_size, 4, dtype='f4')

        self.colormap_texture = get_colormap_texture(self.args['colormap'], self.ctx)

        self.render_buffer = self.ctx.framebuffer(color_attachments=[self.render_texture])

        self._setup_sph_rendering()



        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)


        self.logMapper = self.ctx.program(vertex_shader=load_shader("colormap_vertex_shader.glsl"),
                                          fragment_shader=load_shader("colormap_fragment_shader.glsl"))

        self.accumulator = self.ctx.program(vertex_shader=load_shader("drawtexture_vertex_shader.glsl"),
                                            fragment_shader=load_shader("drawtexture_fragment_shader.glsl"))

        self.textureRender = self.ctx.program(vertex_shader=load_shader("drawtexture_vertex_shader.glsl"),
                                              fragment_shader=load_shader("drawtexture_fragment_shader.glsl"))

        self.scalebarRender = self.ctx.program(vertex_shader=load_shader("scalebar_vertex_shader.glsl"),
                                               fragment_shader=load_shader("scalebar_fragment_shader.glsl"),
                                               geometry_shader=load_shader("scalebar_geometry_shader.glsl"))



        self.logMapper['inputImage'] = 0
        self.logMapper['inputColorMap'] = 1




        self.log_mapper_vertex_array = self.ctx.simple_vertex_array(
            self.logMapper,
            self.triangle_buffer(-1,-1,2,2),
            'in_vert'
        )

        # the scalebar vertex array is just a single point; it will be turned into a line of the right length by the
        # geometry shader
        self.scalebar_vertex_array = self.ctx.vertex_array(
            self.scalebarRender,
            [
                (self.ctx.buffer(np.array([
                    -0.9,-0.9], dtype=np.float32)),
                    '2f', 'in_pos')
            ]
        )


        self.textureRender['inputImage'] = 3


        self.accumulator_vertex_array = self.ctx.vertex_array(
            self.accumulator,
            [
                (self.ctx.buffer(np.array([
                    self._mrg.corners_to_triangles(self._mrg.get_texture_corners_for_level(n))
                    for n in range(1,5)], dtype=np.float32)),
                 '2f', 'from_position'),
                (self.ctx.buffer(np.array(
                    [self._mrg.corners_to_triangles(self._mrg.get_clipspace_corners()) for n in range(1,5)]
                    , dtype=np.float32)),
                 '2f', 'to_position'),
            ]
        )

        self.accumulator['inputImage'] = 0


        self.render_texture.use(0)
        self.colormap_texture.use(1)

        self.setup_kernel_texture()
        self.update_scalebar_length()

    def _setup_sph_rendering(self):

        self.reset_view()

        self.downsample_factor = 1
        self.particleRenderer['scale'] = self.scale
        self.particleRenderer['outputResolution'] = self.render_resolution[0]
        self.particleRenderer['downsampleFactor'] = self.downsample_factor
        self.particleRenderer['smoothScale'] = 1.0
        self._load_data_into_buffers()
        self.vertex_array = self.ctx.vertex_array(
            self.particleRenderer, [(self.points, '3f', 'in_pos'),
                                    (self.smooth, '1f', 'in_smooth'),
                                    (self.mass, '1f', 'in_mass')])
        self.vmin_vmax_is_set = False
        self.last_render = 0
        self.invalidate()


    def reset_view(self):
        self.model_matr = np.eye(4)
        self.scale = 100.0
        self.invalidate()

    def setup_final_render_positions(self, width, height):
        maxdim = max(width, height)

        self.logMapper['texturePortion'] = [2. / 3 * width / maxdim, height / maxdim]
        self.logMapper['textureOffset'] = [1. / 3 * (1. - width / maxdim), (1. - height / maxdim) / 2]

        # to see whole result, including mipmap
        # self.logMapper['texturePortion'] = [1.0, 1.0]
        # self.logMapper['textureOffset'] = [0.0, 0.0]

        cb_width = 2. * self.colorbar_aspect_ratio * height / width
        self.colorbar_vertex_array = self.ctx.vertex_array(
            self.textureRender,
            [
                (self.triangle_buffer(0, 1, 1, -1), '2f', 'from_position'),
                (self.triangle_buffer(1.0 - cb_width, -1,
                                      cb_width, 2), '2f', 'to_position')
            ]
        )

        self._update_scalebar_label_vertex_array()

    def triangle_buffer(self, x0, y0, w, h):
        return self.ctx.buffer(np.array([
            [x0, y0],
            [x0 + w, y0],
            [x0, y0 + h],
            [x0, y0 + h],
            [x0 + w, y0 + h],
            [x0 + w, y0]
        ], dtype=np.float32))
    def _load_data(self):
        f = pynbody.load(self.args['filename'])
        f.physical_units()


        logger.info("Performing centering...")
        if self.args['center'].startswith("halo-"):
            halo_number = int(self.args['center'][5:])
            h = f.halos()
            pynbody.analysis.halo.center(h[halo_number])
            f = f[pynbody.family.get_family(self.args['particle'])]
        elif self.args['center']=='zoom':
            f = f[pynbody.family.get_family(self.args['particle'])]
            pynbody.analysis.halo.center(f[f['mass']<1.01*f['mass'].min()])
        elif self.args['center']=='all':
            pynbody.analysis.halo.center(f)
            f = f[pynbody.family.get_family(self.args['particle'])]
        elif self.args['center']=='none':
            f = f[pynbody.family.get_family(self.args['particle'])]
        else:
            raise ValueError("Unknown centering type")

        try:
            logger.info("Looking for cached smoothing/density data...")
            smooth = pickle.load(open('topsy-smooth.pkl', 'rb'))
            if len(smooth)==len(f):
                f['smooth'] = smooth
            else:
                raise ValueError("Incorrect number of particles in cached smoothing data")
            logger.info("...success!")

            rho = pickle.load(open('topsy-rho.pkl', 'rb'))
            if len(rho)==len(f):
                f['rho'] = rho
            else:
                raise ValueError("Incorrect number of particles in cached density data")
        except:
            logger.info("Generating smoothing/density data - this can take a while but will be cached for future runs")
            pickle.dump(f['smooth'], open('topsy-smooth.pkl', 'wb'))
            pickle.dump(f['rho'], open('topsy-rho.pkl', 'wb'))




        # randomize order to avoid artifacts when downsampling number of particles on display
        random_order = np.random.permutation(len(f))

        return f['pos'].astype(np.float32)[random_order], \
            f['smooth'].astype(np.float32)[random_order], \
            f['mass'].astype(np.float32)[random_order]

    def _load_data_into_buffers(self):
        pos, smooth, mass = self._load_data()
        self.points = self.ctx.buffer(pos)
        self.smooth = self.ctx.buffer(smooth)
        self.mass = self.ctx.buffer(mass)

    def setup_kernel_texture(self, n_samples=128):
        pynbody_sph_kernel = pynbody.sph.Kernel2D()
        x, y = np.meshgrid(np.linspace(-2, 2, n_samples), np.linspace(-2, 2, n_samples))
        distance = np.sqrt(x ** 2 + y ** 2)
        kernel_im = np.array([pynbody_sph_kernel.get_value(d) for d in distance.flatten()]).reshape(n_samples, n_samples)

        self.kernel_texture = self.ctx.texture((n_samples, n_samples), 1, kernel_im.astype(np.float32), dtype='f4')
        self.kernel_texture.use(2)
        self.kernel_texture.build_mipmaps()

        self.particleRenderer['kernel'] = 2


    def render(self, time_now, frametime):


        self.particleRenderer['model'].write(self.model_matr.astype(np.float32))

        screenbuffer = self.ctx.fbo


        if self.needs_render:
            query = self.ctx.query(time=True)
            with query:
                self.render_buffer.use()
                self.render_sph()
                screenbuffer.use()

            time_taken = float(query.elapsed)*1e-9

            if time_taken>0.02 and self.downsample_factor==1:
                self.downsample_factor = int(np.ceil(time_taken/0.02))
                self.particleRenderer['downsampleFactor'] = self.downsample_factor
                logger.info(f"Full res render took {time_taken} seconds; setting interaction downsampling factor to {self.downsample_factor}")

            self.needs_render = False
            self.last_render = time_now


        else:
            pass
            #import time
            #time.sleep(config.INACTIVITY_WAIT)

        self.setup_final_render_positions(self.wnd.width, self.wnd.height)
        self.display_render_buffer()
        self.render_colorbar()
        self.render_scalebar()

        if not self.vmin_vmax_is_set:
            self.set_vmin_vmax()

        if time_now-self.last_render>config.FULL_RESOLUTION_RENDER_AFTER and self.allow_autorender:
            self.downsample_factor = 1
            self.particleRenderer['downsampleFactor'] = 1
            self.particleRenderer['smoothScale'] = 1.0
            self.needs_render = True
            self.allow_autorender = False


    def setup_multires_viewport(self):
        OpenGL.GL.glViewportIndexedf(0, 0, 0, self.render_resolution[0], self.render_resolution[1])

        for i in range(0, 5):
            vp_corners = self._mrg.get_viewport_corners_for_level(i)
            OpenGL.GL.glViewportIndexedf(i, vp_corners[0][0], vp_corners[0][1],
                                         vp_corners[2][0] - vp_corners[0][0],
                                         vp_corners[2][1] - vp_corners[0][1])
            OpenGL.GL.glScissorIndexed(i, vp_corners[0][0], vp_corners[0][1],
                                         vp_corners[2][0] - vp_corners[0][0],
                                         vp_corners[2][1] - vp_corners[0][1])




    def perform_multires_accumulation(self):
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.accumulator_vertex_array.render(moderngl.TRIANGLES)

    def render_colorbar(self):
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        if hasattr(self, 'colorbar_texture'):
            self.colorbar_texture.use(3)
            self.colorbar_vertex_array.render(moderngl.TRIANGLES)

    def render_scalebar(self):
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.scalebar_vertex_array.render(moderngl.POINTS)
        if hasattr(self, 'label_texture'):
            self.label_texture.use(3)
            self.scalebar_label_vertex_array.render(moderngl.TRIANGLES)

    def set_vmin_vmax(self):
        vals = np.log10(self.get_sph_result()).ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals)>200:
            self.vmin, self.vmax = np.percentile(vals, [1.0,99.9])
        else:
            logger.warning("Problem setting vmin/vmax, perhaps there are no particles or something is wrong with them?")
            logger.warning("Press 'r' in the window to try again")
            self.vmin, self.vmax = 0.0, 1.0
        self.logMapper['vmin'] = self.vmin
        self.logMapper['vmax'] = self.vmax
        self.update_matplotlib_colorbar_texture()
        self.vmin_vmax_is_set = True




    def update_matplotlib_colorbar_texture(self):
        """Use matplotlib to get a colorbar, including labels based on vmin/vmax, and return it as a texture"""
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.cm as cmx

        fig = plt.figure(figsize=(10*self.colorbar_aspect_ratio, 10), dpi=200,
                         facecolor=(1.0,1.0,1.0,0.5))

        cmap = matplotlib.colormaps[self.args['colormap']]
        cNorm = colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        cb1 = matplotlib.colorbar.ColorbarBase(fig.add_axes([0.05, 0.05, 0.3, 0.9]),
                                                  cmap=cmap,norm=cNorm, orientation='vertical')
        cb1.set_label(self.colorbar_label)



        fig.canvas.draw()


        import PIL.Image
        img = PIL.Image.frombytes('RGBA', fig.canvas.get_width_height(physical=True), fig.canvas.buffer_rgba())
        img.save("colorbar2.png")


        texture = self.ctx.texture(fig.canvas.get_width_height(physical=True), 4,
                                   fig.canvas.buffer_rgba(), dtype='f1')
        self.colorbar_texture = texture
        self.colorbar_texture.use(3)
        self.colorbar_texture.filter = moderngl.LINEAR, moderngl.LINEAR

        plt.savefig("colorbar.png")
        plt.close(fig)


    def display_render_buffer(self):
        self.ctx.clear(1.0, 0.0, 0.0, 1.0)
        # self.vertex_array.render(moderngl.POINTS)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.log_mapper_vertex_array.render(moderngl.TRIANGLES)

    def render_sph(self):
        self.setup_multires_viewport()
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        OpenGL.GL.glDisable(OpenGL.GL.GL_SCISSOR_TEST)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        OpenGL.GL.glEnable(OpenGL.GL.GL_SCISSOR_TEST)
        self.vertex_array.render(moderngl.POINTS)
        OpenGL.GL.glFinish() # seems needed e.g. on radeon pro vega, otherwise accumulation composites garbage
        self.perform_multires_accumulation()

    def get_sph_result(self):
        mybuffer = np.empty(self.render_resolution, dtype=np.float32)
        self.render_buffer.read_into(mybuffer, viewport=self.render_resolution,
                                     components=1, dtype='f4')
        return mybuffer[::-1]

    def save(self):
        mybuffer = self.get_sph_result()
        import pylab as p
        fig = p.figure()
        p.clf()
        p.set_cmap(self.colormap_name)
        extent = np.array([-1., 1., -1., 1.])*self.scale
        p.imshow(np.log10(mybuffer), vmin=self.vmin,vmax=self.vmax, extent=extent)
        p.xlabel("$x$/kpc")
        p.colorbar().set_label(self.colorbar_label)
        p.savefig("output.pdf")
        p.close(fig)

        # alternatively, save the framebuffer directly:
        # import PIL.Image
        # img = PIL.Image.frombytes('RGB', self.ctx.fbo.size,
        #                           self.ctx.fbo.read(), 'raw', 'RGB', 0, -1)
        # img.save("output2.png")


    def mouse_drag_event(self, x, y, dx, dy):
        dx_rotation_matrix = np.array(self._x_rotation_matrix(dx))
        dy_rotation_matrix = np.array(self._y_rotation_matrix(dy))
        self.model_matr = self.model_matr @ dx_rotation_matrix @ dy_rotation_matrix
        super().mouse_drag_event(x, y, dx, dy)
        self.invalidate()

    def _y_rotation_matrix(self, angle):
        return np.array([[1, 0, 0, 0],
                         [0, np.cos(angle * 0.01), -np.sin(angle * 0.01), 0],
                         [0, np.sin(angle * 0.01), np.cos(angle * 0.01), 0],
                         [0, 0, 0, 1]])

    def _x_rotation_matrix(self, angle):
        return np.array([[np.cos(angle * 0.01), 0, np.sin(angle * 0.01), 0],
                         [0, 1, 0, 0],
                         [-np.sin(angle * 0.01), 0, np.cos(angle * 0.01), 0],
                         [0, 0, 0, 1]])

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        self.scale*=np.exp(y_offset*0.05)
        self.particleRenderer['scale'] = self.scale
        self.update_scalebar_length()
        self.invalidate()

    def key_event(self, key, action, modifiers):
        if chr(key)=="r":
            self.set_vmin_vmax()
        if chr(key)=="s":
            self.save()
        if chr(key)=="h":
            self.reset_view()

    def invalidate(self):
        self.needs_render = True
        self.allow_autorender = True




