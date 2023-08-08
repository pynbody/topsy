import moderngl_window as mglw
import moderngl
import numpy as np
import pynbody
import matplotlib

import multiresolution_geometry

def get_colormap_texture(name='twilight_shifted', num_points=1000):
    cmap = matplotlib.colormaps[name]
    rgba = cmap(np.linspace(0.001, 0.999, num_points)).astype(np.float32)
    return rgba

def load_shader(name):
    with open(name, 'r') as f:
        return f.read()

class Test(mglw.WindowConfig):
    gl_version = (4, 1)
    aspect_ratio = None
    title = "pynbody-vis"

    colormap_name = 'twilight_shifted'
    colorbar_aspect_ratio = 0.15
    # clear_color = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.particleRenderer = self.ctx.program(
            vertex_shader=load_shader("sph_vertex_shader.glsl"),
            fragment_shader=load_shader("sph_fragment_shader.glsl"),
            geometry_shader=load_shader("sph_geometry_shader.glsl")
        )

        self.render_resolution = (500, 500)
        self._mrg = multiresolution_geometry.MultiresolutionGeometry(self.render_resolution[0])

        self.texture_size = (int(np.ceil(self.render_resolution[0]*1.5)), self.render_resolution[0])

        self.model = np.eye(4, dtype=np.float32)

        pos, smooth, mass = self.get_data()
        self.points = self.ctx.buffer(pos)
        self.smooth = self.ctx.buffer(smooth)
        self.mass = self.ctx.buffer(mass)

        self.render_texture = self.ctx.texture(self.texture_size, 4, dtype='f4')
        # self.accumulation_buffer = self.ctx.texture(self.)

        self.colormap_texture = self.ctx.texture((1000, 1), 4, get_colormap_texture(self.colormap_name), dtype='f4')

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

        self.accumulator = self.ctx.program(vertex_shader=load_shader("drawtexture_vertex_shader.glsl"),
                                            fragment_shader=load_shader("drawtexture_fragment_shader.glsl"))

        self.textureRender = self.ctx.program(vertex_shader=load_shader("drawtexture_vertex_shader.glsl"),
                                              fragment_shader=load_shader("drawtexture_fragment_shader.glsl"))

        self.scalebarRender = self.ctx.program(vertex_shader=load_shader("scalebar_vertex_shader.glsl"),
                                               fragment_shader=load_shader("scalebar_fragment_shader.glsl"),
                                               geometry_shader=load_shader("scalebar_geometry_shader.glsl"))



        self.logMapper['inputImage'] = 0
        self.logMapper['inputColorMap'] = 1

        self.setup_final_render_positions(*self.window_size)


        self.log_mapper_vertex_array = self.ctx.simple_vertex_array(
            self.logMapper,
            self.triangle_buffer(-1,-1,2,2),
            'in_vert'
        )

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
                    for n in range(1,6)], dtype=np.float32)),
                 '2f', 'from_position'),
                (self.ctx.buffer(np.array(
                    [self._mrg.corners_to_triangles(self._mrg.get_clipspace_corners()) for n in range(1,6)]
                    , dtype=np.float32)),
                 '2f', 'to_position'),
            ]
        )

        self.accumulator['inputImage'] = 0


        self.render_texture.use(0)
        self.colormap_texture.use(1)

        self.setup_kernel_texture()


    def setup_final_render_positions(self, width, height):
        maxdim = max(width, height)
        self.logMapper['texturePortion'] = [2. / 3 * width / maxdim, height / maxdim]
        self.logMapper['textureOffset'] = [1. / 3 * (1. - width / maxdim), (1. - height / maxdim) / 2]

        cb_width = 2. * self.colorbar_aspect_ratio * height / width
        self.colorbar_vertex_array = self.ctx.vertex_array(
            self.textureRender,
            [
                (self.triangle_buffer(0, 1, 1, -1), '2f', 'from_position'),
                (self.triangle_buffer(1.0 - cb_width, -1,
                                      cb_width, 2), '2f', 'to_position')
            ]
        )

    def triangle_buffer(self, x0, y0, w, h):
        return self.ctx.buffer(np.array([
            [x0, y0],
            [x0 + w, y0],
            [x0, y0 + h],
            [x0, y0 + h],
            [x0 + w, y0 + h],
            [x0 + w, y0]
        ], dtype=np.float32))
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

        # randomize order to avoid artifacts when downsampling number of particles on display
        random_order = np.random.permutation(len(f_region))

        return f_region['pos'].astype(np.float32)[random_order], \
            f_region['smooth'].astype(np.float32)[random_order], \
            f_region['mass'].astype(np.float32)[random_order]

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

        self.setup_final_render_positions(self.wnd.width, self.wnd.height)
        self.display_render_buffer()
        self.render_colorbar()
        self.render_scalebar()

        if not self.vmin_vmax_is_set:
            self.set_vmin_vmax()

        if time-self.last_render>0.3 and self.allow_autorender:
            self.downsample_factor = 1
            self.particleRenderer['downsampleFactor'] = 1
            self.particleRenderer['smoothScale'] = 1.0
            self.needs_render = True
            self.allow_autorender = False


    def setup_multires_viewport(self):
        import OpenGL.GL
        OpenGL.GL.glViewportIndexedf(0, 0, 0, self.render_resolution[0], self.render_resolution[1])

        for i in range(0, 5):
            vp_corners = self._mrg.get_viewport_corners_for_level(i)
            OpenGL.GL.glViewportIndexedf(i, vp_corners[0][0], vp_corners[0][1],
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
        vals = np.log(self.get_sph_result()).ravel()
        vals = vals[np.isfinite(vals)]

        self.vmin, self.vmax = np.percentile(vals, [1.0,99.9])
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

        cmap = matplotlib.colormaps[self.colormap_name]
        cNorm = colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        cb1 = matplotlib.colorbar.ColorbarBase(fig.add_axes([0.05, 0.05, 0.3, 0.9]),
                                                  cmap=cmap,norm=cNorm, orientation='vertical')
        cb1.set_label('Density')



        fig.canvas.draw()


        import PIL.Image
        img = PIL.Image.frombytes('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
        img.save("colorbar2.png")


        texture = self.ctx.texture(fig.canvas.get_width_height(), 4,
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
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vertex_array.render(moderngl.POINTS)
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
        p.imshow(np.log(mybuffer), vmin=self.vmin,vmax=self.vmax, extent=extent).set_label("log density")
        p.xlabel("$x$/kpc")
        p.colorbar().set_label("log density")
        p.savefig("output.png")
        p.close(fig)

        # now also save the framebuffer as a png using pillow
        import PIL.Image
        print(self.ctx.fbo.size)
        print(self.ctx.fbo)
        img = PIL.Image.frombytes('RGB', self.ctx.fbo.size,
                                  self.ctx.fbo.read(), 'raw', 'RGB', 0, -1)
        img.save("output2.png")


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
        self.update_scalebar_length()
        self.invalidate()

    def update_scalebar_length(self):
        # target is for the scalebar to be around 1/3rd of the viewport
        # however the length is to be 10^n or 5*10^n, so we need to find the
        # closest power of 10 to 1/3rd of the viewport

        # in world coordinates the viewport is self.scale kpc wide
        # so we need to find the closest power of 10 to self.scale/3

        physical_scalebar_length = self.scale/3.0
        # now quantize it:
        power_of_ten = np.floor(np.log10(physical_scalebar_length))
        mantissa = physical_scalebar_length/10**power_of_ten
        if mantissa<2.0:
            physical_scalebar_length = 10.0**power_of_ten
        elif mantissa<5.0:
            physical_scalebar_length = 2.0*10.0**power_of_ten
        else:
            physical_scalebar_length = 5.0*10.0**power_of_ten

        import text
        labelRgba = (text.text_to_rgba(f"{physical_scalebar_length:.0f} kpc", dpi=200, color='white')*255).astype(dtype=np.int8)

        print(labelRgba.shape)

        texture = self.ctx.texture(labelRgba.shape[1::-1], 4,
                                   labelRgba, dtype='f1')
        self.label_texture = texture

        aspect_ratio = labelRgba.shape[1]/labelRgba.shape[0]
        target_aspect_ratio = self.wnd.width/self.wnd.height

        self.scalebar_label_vertex_array = self.ctx.vertex_array(
            self.textureRender,
            [
                (self.triangle_buffer(0, 1, 1, -1), '2f', 'from_position'),
                (self.triangle_buffer(-0.9, -0.84, 0.05*aspect_ratio/target_aspect_ratio, 0.05), '2f', 'to_position')
            ]
        )

        self.scalebarRender['length'] = physical_scalebar_length/self.scale

    def key_event(self, key, action, modifiers):
        print("key_event",key,chr(key))
        if chr(key)=="r":
            self.set_vmin_vmax()
        if chr(key)=="s":
            self.save()

    def invalidate(self):
        self.needs_render = True
        self.allow_autorender = True




if __name__=="__main__":
    Test.run()

