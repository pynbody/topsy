import numpy as np
import wgpu
import wgpu.backends.rs # noqa: F401, Select Rust backend

from . import config
from . import canvas
from . import colormap
from . import sph

class Visualizer:
    colorbar_label = r"$\mathrm{log}_{10}$ density / $M_{\odot} / \mathrm{kpc}^2$"
    colormap_name = config.DEFAULT_COLORMAP
    colorbar_aspect_ratio = config.COLORBAR_ASPECT_RATIO
    def __init__(self):
        self.canvas = canvas.VisualizerCanvas(visualizer=self, title="topsy")
        self.adapter: wgpu.GPUAdapter = wgpu.request_adapter(canvas=self.canvas, power_preference="high-performance")
        self.device: wgpu.GPUDevice = self.adapter.request_device()
        self.context: wgpu.GPUCanvasContext = self.canvas.get_context()

        self.canvas_format = self.context.get_preferred_format(self.adapter)
        if self.canvas_format.endswith("-srgb"):
            # matplotlib colours aren't srgb. It might be better to convert
            # but for now, just stop the canvas being srgb
            self.canvas_format = self.canvas_format[:-5]

        self.context.configure(device=self.device, format=self.canvas_format)

        self._render_resolution = 500
        self.render_texture: wgpu.GPUTexture = self.device.create_texture(
            size=(self._render_resolution, self._render_resolution, 1),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT |
                  wgpu.TextureUsage.TEXTURE_BINDING |
                  wgpu.TextureUsage.COPY_SRC,
            format=wgpu.TextureFormat.r32float,
            label="sph_render_texture",
        )


        self._colormap = colormap.Colormap(self)
        self._sph = sph.SPH(self)

        self.vmin_vmax_is_set = False

        self.invalidate()



    def invalidate(self):
        self.canvas.request_draw(self.draw)

    def rotate(self, dx, dy):
        dx_rotation_matrix = self._x_rotation_matrix(dx*0.01)
        dy_rotation_matrix = self._y_rotation_matrix(dy*0.01)
        self._sph.rotation_matrix = dx_rotation_matrix @ dy_rotation_matrix @ self._sph.rotation_matrix
        self.invalidate()

    @property
    def scale(self):
        return self._sph.scale
    @scale.setter
    def scale(self, value):
        self._sph.scale = value
        self.invalidate()

    @staticmethod
    def _y_rotation_matrix(angle):
        return np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])

    @staticmethod
    def _x_rotation_matrix(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])

    def draw(self):
        command_encoder = self.device.create_command_encoder()

        self._sph.encode_sph_render_pass(command_encoder)

        if not self.vmin_vmax_is_set:
            self.device.queue.submit([command_encoder.finish()]) # have to render the image to get the min/max
            self._colormap.set_vmin_vmax()
            command_encoder = self.device.create_command_encoder() # new command encoder needed
            self.vmin_vmax_is_set = True

        self._colormap.encode_render_pass(command_encoder)

        self.device.queue.submit([command_encoder.finish()])

    def get_rendered_image(self) -> np.ndarray:
        im = self.device.queue.read_texture({'texture':self.render_texture, 'origin':(0, 0, 0)},
                                            {'bytes_per_row':4*self._render_resolution},
                                            (self._render_resolution, self._render_resolution, 1))
        im = np.frombuffer(im, dtype=np.float32).reshape((self._render_resolution, self._render_resolution))
        return im

    def save(self):
        mybuffer = self.get_rendered_image()
        import pylab as p
        fig = p.figure()
        p.clf()
        p.set_cmap(self.colormap_name)
        extent = np.array([-1., 1., -1., 1.])*self.scale
        p.imshow(np.log10(mybuffer), vmin=self._colormap.vmin, vmax=self._colormap.vmax, extent=extent)
        p.xlabel("$x$/kpc")
        p.colorbar().set_label(self.colorbar_label)
        p.savefig("output.pdf")
        p.close(fig)

    def run(self):
        wgpu.gui.auto.run()
