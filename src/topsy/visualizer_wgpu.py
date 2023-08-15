import numpy as np
import pynbody
import wgpu
import wgpu.backends.rs # noqa: F401, Select Rust backend
import matplotlib

from . import config
from . import canvas

def load_shader(name):
    from importlib import resources
    with open(resources.files("topsy.shaders") / name, "r") as f:
        return f.read()

class Visualizer:
    colorbar_label = r"$\mathrm{log}_{10}$ density / $M_{\odot} / \mathrm{kpc}^2$"
    colormap_name = config.DEFAULT_COLORMAP
    colorbar_aspect_ratio = config.COLORBAR_ASPECT_RATIO
    def __init__(self):
        self.canvas = canvas.VisualizerCanvas(visualizer=self, title="topsy")
        self.adapter: wgpu.GPUAdapter = wgpu.request_adapter(canvas=self.canvas, power_preference="high-performance")
        self.device: wgpu.GPUDevice = self.adapter.request_device()
        self.context: wgpu.GPUCanvasContext = self.canvas.get_context()

        self._canvas_format = self.context.get_preferred_format(self.adapter)
        if self._canvas_format.endswith("-srgb"):
            # matplotlib colours aren't srgb. It might be better to convert
            # but for now, just stop the canvas being srgb
            self._canvas_format = self._canvas_format[:-5]

        self.context.configure(device=self.device, format=self._canvas_format)

        self._render_resolution = 500
        self._render_texture: wgpu.GPUTexture = self.device.create_texture(
            size=(self._render_resolution, self._render_resolution, 1),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT |
                  wgpu.TextureUsage.TEXTURE_BINDING |
                  wgpu.TextureUsage.COPY_SRC,
            format=wgpu.TextureFormat.r32float,
            label="sph_render_texture",
        )

        self._setup_sph_shader_module()
        self._setup_particle_buffer()
        self._setup_transform_matrix()
        self._setup_vertex_offsets()
        self._setup_kernel_texture()
        self._setup_sph_render_pipeline()

        self._setup_colormap_texture()
        self._setup_colormap_shader_module()
        self._setup_colormap_render_pipeline()

        self._scale = 1.0



        self.vmin_vmax_is_set = False

        self.invalidate()



    def invalidate(self):
        self.canvas.request_draw(self.draw)

    def _setup_colormap_shader_module(self):
        self._colormap_shader = self.device.create_shader_module(code=load_shader("colormap.wgsl"), label="colormap")
    def _setup_sph_shader_module(self):
        self._sph_shader = self.device.create_shader_module(code=load_shader("sph.wgsl"), label="sph")
    def _setup_particle_buffer(self):
        self._n_particles = int(5e6)
        data = np.zeros((self._n_particles, 4), dtype=np.float32) #np.random.normal(size=(self._n_particles, 4)).astype(np.float32)

        # xyz coordinates
        data[:,:3] = np.random.normal(size=(self._n_particles, 3),scale=0.2).astype(np.float32)

        data[:self._n_particles // 2, :3] = \
            np.random.normal(size=(self._n_particles // 2, 3), scale=0.4).astype(np.float32)*[1.0,0.05,1.0]

        data[:self._n_particles//4, :3] = \
            np.random.normal(size=(self._n_particles//4, 3), scale=0.1).astype(np.float32) \
            + [0.6,0.0,0.0]

        # kernel size
        data[:,3] = np.random.uniform(0.01,0.05,size=(self._n_particles,))

        self._particle_buffer = self.device.create_buffer_with_data(
            data = data,
            usage = wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.UNIFORM
        )

    def _setup_transform_matrix(self):
        self._transform: np.ndarray = np.eye(4, dtype=np.float32)
        self._transform_buffer = self.device.create_buffer(
            size = self._transform.nbytes + 16, # one extra float32 then padding
            usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

    def _setup_vertex_offsets(self):
        self._vertex_offsets_buffer = self.device.create_buffer_with_data(
            data = np.array([[0.0,0.0],[0.0,0.0],
                             # TODO - mystery why the above padding is needed
                             # The remainder is 2 triangles making up a square
                             [0.0,1.0],
                             [1.0,1.0],
                             [1.0,0.0],
                             [1.0,0.0],
                             [0.0,0.0],
                             [0.0,1.0]
                             ],dtype=np.float32),
            usage = wgpu.BufferUsage.VERTEX
        )

    def rotate(self, dx, dy):
        dx_rotation_matrix = self._x_rotation_matrix(dx)
        dy_rotation_matrix = self._y_rotation_matrix(dy)
        self._transform = dx_rotation_matrix @ dy_rotation_matrix @ self._transform
        self.invalidate()

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value
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

    def _setup_sph_render_pipeline(self):
        self._sph_bind_group_layout = \
            self.device.create_bind_group_layout(
                label="sph_bind_group_layout",
                entries=[
                    {
                        "binding": 0,
                        "visibility": wgpu.ShaderStage.VERTEX,
                        "buffer": {"type": wgpu.BufferBindingType.uniform},
                    },
                    {
                        "binding": 1,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "texture": {"sample_type": wgpu.TextureSampleType.float,
                                    "view_dimension": wgpu.TextureViewDimension.d2},
                    },
                    {
                        "binding": 2,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "sampler": {"type": wgpu.SamplerBindingType.filtering},
                    },
                ]
            )

        self._sph_bind_group = \
            self.device.create_bind_group(
                label="sph_bind_group",
                layout=self._sph_bind_group_layout,
                entries=[
                    {"binding": 0,
                     "resource": {
                         "buffer": self._transform_buffer,
                         "offset": 0,
                         "size": self._transform_buffer.size
                      }
                     },
                    {"binding": 1,
                        "resource": self._kernel_texture.create_view(),
                    },
                    {"binding": 2,
                        "resource": self._kernel_sampler,
                    }
                ]
            )

        self._sph_pipeline_layout = \
            self.device.create_pipeline_layout(
                label="sph_pipeline_layout",
                bind_group_layouts=[self._sph_bind_group_layout]
            )

        self._sph_render_pipeline = \
            self.device.create_render_pipeline(
                layout=self._sph_pipeline_layout,
                label="sph_render_pipeline",
                vertex = {
                    "module": self._sph_shader,
                    "entry_point": "vertex_main",
                    "buffers": [
                        {
                            "array_stride": 16,
                            "step_mode": wgpu.VertexStepMode.instance,
                            "attributes": [
                                {
                                    "format": wgpu.VertexFormat.float32x4,
                                    "offset": 0,
                                    "shader_location": 0,
                                }
                            ]
                        },
                        {
                            "array_stride": 8,
                            "step_mode": wgpu.VertexStepMode.vertex,
                            "attributes": [
                                {
                                    "format": wgpu.VertexFormat.float32x2,
                                    "offset": 16,
                                    "shader_location": 1,
                                }
                            ]
                        }
                    ]
                },
                primitive={
                    "topology": wgpu.PrimitiveTopology.triangle_list,
                },
                depth_stencil=None,
                multisample=None,
                fragment={
                    "module": self._sph_shader,
                    "entry_point": "fragment_main",
                    "targets": [
                        {
                            "format": wgpu.TextureFormat.r32float,
                            "blend": {
                                "color": {
                                    "src_factor": wgpu.BlendFactor.one,
                                    "dst_factor": wgpu.BlendFactor.one,
                                    "operation": wgpu.BlendOperation.add,
                                },
                                "alpha": {
                                    "src_factor": wgpu.BlendFactor.one,
                                    "dst_factor": wgpu.BlendFactor.one,
                                    "operation": wgpu.BlendOperation.add,
                                },
                              }
                        }
                    ]
                }
            )

    def _rgba_to_srgba(self, image_rgba):
        image_srgba = ((image_rgba+0.055)/1.055)**2.4
        image_srgba[image_rgba <= 0.04045] = image_rgba[image_rgba <= 0.04045] / 12.92

        return image_rgba

    def _setup_colormap_texture(self, num_points=config.COLORMAP_NUM_SAMPLES):
        cmap = matplotlib.colormaps[self.colormap_name]
        rgba = cmap(np.linspace(0.001, 0.999, num_points)).astype(np.float32)
        rgba = self._rgba_to_srgba(rgba)

        self._colormap_texture = self.device.create_texture(
            label="colormap_texture",
            size=(num_points, 1, 1),
            dimension=wgpu.TextureDimension.d1,
            format=wgpu.TextureFormat.rgba32float,
            mip_level_count=1,
            sample_count=1,
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING
        )

        self.device.queue.write_texture(
            {
                "texture": self._colormap_texture,
                "mip_level": 0,
                "origin": [0, 0, 0],
            },
            rgba.tobytes(),
            {
                "bytes_per_row": 4 * 4 * num_points,
                "offset": 0,
            },
            (num_points, 1, 1)
        )

    def _setup_colormap_render_pipeline(self):
        self._vmin_vmax_buffer = self.device.create_buffer(size = 4*2,
                                                           usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        self._colormap_bind_group_layout = \
            self.device.create_bind_group_layout(
                label="colormap_bind_group_layout",
                entries=[
                    {
                        "binding": 0,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "texture": {"sample_type": wgpu.TextureSampleType.float,
                                    "view_dimension": wgpu.TextureViewDimension.d2},
                    },
                    {
                        "binding": 1,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "sampler": {"type": wgpu.SamplerBindingType.filtering},
                    },
                    {
                        "binding": 2,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "texture": {"sample_type": wgpu.TextureSampleType.float,
                                    "view_dimension": wgpu.TextureViewDimension.d1},
                    },
                    {
                        "binding": 3,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "sampler": {"type": wgpu.SamplerBindingType.filtering},
                    },
                    {
                        "binding": 4,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "buffer": {"type": wgpu.BufferBindingType.uniform}
                    }
                ]
            )

        self._colormap_input_interpolation = self.device.create_sampler(label="colormap_sampler",
                                                                        mag_filter=wgpu.FilterMode.linear, )

        self._colormap_bind_group = \
            self.device.create_bind_group(
                label="colormap_bind_group",
                layout=self._colormap_bind_group_layout,
                entries=[
                    {"binding": 0,
                     "resource": self._render_texture.create_view(),
                     },
                    {"binding": 1,
                     "resource": self._colormap_input_interpolation,
                     },
                    {"binding": 2,
                     "resource": self._colormap_texture.create_view(),
                     },
                    {"binding": 3,
                     "resource": self._colormap_input_interpolation,
                     },
                    {"binding": 4,
                        "resource": {"buffer": self._vmin_vmax_buffer,
                                     "offset": 0,
                                     "size": self._vmin_vmax_buffer.size}
                     }
                ]
            )

        self._colormap_pipeline_layout = \
            self.device.create_pipeline_layout(
                label="colormap_pipeline_layout",
                bind_group_layouts=[self._colormap_bind_group_layout]
            )


        self._colormap_pipeline = \
            self.device.create_render_pipeline(
                layout=self._colormap_pipeline_layout,
                label="colormap_pipeline",
                vertex={
                    "module": self._colormap_shader,
                    "entry_point": "vertex_main",
                    "buffers": []
                },
                primitive={
                    "topology": wgpu.PrimitiveTopology.triangle_strip,
                },
                depth_stencil=None,
                multisample=None,
                fragment={
                    "module": self._colormap_shader,
                    "entry_point": "fragment_main",
                    "targets": [
                        {
                            "format": self._canvas_format,
                            "blend": {
                                "color": {
                                    "src_factor": wgpu.BlendFactor.one,
                                    "dst_factor": wgpu.BlendFactor.zero,
                                    "operation": wgpu.BlendOperation.add,
                                },
                                "alpha": {
                                    "src_factor": wgpu.BlendFactor.one,
                                    "dst_factor": wgpu.BlendFactor.zero,
                                    "operation": wgpu.BlendOperation.add,
                                },
                            }
                        }
                    ]
                }
            )

    def draw(self):
        self._update_transform_buffer()

        command_encoder = self.device.create_command_encoder()

        self._encode_sph_render_pass(command_encoder)

        if not self.vmin_vmax_is_set:
            self.device.queue.submit([command_encoder.finish()]) # have to render the image to get the min/max
            self.set_vmin_vmax()
            command_encoder = self.device.create_command_encoder() # new command encoder needed

        self._encode_colormap_render_pass(command_encoder)

        self.device.queue.submit([command_encoder.finish()])

    def _encode_colormap_render_pass(self, command_encoder):
        display_texture = self.context.get_current_texture()
        colormap_render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": display_texture,
                    "resolve_target": None,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                    "load_op": wgpu.LoadOp.load,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )
        colormap_render_pass.set_pipeline(self._colormap_pipeline)
        colormap_render_pass.set_bind_group(0, self._colormap_bind_group, [], 0, 99)
        colormap_render_pass.draw(5, 1, 0, 0)
        colormap_render_pass.end()

    def _encode_sph_render_pass(self, command_encoder):
        view: wgpu.GPUTextureView = self._render_texture.create_view()
        sph_render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": view,
                    "resolve_target": None,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )
        sph_render_pass.set_pipeline(self._sph_render_pipeline)
        sph_render_pass.set_vertex_buffer(0, self._particle_buffer)
        sph_render_pass.set_vertex_buffer(1, self._vertex_offsets_buffer)
        sph_render_pass.set_bind_group(0, self._sph_bind_group, [], 0, 99)
        sph_render_pass.draw(6, self._n_particles, 0, 0)
        sph_render_pass.end()

    def _update_transform_buffer(self):
        # self._transform is the transformation around the origin (fine for opengl)
        # but in webgpu, the clip space in the z direction is [0,1]
        # so we need a matrix that brings z=0. to z=0.5 and squishes the z direction
        # so that the clipping is the same in all dimensions
        displace = np.array([[1.0, 0, 0, 0.0],
                             [0, 1.0, 0, 0.0],
                             [0, 0, 0.5, 0.5],
                             [0, 0, 0.0, 1.0]])
        scaled_transform = self._transform / self._scale
        scaled_transform[3, 3] = 1.0  # leave w alone, thank you!
        scaled_displaced_transform = (displace @ scaled_transform).T
        transform_params_dtype = [("transform", np.float32, (4, 4)),
                                  ("scale_factor", np.float32, (1,))]
        transform_params = np.zeros((), dtype=transform_params_dtype)
        transform_params["transform"] = scaled_displaced_transform
        transform_params["scale_factor"] = 1. / self._scale
        self.device.queue.write_buffer(self._transform_buffer, 0, transform_params)

    def _update_vmin_vmax_buffer(self):
        vmin_vmax_dtype = [("vmin", np.float32, (1,)),
                           ("vmax", np.float32, (1,))]
        vmin_vmax = np.zeros((), dtype=vmin_vmax_dtype)
        vmin_vmax["vmin"] = self.vmin
        vmin_vmax["vmax"] = self.vmax
        self.device.queue.write_buffer(self._vmin_vmax_buffer, 0, vmin_vmax)

    def get_rendered_image(self) -> np.ndarray:
        im = self.device.queue.read_texture({'texture':self._render_texture, 'origin':(0,0,0)},
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
        p.imshow(np.log10(mybuffer), vmin=self.vmin, vmax=self.vmax, extent=extent)
        p.xlabel("$x$/kpc")
        p.colorbar().set_label(self.colorbar_label)
        p.savefig("output.pdf")
        p.close(fig)

    def set_vmin_vmax(self):
        """Set the vmin and vmax values for the colormap based on the most recent SPH render"""

        # This can and probably should be done on-GPU using a compute shader, but for now
        # we'll do it on the CPU
        vals = np.log10(self.get_rendered_image()).ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals)>200:
            self.vmin, self.vmax = np.percentile(vals, [1.0,99.9])
        else:
            logger.warning("Problem setting vmin/vmax, perhaps there are no particles or something is wrong with them?")
            logger.warning("Press 'r' in the window to try again")
            self.vmin, self.vmax = 0.0, 1.0

        self._update_vmin_vmax_buffer()
        #self.logMapper['vmin'] = self.vmin
        #self.logMapper['vmax'] = self.vmax
        #self.update_matplotlib_colorbar_texture()
        self.vmin_vmax_is_set = True

    def _setup_kernel_texture(self, n_samples=64):
        pynbody_sph_kernel = pynbody.sph.Kernel2D()
        x, y = np.meshgrid(np.linspace(-2, 2, n_samples), np.linspace(-2, 2, n_samples))
        distance = np.sqrt(x ** 2 + y ** 2)
        kernel_im = np.array([pynbody_sph_kernel.get_value(d) for d in distance.flatten()]).reshape(n_samples, n_samples)

        # make kernel explicitly mass conserving; naive pixelization makes it not automatically do this.
        # It should be normalized such that the integral over the kernel is 1/h^2. We have h=1 here, and the
        # full width is 4h, so the width of a pixel is dx=4/n_samples. So we need to multiply by dx^2=(n_samples/4)^2.
        # This results in a correction of a few percent, typically; not huge but not negligible either.
        #
        # (Obviously h!=1 generally, so the h^2 normalization occurs within the shader later.)
        kernel_im *= (n_samples/4)**2 / kernel_im.sum()

        self._kernel_texture = self.device.create_texture(
            label="kernel_texture",
            size=(n_samples, n_samples, 1),
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.r32float,
            mip_level_count=1,
            sample_count=1,
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
        )

        self.device.queue.write_texture(
            {
                "texture": self._kernel_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            kernel_im.astype(np.float32).tobytes(),
            {
                "offset": 0,
                "bytes_per_row": 4*n_samples,
            },
            (n_samples, n_samples, 1)
        )


        self._kernel_sampler = self.device.create_sampler(label="kernel_sampler",
                                                          mag_filter=wgpu.FilterMode.linear,)

    def run(self):
        #wgpu.print_report()

        wgpu.gui.auto.run()
