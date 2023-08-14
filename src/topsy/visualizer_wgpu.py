import numpy as np
import pynbody
import wgpu
import wgpu.backends.rs # noqa: F401, Select Rust backend

from wgpu.gui.auto import WgpuCanvas, run, call_later

def load_shader(name):
    from importlib import resources
    with open(resources.files("topsy.shaders") / name, "r") as f:
        return f.read()

class VisualizerCanvas(WgpuCanvas):
    def __init__(self, *args, **kwargs):
        self._visualizer : Visualizer = kwargs.pop("visualizer")
        super().__init__(*args, **kwargs)
        self._last_x = 0
        self._last_y = 0

    def handle_event(self, event):
        if event['event_type']=='pointer_move' and len(event['buttons'])>0:
            self.drag(event['x']-self._last_x, event['y']-self._last_y)

        if event['event_type']=='wheel':
            self.mouse_wheel(event['dx'], event['dy'])

        if 'x' in event and 'y' in event:
            self._last_x = event['x']
            self._last_y = event['y']

    def drag(self, dx, dy):
        self._visualizer.rotate(dx, dy)

    def mouse_wheel(self, delta_x, delta_y):
        self._visualizer.scale*=np.exp(delta_y/1000)
class Visualizer:
    def __init__(self):
        load_shader("sph.wgsl")
        self.canvas = VisualizerCanvas(visualizer=self, title="topsy")
        self.adapter : wgpu.GPUAdapter = wgpu.request_adapter(canvas=self.canvas, power_preference="high-performance")
        self.device : wgpu.GPUDevice = self.adapter.request_device()
        self.context = self.canvas.get_context()
        self.context.configure(device=self.device, format=self.context.get_preferred_format(self.adapter))

        self._setup_sph_shader_module()
        self._setup_particle_buffer()
        self._setup_transform_matrix()
        self._setup_vertex_offsets()
        self._setup_kernel_texture()

        self._setup_sph_render_layout()

        self._scale = 1.0

        self.canvas.request_draw(self.draw)

    def _setup_sph_shader_module(self):
        self._sph_shader = self.device.create_shader_module(code=load_shader("sph.wgsl"), label="sph")
    def _setup_particle_buffer(self):
        self._n_particles = int(1e6)
        data = np.zeros((self._n_particles, 4), dtype=np.float32) #np.random.normal(size=(self._n_particles, 4)).astype(np.float32)

        # xyz coordinates
        data[:,:3] = np.random.normal(size=(self._n_particles, 3),scale=0.2).astype(np.float32)

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
        self.canvas.request_draw(self.draw)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value
        self.canvas.request_draw(self.draw)
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

    def _setup_sph_render_layout(self):


        self._bind_group_layout = \
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

        self._bind_group = \
            self.device.create_bind_group(
                label="sph_bind_group",
                layout=self._bind_group_layout,
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
                bind_group_layouts=[self._bind_group_layout]
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
                            "format": self.context.get_preferred_format(self.adapter),
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

    def draw(self):
        texture = self.context.get_current_texture()
        command_encoder = self.device.create_command_encoder(label="sph_command_encoder")

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": texture,
                    "resolve_target": None,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )


        render_pass.set_pipeline(self._sph_render_pipeline)
        render_pass.set_vertex_buffer(0, self._particle_buffer)
        render_pass.set_vertex_buffer(1, self._vertex_offsets_buffer)
        render_pass.set_bind_group(0, self._bind_group, [], 0, 99)
        render_pass.draw(6, self._n_particles, 0, 0)
        render_pass.end()



        # self._transform is the transformation around the origin (fine for opengl)
        # but in webgpu, the clip space in the z direction is [0,1]
        # so we need a matrix that brings z=0. to z=0.5 and squishes the z direction
        # so that the clipping is the same in all dimensions

        displace = np.array([[1.0, 0, 0, 0.0],
                                [0, 1.0, 0, 0.0],
                                [0, 0, 0.5, 0.5],
                                [0, 0, 0.0, 1.0]])

        scaled_transform = self._transform/self._scale
        scaled_transform[3,3] = 1.0 # leave w alone, thank you!

        scaled_displaced_transform = (displace@scaled_transform).T

        transform_params_dtype = [("transform", np.float32, (4, 4)),
                                  ("scale_factor", np.float32, (1,))]

        transform_params = np.zeros((), dtype=transform_params_dtype)
        transform_params["transform"] = scaled_displaced_transform
        transform_params["scale_factor"] = 1./self._scale

        self.device.queue.write_buffer(self._transform_buffer, 0, transform_params)

        self.device.queue.submit([command_encoder.finish()])



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


        self._kernel_sampler = self.device.create_sampler(label="kernel_sampler")

    def run(self):
        #wgpu.print_report()

        wgpu.gui.auto.run()
