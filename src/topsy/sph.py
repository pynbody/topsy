import numpy as np
import wgpu
import pynbody

from .util import load_shader

class SPH:
    def __init__(self, visualizer):
        self._visualizer = visualizer
        self._device = visualizer.device

        self._setup_sph_shader_module()
        self._setup_particle_buffer()
        self._setup_transform_buffer()
        self._setup_kernel_texture()
        self._setup_sph_render_pipeline()

        self.scale = 1.0
        self.rotation_matrix = np.eye(3)


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

        self._particle_buffer = self._device.create_buffer_with_data(
            data = data,
            usage = wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.UNIFORM)
    def _setup_sph_shader_module(self):
        self._sph_shader = self._device.create_shader_module(code=load_shader("sph.wgsl"), label="sph")

    def _setup_transform_buffer(self):
        self._transform_buffer = self._device.create_buffer(
            size=(4*4*4) + 4 + 8 + 4,  # 4x4 float32 matrix + one float32 scale + float32 min,max size + padding
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

    def _setup_sph_render_pipeline(self):
        self._sph_bind_group_layout = \
            self._device.create_bind_group_layout(
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
            self._device.create_bind_group(
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
            self._device.create_pipeline_layout(
                label="sph_pipeline_layout",
                bind_group_layouts=[self._sph_bind_group_layout]
            )

        self._sph_render_pipeline = \
            self._device.create_render_pipeline(
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

    def _update_transform_buffer(self):
        # self._transform is the transformation around the origin (fine for opengl)
        # but in webgpu, the clip space in the z direction is [0,1]
        # so we need a matrix that brings z=0. to z=0.5 and squishes the z direction
        # so that the clipping is the same in all dimensions
        displace = np.array([[1.0, 0, 0, 0.0],
                             [0, 1.0, 0, 0.0],
                             [0, 0, 0.5, 0.5],
                             [0, 0, 0.0, 1.0]])
        transform = np.zeros((4, 4))

        transform[:3,:3] = self.rotation_matrix
        scaled_transform = transform / self.scale

        scaled_transform[3, 3] = 1.0  # w should be unchanged after transform

        scaled_displaced_transform = (displace @ scaled_transform).T
        transform_params_dtype = [("transform", np.float32, (4, 4)),
                                  ("scale_factor", np.float32, (1,)),
                                  ("min_max_size", np.float32, (2,))]
        transform_params = np.zeros((), dtype=transform_params_dtype)
        transform_params["transform"] = scaled_displaced_transform
        transform_params["scale_factor"] = 1. / self.scale
        transform_params["min_max_size"] = (0.0, 1000.0) # no restriction, for now

        self._device.queue.write_buffer(self._transform_buffer, 0, transform_params)

    def encode_sph_render_pass(self, command_encoder):
        self._update_transform_buffer()
        view: wgpu.GPUTextureView = self._visualizer.render_texture.create_view()
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
        sph_render_pass.set_bind_group(0, self._sph_bind_group, [], 0, 99)
        sph_render_pass.draw(6, self._n_particles, 0, 0)
        sph_render_pass.end()


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

        self._kernel_texture = self._device.create_texture(
            label="kernel_texture",
            size=(n_samples, n_samples, 1),
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.r32float,
            mip_level_count=1,
            sample_count=1,
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
        )

        self._device.queue.write_texture(
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


        self._kernel_sampler = self._device.create_sampler(label="kernel_sampler",
                                                           mag_filter=wgpu.FilterMode.linear, )