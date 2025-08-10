import numpy as np
import wgpu
from .implementation import Colormap
from .. import config
from ..util import load_shader, preprocess_shader

class ColorAsSurfaceMap(Colormap):
    """A colormap that renders surfaces with lighting instead of colormaps"""
    
    fragment_shader = "fs_main"
    
    _default_params = {
        'depth_scale': 1.0,
        'light_direction': [0.0, 1.0/np.sqrt(2.), 1.0/np.sqrt(2.)],
        'light_color': [1.0, 1.0, 1.0],
        'ambient_color': [0.0, 0.0, 0.2],
        'log_den_threshold': None,
        'smoothing_scale': 0.01,
        'weighted_average': False,
        'vmin': 0.0,
        'vmax': 1.0,
        'log': False,
        'colormap_name': config.DEFAULT_COLORMAP,
    }
    
    shader_parameter_dtype = np.dtype([
        ("depthScale", np.float32, (1,)),
        ("_pad0", np.float32, (3,)),        # Padding to align next vec3
        ("lightDirection", np.float32, (3,)),
        ("_pad1", np.float32, (1,)),        # Padding to align next vec3
        ("lightColor", np.float32, (3,)),
        ("_pad2", np.float32, (1,)),        # Padding to align next vec3
        ("ambientColor", np.float32, (3,)),
        ("_pad3", np.float32, (1,)),        # Padding to align next vec2
        ("texelSize", np.float32, (2,)),
        ("windowAspectRatio", np.float32, (1,)),
        ("vmin", np.float32, (1,)),
        ("vmax", np.float32, (1,)),
        ("_pad4", np.float32, (3,))
    ])
    
    smooth_parameter_dtype = np.dtype([
        ("spatial_sigma", np.float32, (1,)),
        ("range_sigma", np.float32, (1,)),
        ("kernel_size", np.int32, (1,)),
        ("padding", np.int32, (1,))         # For 16-byte alignment
    ])

    @classmethod
    def accepts_parameters(cls, parameters: dict) -> bool:
        return parameters.get("type", None) == "surface"
    
    def _setup_compute_shader(self):
        # Load and create compute shader for bilateral filtering
        compute_shader_code = load_shader("smooth.wgsl")
        self._compute_shader = self._device.create_shader_module(code=compute_shader_code, label="bilateral_filter")
        
        # Create bilateral filter parameter buffer
        self._bilateral_parameter_buffer = self._device.create_buffer(
            size=self.smooth_parameter_dtype.itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        
        # Create compute bind group layout
        self._compute_bind_group_layout = self._device.create_bind_group_layout(
            label="bilateral_filter_bind_group_layout",
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "storage_texture": {
                        "access": wgpu.StorageTextureAccess.write_only,
                        "format": wgpu.TextureFormat.rg32float,
                        "view_dimension": wgpu.TextureViewDimension.d2
                    },
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.uniform}
                }
            ]
        )

        self._compute_pipeline_layout = self._device.create_pipeline_layout(
            label="bilateral_filter_pipeline_layout",
            bind_group_layouts=[self._compute_bind_group_layout]
        )
        
        self._compute_pipeline = self._device.create_compute_pipeline(
            layout=self._compute_pipeline_layout,
            label="bilateral_filter_pipeline",
            compute={
                "module": self._compute_shader,
                "entry_point": "bilateral_filter_main"
            }
        )
    
    def _setup_shader_module(self, active_flags=None):
        if active_flags is None:
            active_flags = []
        if self.get_parameter('weighted_average'):
            active_flags.append("MATERIAL_COLORMAP")
            if self.get_parameter('log'):
                active_flags.append("MATERIAL_LOG")
        shader_code = preprocess_shader(load_shader("surface.wgsl"), active_flags=active_flags)
        self._shader = self._device.create_shader_module(code=shader_code, label="surface")
    
    def _setup_render_pipeline(self):
        # Create smoothed depth texture for bilateral filter output
        self._smoothed_texture = self._device.create_texture(
            size=self._input_texture.size,
            format=wgpu.TextureFormat.rg32float,
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
            label="smoothed_depth_texture"
        )
        
        # Create compute bind group for bilateral filter
        self._setup_compute_shader()
        self._smooth_compute_bind_group = self._device.create_bind_group(
            label="bilateral_filter_bind_group",
            layout=self._compute_bind_group_layout,
            entries=[
                {"binding": 0, "resource": self._input_texture.create_view()},
                {"binding": 1, "resource": self._smoothed_texture.create_view()},
                {"binding": 2, "resource": {"buffer": self._bilateral_parameter_buffer, "offset": 0, "size": self._bilateral_parameter_buffer.size}}
            ]
        )
        
        self._parameter_buffer = self._device.create_buffer(
            size=self.shader_parameter_dtype.itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        
        self._bind_group_layout = self._device.create_bind_group_layout(
            label="surface_bind_group_layout",
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.uniform}
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d1
                    }
                }
            ]
        )
        
        self._input_interpolation = self._device.create_sampler(
            label="surface_sampler",
            mag_filter=wgpu.FilterMode.linear,
        )

        self._surface_render_bind_group = self._create_bind_group(self._smoothed_texture)
        
        self._pipeline_layout = self._device.create_pipeline_layout(
            label="surface_pipeline_layout",
            bind_group_layouts=[self._bind_group_layout]
        )
        
        self._pipeline = self._device.create_render_pipeline(
            layout=self._pipeline_layout,
            label="surface_pipeline",
            vertex={
                "module": self._shader,
                "entry_point": "vertex_main",
                "buffers": []
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_strip,
            },
            depth_stencil=None,
            multisample=None,
            fragment={
                "module": self._shader,
                "entry_point": self.fragment_shader,
                "targets": [
                    {
                        "format": self._output_format,
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
    
    def _create_bind_group(self, input_texture: wgpu.GPUTexture):
        return self._device.create_bind_group(
            label="surface_bind_group",
            layout=self._bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": input_texture.create_view(),
                },
                {
                    "binding": 1,
                    "resource": self._input_interpolation,
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self._parameter_buffer,
                        "offset": 0,
                        "size": self._parameter_buffer.size
                    }
                },
                {
                    "binding": 3,
                    "resource": self._texture.create_view(),
                }
            ]
        )

    def autorange_vmin_vmax(self, vals):
        valid = vals[..., 1].ravel()>0.0
        vals = vals[..., 0].ravel()
        self._autorange_using_values(vals[valid])

    def encode_render_pass(self, command_encoder, target_texture_view, bind_group=None):
        self._encode_smoothing_filter_pass(command_encoder)
        super().encode_render_pass(command_encoder, target_texture_view, self._surface_render_bind_group)
    
    def _encode_smoothing_filter_pass(self, command_encoder, compute_bind_group=None):
        if compute_bind_group is None:
            compute_bind_group = self._smooth_compute_bind_group
        bilateral_params = np.zeros((), dtype=self.smooth_parameter_dtype)
        sig = self._params.get('smoothing_scale', 0.01)
        if sig < 1e-5:
            sig = 1e-5
        bilateral_params["spatial_sigma"] = sig * self._input_texture.size[0]
        bilateral_params["range_sigma"] = sig * 2

        n_pix = int(bilateral_params["spatial_sigma"] * 4) + 1
        if n_pix > config.MAX_SURFACE_SMOOTH_PIXELS:
            n_pix = config.MAX_SURFACE_SMOOTH_PIXELS

        bilateral_params["kernel_size"] = n_pix

        self._device.queue.write_buffer(self._bilateral_parameter_buffer, 0, bilateral_params)

        compute_pass = command_encoder.begin_compute_pass(label="bilateral_filter_pass")
        compute_pass.set_pipeline(self._compute_pipeline)
        compute_pass.set_bind_group(0, compute_bind_group, [], 0, 99)

        width = self._input_texture.size[0]
        height = self._input_texture.size[1]
        workgroup_size_x = (width + 7) // 8  # 8x8 workgroup size
        workgroup_size_y = (height + 7) // 8
        
        compute_pass.dispatch_workgroups(workgroup_size_x, workgroup_size_y, 1)
        compute_pass.end()
    
    def _smooth_numpy(self, input_array: np.ndarray) -> np.ndarray:
        """
        Smooth a 2D numpy array (with 2 channels) using the bilateral filter compute shader.
        Returns the smoothed array as a numpy array of shape (height, width, 2).
        """
        # Ensure input is float32 and 3D (height, width, 2)
        arr = np.asarray(input_array, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 2:
            raise ValueError("Input array must be 3D with shape (height, width, 2)")
        height, width, channels = arr.shape

        assert channels == 2, "Input array must have exactly 2 channels"

        # Create input texture (rg32float)
        input_texture = self._device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rg32float,
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
            label="smooth_numpy_input"
        )

        self._device.queue.write_texture(
            {"texture": input_texture},
            arr.tobytes(),
            {"bytes_per_row": width * 8, "rows_per_image": height},
            (width, height, 1)
        )

        output_texture = self._device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rg32float,
            usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.STORAGE_BINDING,
            label="smooth_numpy_output"
        )

        compute_bind_group = self._device.create_bind_group(
            label="smooth_numpy_bind_group",
            layout=self._compute_bind_group_layout,
            entries=[
                {"binding": 0, "resource": input_texture.create_view()},
                {"binding": 1, "resource": output_texture.create_view()},
                {"binding": 2, "resource": {"buffer": self._bilateral_parameter_buffer, 
                                            "offset": 0, "size": self._bilateral_parameter_buffer.size}}
            ]
        )

        command_encoder = self._device.create_command_encoder(label="smooth_numpy_encoder")
        self._encode_smoothing_filter_pass(command_encoder, compute_bind_group)
        self._device.queue.submit([command_encoder.finish()])

        result = self._device.queue.read_texture(
            {
            "texture": output_texture,
            "mip_level": 0,
            "origin": (0, 0, 0)
            },
            {
            "bytes_per_row": width * 8,
            "rows_per_image": height
            },
            (width, height, 1),
        )
        result = np.frombuffer(result, dtype=np.float32).reshape((height, width, 2))
        return result

    def sph_raw_output_to_content(self, numpy_image: np.ndarray):
        return self._smooth_numpy(numpy_image)
    
    def _update_parameter_buffer(self, width, height, mass_scale):
        parameters = np.zeros((), dtype=self.shader_parameter_dtype)
        
        parameters["depthScale"] = self._params.get('depth_scale', 1.0)
        parameters["lightDirection"] = self._params.get('light_direction', [0.0, 0.0, 1.0])
        parameters["lightColor"] = self._params.get('light_color', [1.0, 1.0, 1.0])
        parameters["ambientColor"] = self._params.get('ambient_color', [0.2, 0.2, 0.2])
        parameters["texelSize"] = [1.0 / width, 1.0 / height]
        parameters["windowAspectRatio"] = float(width) / height
        parameters["vmin"] = self.get_parameter("vmin")
        parameters["vmax"] = self.get_parameter("vmax")

        self._device.queue.write_buffer(self._parameter_buffer, 0, parameters)
