class Colormap:
    def _setup_colormap_shader_module(self):
        self._colormap_shader = self.device.create_shader_module(code=load_shader("colormap.wgsl"), label="colormap")