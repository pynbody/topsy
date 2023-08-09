import numpy as np

from . import text

class Scalebar:
    def update_scalebar_length(self):
        physical_scalebar_length = self._recommend_physical_scalebar_length()
        self.scalebarRender['length'] = physical_scalebar_length / self.scale
        self._update_scalebar_label(physical_scalebar_length)

    def _update_scalebar_label_vertex_array(self):
        target_aspect_ratio = self.wnd.width / self.wnd.height
        self.scalebar_label_vertex_array = self.ctx.vertex_array(
            self.textureRender,
            [
                (self.triangle_buffer(0, 1, 1, -1), '2f', 'from_position'),
                (self.triangle_buffer(-0.9, -0.84, 0.05 * self.label_aspect_ratio / target_aspect_ratio, 0.05), '2f',
                 'to_position')
            ]
        )

    def _get_scalebar_label_text(self, physical_scalebar_length_kpc):
        if physical_scalebar_length_kpc < 1:
            return f"{physical_scalebar_length_kpc * 1000:.0f} pc"
        if physical_scalebar_length_kpc < 1000:
            return f"{physical_scalebar_length_kpc:.0f} kpc"
        else:
            return f"{physical_scalebar_length_kpc / 1000:.0f} Mpc"

    def _update_scalebar_label(self, physical_scalebar_length):
        if getattr(self, "_scalebar_label_is_for_length", None) != physical_scalebar_length:
            labelRgba = (text.text_to_rgba(self._get_scalebar_label_text(physical_scalebar_length), dpi=200,
                                           color='white') * 255).astype(
                dtype=np.int8)
            texture = self.ctx.texture(labelRgba.shape[1::-1], 4,
                                       labelRgba, dtype='f1')
            self.label_texture = texture
            self.label_aspect_ratio = labelRgba.shape[1] / labelRgba.shape[0]
            self._scalebar_label_is_for_length = physical_scalebar_length

    def _recommend_physical_scalebar_length(self):
        # target is for the scalebar to be around 1/3rd of the viewport
        # however the length is to be 10^n or 5*10^n, so we need to find the
        # closest power of 10 to 1/3rd of the viewport

        # in world coordinates the viewport is self.scale kpc wide
        # so we need to find the closest power of 10 to self.scale/3

        physical_scalebar_length = self.scale / 3.0
        # now quantize it:
        power_of_ten = np.floor(np.log10(physical_scalebar_length))
        mantissa = physical_scalebar_length / 10 ** power_of_ten
        if mantissa < 2.0:
            physical_scalebar_length = 10.0 ** power_of_ten
        elif mantissa < 5.0:
            physical_scalebar_length = 2.0 * 10.0 ** power_of_ten
        else:
            physical_scalebar_length = 5.0 * 10.0 ** power_of_ten
        return physical_scalebar_length

