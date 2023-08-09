import numpy as np

class MultiresolutionGeometry:
    def __init__(self, n_pixels_max):
        """Class to calculate location of images in textures and viewports

        Args:
            n_pixels_max (int): Number of pixels for the width of the final image
        """
        self.n_pixels_max = n_pixels_max

    def get_npix_for_level(self, level):
        """Get the width/height of the image for a given resolution level"""
        return self.n_pixels_max // 2**level

    @classmethod
    def corners_to_triangles(cls, corners):
        return np.array([[corners[0], corners[1], corners[2]],
                         [corners[0], corners[2], corners[3]]])

    def get_viewport_corners_for_level(self, level):
        """Get the viewport corners of the image for a given resolution level"""
        if level==0:
            npix = self.n_pixels_max
            return np.array([[0, 0], [npix, 0], [npix, npix], [0, npix]])
        else:
            # lower left corner of level 1 is (self.n_pixels_max,0)
            # lower left corner of level 2 is (self.n_pixels_max,self.n_pixels_max//2)
            # lower left corner of level n is (self.n_pixels_max,self.n_pixels_max*(1-2**(1-n)))
            npix = self.get_npix_for_level(level)
            x0 = self.n_pixels_max
            y0 = np.ceil(self.n_pixels_max*(1.0-2.0**(1-level))).astype(np.int32)
            return np.array([[x0, y0], [x0+npix, y0], [x0+npix, y0+npix], [x0, y0+npix]])

    def get_buffer_dimensions(self):
        """Get the total render buffer dimensions (with all levels)"""
        return (self.n_pixels_max*3)//2, self.n_pixels_max

    def get_texture_corners_for_level(self, level):
        """Get the texture corners of the image for a given resolution level"""
        vp_corners = self.get_viewport_corners_for_level(level)
        return np.asarray(vp_corners, dtype=np.float32)/self.get_buffer_dimensions()

    def get_clipspace_corners(self):
        """Get the clipspace corners of the image when viewport is correctly configured"""
        return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])