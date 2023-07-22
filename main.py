import moderngl_window as mglw
import moderngl
import numpy as np

vertex_shader = """
#version 330
uniform mat4 model;
uniform float scale;

in vec3 in_vert;
in float in_size;
out float color;


void main() {
    gl_Position = model * vec4(in_vert/scale, 1.0);
    color = 0.01;
    gl_PointSize = 500*in_size/scale;
}

"""

fragment_shader="""
#version 330
in float color;
out vec4 fragColor;
void main() {
    float distance = (gl_PointCoord.x-0.5)*(gl_PointCoord.x-0.5)+(gl_PointCoord.y-0.5)*(gl_PointCoord.y-0.5);
    fragColor = vec4(color,color,color,distance<0.25?1.0:0.0);
}
"""

vertex_shader_log = """

#version 330
in vec2 in_vert;
out vec2 textureLocation;

void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    textureLocation = in_vert/2+0.5;
}
"""

fragment_shader_log = """
#version 330

uniform sampler2D inputImage;
uniform sampler2D inputColorMap;
uniform float vmin;
uniform float vmax;
out vec4 fragColor;
in vec2 textureLocation;

void main() {
    vec4 tex = texture(inputImage, textureLocation);
    float scaledLogVal = clamp((log(tex.x)-vmin)/(vmax-vmin), 0.001, 0.999);
    fragColor = texture(inputColorMap, vec2(scaledLogVal,0.5));
    // fragColor = texture(inputImage, vec2(tex.x,0.5));
    // fragColor = vec4(tex.x,tex.x,tex.x,1);
    
}
    
"""

def get_colormap_texture(name='twilight_shifted', num_points=1000):
    import matplotlib.cm as cm
    cmap = cm.get_cmap(name)
    rgba = cmap(np.linspace(0.001, 0.999, num_points)).astype(np.float32)
    return rgba

class Test(mglw.WindowConfig):
    gl_version = (3, 3)
    aspect_ratio = None
    title = "pynbody-vis"
    # clear_color = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.render_resolution = (500, 500)

        self.model = np.eye(4, dtype=np.float32)

        pos, smooth = self.get_data()
        self.points = self.ctx.buffer(pos)
        self.smooth = self.ctx.buffer(smooth)

        self.texture = self.ctx.texture(self.render_resolution, 4,  dtype='f4')

        self.colormap_texture = self.ctx.texture((1000, 1), 4, get_colormap_texture(), dtype='f4')

        print(np.frombuffer(self.colormap_texture.read(),dtype='f4').reshape(1000,4))

        self.render_buffer = self.ctx.framebuffer(color_attachments=[self.texture])

        self.particleRenderer = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        self.scale = 100.0
        self.particleRenderer['scale'] = self.scale
        self.vertex_array = self.ctx.vertex_array(
            self.particleRenderer, [(self.points, '3f', 'in_vert'),
                                    (self.smooth, '1f', 'in_size')])

        self.model_matr = np.eye(4)

        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)

        self.vmin_vmax_is_set = False
        self.needs_render = True

        self.logMapper = self.ctx.program(vertex_shader=vertex_shader_log, fragment_shader=fragment_shader_log)
        self.log_mapper_vertex_array = self.ctx.simple_vertex_array(
            self.logMapper,
            self.ctx.buffer(np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0],
                                      [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]], dtype=np.float32)),
            'in_vert'
        )
        self.logMapper['inputImage'] = 0
        self.logMapper['inputColorMap'] = 1
        self.texture.use(0)
        self.colormap_texture.use(1)

    def get_data(self):
        import pynbody
        f = pynbody.load("/Users/app/Science/pynbody/testdata/g15784.lr.01024")
        f.physical_units()
        pynbody.analysis.halo.center(f.st)
        return f.dm['pos'].astype(np.float32), f.dm['smooth'].astype(np.float32)
    def render(self, time, frametime):



        self.model = self.model_matr.astype(np.float32)
        self.particleRenderer['model'].write(self.model)

        screenbuffer = self.ctx.fbo


        self.render_buffer.use()

        self.render_sph()

        if not self.vmin_vmax_is_set:
            self.set_vmin_vmax()


        screenbuffer.use()

        #self.render_sph()
        self.display_render_buffer()

        self.needs_render=False


    def set_vmin_vmax(self):
        vals = np.log(self.get_sph_result()).ravel()
        vals = vals[np.isfinite(vals)]

        vmin, vmax = np.percentile(vals, [1.0,99.9])
        print(len(vals), vals.min(), vals.max(), vmin, vmax)
        self.logMapper['vmin'] = vmin
        self.logMapper['vmax'] = vmax
        print("RANGE:",vmin,vmax)
        self.save()
        self.vmin_vmax_is_set = True

    def display_render_buffer(self):
        self.ctx.clear(1.0, 0.0, 0.0, 1.0)
        # self.vertex_array.render(moderngl.POINTS)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.log_mapper_vertex_array.render(moderngl.TRIANGLES)

    def render_sph(self):
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vertex_array.render(moderngl.POINTS)

    def get_sph_result(self):
        mybuffer = np.empty(self.render_resolution, dtype=np.float32)
        self.render_buffer.read_into(mybuffer, components=1, dtype='f4')
        return mybuffer

    def save(self):
        mybuffer = self.get_sph_result()
        import pylab as p
        p.imshow(np.log(mybuffer))
        p.colorbar()
        p.savefig("output.png")
        #exit(0)


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
        self.needs_render = True



    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        self.scale*=np.exp(y_offset*0.05)
        self.particleRenderer['scale'] = self.scale
        if abs(y_offset)>0.001:
            self.vmin_vmax_is_set = False

        self.needs_render = True




Test.run()
