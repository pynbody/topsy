import moderngl_window as mglw
import moderngl
import numpy as np

vertex_shader = """
#version 330
uniform mat4 model;
in vec3 in_vert;
out float color;

void main() {
    gl_Position = model * vec4(in_vert, 1.0);
    color = 0.01;
    gl_PointSize = 20.0;
}

"""

fragment_shader="""
#version 330
in float color;
out float fragColor;
void main() {
    fragColor = color;
    float distance = (gl_PointCoord.x-0.5)*(gl_PointCoord.x-0.5)+(gl_PointCoord.y-0.5)*(gl_PointCoord.y-0.5);
    fragColor*=exp(-distance*10);
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
out vec4 fragColor;
in vec2 textureLocation;

void main() {
    vec4 tex = texture(inputImage, textureLocation);
    fragColor = texture(inputColorMap, vec2(clamp(log(tex.x),0,1),0.5));
    // fragColor = texture(inputImage, vec2(tex.x,0.5));
    // fragColor = vec4(tex.x,tex.x,tex.x,1);
    
}
    
"""

def get_colormap_texture(name='bone', num_points=1000):
    import matplotlib.cm as cm
    cmap = cm.get_cmap(name)
    rgba = cmap(np.linspace(0, 1, num_points)).astype(np.float32)
    print(rgba)
    return rgba

class Test(mglw.WindowConfig):
    gl_version = (3, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        NUM_VERTICES = 200000
        vals = np.random.normal(0.0, 0.1, (NUM_VERTICES, 3)).astype(np.float32)

        self.model = np.eye(4, dtype=np.float32)

        self.points = self.ctx.buffer(vals)

        self.texture = self.ctx.texture((500, 500), 4,  dtype='f4')

        self.colormap_texture = self.ctx.texture((1000, 1), 4, get_colormap_texture(), dtype='f4')

        print(np.frombuffer(self.colormap_texture.read(),dtype='f4').reshape(1000,4))

        self.render_buffer = self.ctx.framebuffer(color_attachments=[self.texture])

        self.particleRenderer = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        self.vertex_array = self.ctx.simple_vertex_array(
            self.particleRenderer, self.points, 'in_vert')

        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)


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


    def render(self, time, frametime):

        # now rotate the model matrix:
        self.model = np.array([[np.cos(time), 0.0 ,  -np.sin(time), 0.0],
            [0.0 ,1.0 , 0.0, 0.0],
            [np.sin(time), 0.0,np.cos(time), 0.0],
            [0., 0., 0., 1.]], dtype=np.float32)

        self.particleRenderer['model'].write(self.model)

        screenbuffer = self.ctx.fbo

        self.render_buffer.use()

        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.ctx.clear(0.1, 0.0, 0.0, 1.0)
        self.vertex_array.render(moderngl.POINTS)

        screenbuffer.use()
       



        self.ctx.clear(1.0, 0.5, 0.5, 1.0)
        #self.vertex_array.render(moderngl.POINTS)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.log_mapper_vertex_array.render(moderngl.TRIANGLES)

        # self.ctx.copy_framebuffer(screenbuffer, self.render_buffer)

        #self.render_buffer.read_into(mybuffer, dtype='f4')
        #import pylab as p
        #p.imshow(np.log10(mybuffer[:,:,1]))
        #p.show()
        #exit(0)



Test.run()
