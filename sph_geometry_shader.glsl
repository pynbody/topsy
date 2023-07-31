#version 330

layout(points) in;
layout(points, max_vertices = 1) out;

in float intensity[];
out float fragIntensity;

uniform int downsampleFactor;
uniform float smoothScale;

void main() {
    if(gl_PrimitiveIDIn%downsampleFactor == 0 ) {
        fragIntensity = intensity[0] * downsampleFactor;
        gl_Position = gl_in[0].gl_Position;
        gl_PointSize = gl_in[0].gl_PointSize * smoothScale;
        EmitVertex();
    }
}