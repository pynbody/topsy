#version 410

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

        // To improve efficiency, large points are rendered in dinky viewports,
        // and everything will be added together in the end.
        //
        // At present, this is hardcoded to 5 levels of detail and
        // the if-elif chain may be inefficient -- TODO

        if(gl_PointSize>80.0) {
            gl_ViewportIndex = 4;
            gl_PointSize/=16.0;
        } else if(gl_PointSize>40.0) {
            gl_ViewportIndex = 3;
            gl_PointSize/=8.0;
        } else if(gl_PointSize>20.0) {
            gl_ViewportIndex = 2;
            gl_PointSize/=4.0;
        } else if(gl_PointSize>10.0) {
            gl_ViewportIndex = 1;
            gl_PointSize/=2.0;
        } else
            gl_ViewportIndex = 0;


        EmitVertex();
    }
}