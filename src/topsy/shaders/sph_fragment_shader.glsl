#version 410

in float fragIntensity;

out vec4 fragColor;
uniform sampler2D kernel;

void main() {
    float kernelValue = texture(kernel, gl_PointCoord).x;
    fragColor = vec4(fragIntensity,fragIntensity,fragIntensity,kernelValue);
}
