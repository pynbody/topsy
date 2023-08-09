#version 410

in float fragIntensity;

out vec4 fragColor;
uniform sampler2D kernel;

void main() {
    float kernelValue = texture(kernel, gl_PointCoord).x;
    fragColor = vec4(fragIntensity,fragIntensity,fragIntensity,kernelValue);
    // fac 4 to put the whole kernel in the (0,1) range (it's parameterised in distance^2 and distance goes up to 2)
}
