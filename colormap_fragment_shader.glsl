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

