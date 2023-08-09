#version 410

uniform sampler2D inputImage;
uniform sampler2D inputColorMap;
uniform float vmin;
uniform float vmax;
out vec4 fragColor;
in vec2 textureLocation;

#define LOG_10 2.302585093

void main() {
    vec4 tex = texture(inputImage, textureLocation);
    float scaledLogVal = clamp((log(tex.x)/LOG_10-vmin)/(vmax-vmin), 0.001, 0.999);
    fragColor = texture(inputColorMap, vec2(scaledLogVal,0.5));
}

