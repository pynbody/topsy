#version 410

uniform sampler2D inputImage;

out vec4 fragColor;


in vec2 textureLocation;

void main() {
    fragColor = texture(inputImage, textureLocation);
}

