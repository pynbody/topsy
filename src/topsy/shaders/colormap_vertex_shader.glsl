#version 410
in vec2 in_vert;
out vec2 textureLocation;

uniform vec2 texturePortion;
uniform vec2 textureOffset;

void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    textureLocation = in_vert/2+0.5;
    textureLocation*=texturePortion;
    textureLocation+=textureOffset;
}

