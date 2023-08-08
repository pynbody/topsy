#version 410
in vec2 from_position;
in vec2 to_position;

out vec2 textureLocation;

void main() {
    gl_Position = vec4(to_position, 0.0, 1.0);
    textureLocation = from_position;
}

