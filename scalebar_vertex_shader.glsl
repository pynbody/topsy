#version 410

uniform float length;

in vec2 in_pos;

out float out_length;
out vec4 out_pos;

void main() {
    out_pos = vec4(in_pos, 0, 1);
    // out_pos = vec4(-0.5,-0.5,0,1); // temporary
    out_length = length;
}
