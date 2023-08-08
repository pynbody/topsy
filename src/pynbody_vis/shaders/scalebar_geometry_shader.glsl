#version 410

layout(points) in;
layout(triangle_strip, max_vertices = 6) out;

in float out_length[];
in vec4 out_pos[];

void main() {
    gl_Position = out_pos[0];
    EmitVertex();
    gl_Position = out_pos[0] + vec4(out_length[0], 0.0, 0.0, 0.0);
    EmitVertex();
    gl_Position = out_pos[0] + vec4(0.0, 0.02, 0.0, 0.0);
    EmitVertex();
    gl_Position = out_pos[0] + vec4(out_length[0], 0.02, 0.0, 0.0);
    EmitVertex();
}
