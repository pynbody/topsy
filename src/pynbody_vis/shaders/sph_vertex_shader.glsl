#version 330

uniform mat4 model;
uniform float scale;
uniform float outputResolution;

in vec3 in_pos;
in float in_smooth;
in float in_mass;

out float intensity;

void main() {
    gl_Position = model * vec4(in_pos/scale, 1.0);
    intensity = in_mass/(in_smooth*in_smooth); // TODO

    gl_PointSize = 2*outputResolution*2*2*in_smooth/scale;
    // 2 because it's the diameter not the radius measured here, another 2 to go out to 2h not h

}
