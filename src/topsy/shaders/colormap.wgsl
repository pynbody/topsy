struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
}

struct FragmentOutput {
    @location(0) color: vec4<f32>
}

@vertex
fn vertex_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var pos = array<vec2<f32>, 4>(
        vec2(-1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, 1.0),
        vec2(1.0, -1.0)
      );

    var texc = array<vec2<f32>, 4>(
        vec2(0.0, 1.0),
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),
        vec2(1.0, 1.0)
      );

    var output: VertexOutput;

    output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    output.texcoord = texc[vertexIndex];

    return output;
}

@group(0) @binding(1)
var kernel_texture: texture_2d<f32>;

@group(0) @binding(2)
var kernel_sampler: sampler;

@fragment
fn fragment_main(input: VertexOutput) -> FragmentOutput {
    var output: FragmentOutput;

    var value = textureSample(kernel_texture, kernel_sampler, input.texcoord).r;

    output.color = vec4<f32>(value, value, value, 1.0);
    return output;
}
