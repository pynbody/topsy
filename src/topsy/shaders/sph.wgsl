struct TransformParams {
    transform: mat4x4<f32>,
    scale_factor: f32,
    clipspace_size_min: f32,
    clipspace_size_max: f32
};

@group(0) @binding(0)
var<uniform> trans_params: TransformParams;


struct VertexInput {
   @location(0) pos: vec4<f32>, // NB w is used for the smoothing length
   @builtin(vertex_index) vertexIndex: u32
}

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
    @location(1) z: f32
}


struct FragmentOutput {
    @location(0) color: vec4<f32>
}


@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    // triangle position offsets for making a square of 2 units side length
    var posOffset = array<vec2<f32>, 6>(
                vec2(-1.0, -1.0),
                vec2(-1.0, 1.0),
                vec2(1.0, 1.0),
                vec2(1.0, -1.0),
                vec2(-1.0, -1.0),
                vec2(1.0, 1.0)
              );

    // corresponding texture coordinates
    var texCoords = array<vec2<f32>, 6>(
                vec2(0.0, 0.0),
                vec2(0.0, 1.0),
                vec2(1.0, 1.0),
                vec2(1.0, 0.0),
                vec2(0.0, 0.0),
                vec2(1.0, 1.0)
              );

    var output: VertexOutput;

    // smoothing length is stored in w
    var size = trans_params.scale_factor*input.pos.w;

    if(size<trans_params.clipspace_size_min || size>trans_params.clipspace_size_max) {
        output.pos.z = 100.0; // discard the vertex
    } else {
        // perform transformation
        output.pos = input.pos;
        output.pos.w = 1.0;
        output.pos = (trans_params.transform * output.pos);
        output.z = output.pos.z;
        output.pos += vec4<f32>(size*posOffset[input.vertexIndex],0.0,0.0);
        output.texcoord = texCoords[input.vertexIndex];
    }
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
