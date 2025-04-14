struct TransformParams {
    transform: mat4x4<f32>,
    scale_factor: f32,
    clipspace_size_min: f32,
    clipspace_size_max: f32,
    boxsize_by_2_clipspace: f32
};

@group(0) @binding(0)
var<uniform> trans_params: TransformParams;


struct VertexInput {
   @location(0) pos: vec4<f32>, // NB w is used for the smoothing length
   [[WEIGHTED]] @location(1) mass: f32,
   [[WEIGHTED]] @location(2) quantity: f32,
   [[CHANNELED]] @location(1) rgb_mass: vec3<f32>,
   @builtin(vertex_index) vertexIndex: u32,
   @builtin(instance_index) instanceIndex: u32
}

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
    [[WEIGHTED]] @location(1) weight: f32,
    [[WEIGHTED]] @location(2) quantity: f32
    [[CHANNELED]] @location(1) channel_weights: vec4<f32>
}


struct FragmentOutput {
    [[DENSITY]] @location(0) output: f32,
    [[WEIGHTED]] @location(0) output: vec2<f32>,
    [[CHANNELED]] @location(0) output: vec4<f32>
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

    var smooth_length: f32 = input.pos.w;

    // factor 2: going out to 2h.
    var clipspace_size = trans_params.scale_factor*smooth_length*2.0;

    if(clipspace_size<trans_params.clipspace_size_min || clipspace_size>trans_params.clipspace_size_max) {
        output.pos.w = 1.0;
        output.pos.z = 100.0; // discard the vertex
        return output;
    }

    // perform transformation
    output.pos = input.pos;
    output.pos.w = 1.0;
    output.pos = (trans_params.transform * output.pos);
    output.pos += vec4<f32>(clipspace_size*posOffset[input.vertexIndex],0.0,0.0);
    output.texcoord = texCoords[input.vertexIndex];
    [[WEIGHTED]] output.weight = input.mass/(smooth_length*smooth_length);
    [[WEIGHTED]] output.quantity = input.quantity;

    [[DEPTH]] output.quantity = output.pos.z;

    [[CHANNELED]] output.channel_weights = vec4<f32>(input.rgb_mass,0) /(smooth_length*smooth_length);

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

    [[WEIGHTED]] value *= input.weight;
    [[WEIGHTED]] output.output = vec2<f32>(value, value*input.quantity);

    [[CHANNELED]] output.output = input.channel_weights * value;


    return output;
}
