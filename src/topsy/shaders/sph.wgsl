struct TransformParams {
    transform: mat4x4<f32>,
    scale_factor: f32,
    clipspace_size_min: f32,
    clipspace_size_max: f32,
    boxsize_by_2_clipspace: f32
};

struct VertexInput {
   @location(0) pos: vec4<f32>, // NB w is used for the smoothing length
   @location(1) quantities: vec3<f32>,
   @builtin(vertex_index) vertexIndex: u32,
   @builtin(instance_index) instanceIndex: u32
}

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
    @location(1) intensities: vec3<f32>
}

@group(0) @binding(0)
var<uniform> trans_params: TransformParams;

@group(0) @binding(1)
var kernel_texture: texture_2d<f32>;

@group(0) @binding(2)
var kernel_sampler: sampler;



// triangle position offsets for making a square of 2 units side length
const posOffset = array<vec2<f32>, 6>(
            vec2(-1.0, -1.0),
            vec2(-1.0, 1.0),
            vec2(1.0, 1.0),
            vec2(1.0, -1.0),
            vec2(-1.0, -1.0),
            vec2(1.0, 1.0)
          );

// corresponding texture coordinates
const texCoords = array<vec2<f32>, 6>(
            vec2(0.0, 0.0),
            vec2(0.0, 1.0),
            vec2(1.0, 1.0),
            vec2(1.0, 0.0),
            vec2(0.0, 0.0),
            vec2(1.0, 1.0)
          );

fn vertex_calculate_positions(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // factor 2: going out to 2h.
    var clipspace_size = trans_params.scale_factor*input.pos.w*2.0;

    output.pos = input.pos;
    output.pos.w = 1.0;
    output.pos = (trans_params.transform * output.pos);
    output.pos += vec4<f32>(clipspace_size*posOffset[input.vertexIndex],0.0,0.0);
    output.texcoord = texCoords[input.vertexIndex];
    return output;
}

@vertex
fn vertex_rgb(input: VertexInput) -> VertexOutput {
    var output: VertexOutput = vertex_calculate_positions(input);
    output.intensities = input.quantities/(input.pos.w * input.pos.w);
    return output;
}

@vertex
fn vertex_weighting(input: VertexInput) -> VertexOutput {
    var output: VertexOutput = vertex_calculate_positions(input);

    output.intensities.x = input.quantities.x/(input.pos.w * input.pos.w);
    output.intensities.y = input.quantities.y;

    return output;
}

@vertex
fn vertex_depth(input: VertexInput) -> VertexOutput {
    var output: VertexOutput = vertex_calculate_positions(input);
    output.intensities.x = input.quantities.x/(input.pos.w * input.pos.w);
    output.intensities.y = output.pos.z;
    return output;
}


struct FragmentOutputWeighting {
    @location(0) output: vec2<f32>
}

struct FragmentOutputRGB {
    @location(0) output: vec4<f32>
}

@fragment
fn fragment_weighting(input: VertexOutput) -> FragmentOutputWeighting {
    var value = textureSample(kernel_texture, kernel_sampler, input.texcoord).r;

    value *= input.intensities.x;
    var output = FragmentOutputWeighting(vec2<f32>(value, value*input.intensities.y));

    return output;
}

@fragment
fn fragment_rgb(input: VertexOutput) -> FragmentOutputRGB {
    var value = textureSample(kernel_texture, kernel_sampler, input.texcoord).r;
    var output = FragmentOutputRGB(vec4<f32>(input.intensities * value, 1.0));
    return output;
}
