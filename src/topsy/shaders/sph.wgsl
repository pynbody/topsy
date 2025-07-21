struct TransformParams {
    transform: mat4x4<f32>,
    scale_factor: f32,
    clipspace_size_min: f32,
    clipspace_size_max: f32,
    boxsize_by_2_clipspace: f32,
    density_cut: f32
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

@vertex
fn vertex_depth_with_cut(input: VertexInput) -> VertexOutput {
    // This could be made more efficient by a compute shader passing once through the buffer
    // which would only need to be updated when the user changed the density threshold
    var result: VertexOutput;

    var rho: f32 = input.quantities.x / pow(input.pos.w,3.0f);

    if(rho > trans_params.density_cut) {
        result = vertex_calculate_positions(input);
        result.intensities.x = input.quantities.y; // quantity value
        result.intensities.y = result.pos.z; // depth value

        // "z" component of "intensities" is actually the depth scale of the sphere to be
        // rendered on this tile
        //
        // Factors: Sphere extends to 2*h, but that's already baked into the kernel image
        // input.pos.w*trans_params.scale_factor gives the extent of h in (x,y) clip space,
        // but note the z direction is squsiehd into (0,1) while (x,y) are in (-1,1)
        // so there is a factor of 0.5 in the z direction.
        result.intensities.z = input.pos.w * trans_params.scale_factor*0.5;
    } else {
        // put somewhere out of the clip space:
        result.pos.x = 100;
        result.pos.y = 100;
        result.pos.w = 100;
    }

    return result;
}


struct FragmentOutputWeighting {
    @location(0) output: vec2<f32>
}

struct FragmentOutputRGB {
    @location(0) output: vec4<f32>
}

struct FragmentOutputRaw {
    @location(0) output: vec2<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fragment_weighting(input: VertexOutput) -> FragmentOutputWeighting {
    var value = textureSample(kernel_texture, kernel_sampler, input.texcoord).r;

    value *= input.intensities.x;
    var output = FragmentOutputWeighting(vec2<f32>(value, value*input.intensities.y));

    return output;
}

@fragment
fn fragment_raw(input: VertexOutput) -> FragmentOutputRaw {
    var value = textureSample(kernel_texture, kernel_sampler, input.texcoord).r;
    var depth: f32 = input.intensities.y + input.intensities.z*value;

    if (value<0.0) {
        discard;
    }

    return FragmentOutputRaw(vec2<f32>(input.intensities.x, depth), depth);
}

@fragment
fn fragment_rgb(input: VertexOutput) -> FragmentOutputRGB {
    var value = textureSample(kernel_texture, kernel_sampler, input.texcoord).r;
    var output = FragmentOutputRGB(vec4<f32>(input.intensities * value, 1.0));
    return output;
}
