struct TransformParams {
    transform: mat4x4<f32>,
    scale_factor: f32,
};

@group(0) @binding(0)
var<uniform> trans_params: TransformParams;


struct VertexInput {
   @location(0) pos: vec4<f32>, // NB w is used for the smoothing length
   @location(1) offset: vec2<f32>, // the offset is used to expand into a quad
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
    var output: VertexOutput;
    var size = trans_params.scale_factor*input.pos.w;
    // w is hijacked for the smoothing length. However since it will be used in the
    // transformation and clipping, we now need to set it to the 'standard' 1.0.
    output.pos = input.pos;
    output.pos.w = 1.0;
    output.pos = (trans_params.transform * output.pos);
    output.z = output.pos.z;
    output.pos.x+=size*(input.offset.x - 0.5);
    output.pos.y+=size*(input.offset.y - 0.5);
    output.texcoord = input.offset.xy;
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
