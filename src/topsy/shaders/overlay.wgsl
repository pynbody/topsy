struct OverlayParams {
    clipspace_origin: vec2<f32>,
    clipspace_extent: vec2<f32>,
    texturespace_origin: vec2<f32>,
    texturespace_extent: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> overlay_params: OverlayParams;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
    @location(1) weight: f32,
};


@vertex
fn vertex_main(@builtin(vertex_index) vertexIndex : u32,
               @location(0) instanceOffset : vec2<f32>,
               @location(1) instanceWeight : f32) -> VertexOutput {
    var offsets = array<vec2<f32>, 4>(
            vec2(0.0, 0.0),
            vec2(0.0, 1.0),
            vec2(1.0, 0.0),
            vec2(1.0, 1.0)
          );

    var output: VertexOutput;

    var vertexOffset = offsets[vertexIndex];
    vertexOffset.y = 1.0 - vertexOffset.y;
    vertexOffset *= overlay_params.clipspace_extent;

    output.pos = vec4<f32>(overlay_params.clipspace_origin + instanceOffset + vertexOffset, 0.0, 1.0);
    output.texcoord = overlay_params.texturespace_origin + offsets[vertexIndex]*overlay_params.texturespace_extent;
    output.weight = instanceWeight;
    return output;
}

@group(0) @binding(1)
var image_texture: texture_2d<f32>;

@group(0) @binding(2)
var image_sampler: sampler;


@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
   return textureSample(image_texture, image_sampler, input.texcoord) * input.weight;
}
