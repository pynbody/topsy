struct OverlayParams {
    origin: vec2<f32>,
    extent: vec2<f32>
};

@group(0) @binding(0)
var<uniform> overlay_params: OverlayParams;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
};


@vertex
fn vertex_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var texc = array<vec2<f32>, 4>(
            vec2(0.0, 0.0),
            vec2(0.0, 1.0),
            vec2(1.0, 0.0),
            vec2(1.0, 1.0)
          );

    var output: VertexOutput;

    output.texcoord = texc[vertexIndex];

    var posOffset = output.texcoord;
    posOffset.y = 1.0 - posOffset.y;
    posOffset *= overlay_params.extent;

    output.pos = vec4<f32>(overlay_params.origin + posOffset, 0.0, 1.0);

    return output;
}

@group(0) @binding(1)
var image_texture: texture_2d<f32>;

@group(0) @binding(2)
var image_sampler: sampler;


@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
   return textureSample(image_texture, image_sampler, input.texcoord);
}
