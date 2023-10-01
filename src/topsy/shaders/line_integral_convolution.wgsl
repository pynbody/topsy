struct Params {
    scale: f32,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) texcoord: vec2<f32>
};

@group(0) @binding(0)
var<uniform> lic_params: Params;


@vertex
fn vertex_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var offsets = array<vec2<f32>, 4>(
            vec2(0.0, 0.0),
            vec2(0.0, 1.0),
            vec2(1.0, 0.0),
            vec2(1.0, 1.0)
          );

    var output: VertexOutput;

    var offset = offsets[vertexIndex];

    output.pos = vec4<f32>(offset*2.0 - 1.0, 0.0, 1.0);
    output.texcoord = offset;

    return output;
}

@group(0) @binding(1)
var image_texture: texture_2d<f32>;

@group(0) @binding(2)
var image_sampler: sampler;

 @group(0) @binding(3)
 var xy_offset: texture_2d<f32>;



@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var offset : vec2<f32> = textureSample(xy_offset, image_sampler, input.texcoord).rg * lic_params.scale;

    return 0.25*(textureSample(image_texture, image_sampler, input.texcoord)
                 + textureSample(image_texture, image_sampler, input.texcoord + offset*0.25)
                 + textureSample(image_texture, image_sampler, input.texcoord + offset*0.75)
                 + textureSample(image_texture, image_sampler, input.texcoord + offset)
                 );
    //return vec4<f32>(1.0,1.0,0.0,1.0);
}
