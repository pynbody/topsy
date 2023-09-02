struct ColormapParams {
    vmin: f32,
    vmax: f32,
    window_aspect_ratio: f32
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
}

struct FragmentOutput {
    @location(0) color: vec4<f32>
}


@group(0) @binding(0)
var image_texture: texture_2d<f32>;

@group(0) @binding(1)
var image_sampler: sampler;

@group(0) @binding(2)
var colormap_texture: texture_1d<f32>;

@group(0) @binding(3)
var colormap_sampler: sampler;

@group(0) @binding(4)
var<uniform> colormap_params: ColormapParams;


@vertex
fn vertex_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var pos = array<vec2<f32>, 4>(
            vec2(-1.0, -1.0),
            vec2(-1.0, 1.0),
            vec2(1.0, -1.0),
            vec2(1.0, 1.0)
          );

     if(colormap_params.window_aspect_ratio>1.0) {
        for(var i = 0u; i<4u; i=i+1u) {
            pos[i].y = pos[i].y*colormap_params.window_aspect_ratio;
        }
     } else {
         for(var i = 0u; i<4u; i=i+1u) {
                pos[i].x = pos[i].x/colormap_params.window_aspect_ratio;
          }
     }

    var texc = array<vec2<f32>, 4>(
            vec2(0.0, 1.0),
            vec2(0.0, 0.0),
            vec2(1.0, 1.0),
            vec2(1.0, 0.0)
          );

    var output: VertexOutput;

    output.pos = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    output.texcoord = texc[vertexIndex];

    return output;
}


@fragment
fn fragment_main(input: VertexOutput) -> FragmentOutput {
    var LN_10 = 2.30258509;

    var output: FragmentOutput;

    var values = textureSample(image_texture, image_sampler, input.texcoord);

    var value : f32;

    // Note the following lines are selected by python before compile time
    // ultimately, this should be possible within wgsl itself by using a const, but this doesn't
    // seem to be supported at present

    [[WEIGHTED_MEAN]] value = values.g/values.r;
    [[DENSITY]] value = values.r;
    [[LOG_SCALE]] value = log(value)/LN_10;

    value = clamp((value-colormap_params.vmin)/(colormap_params.vmax-colormap_params.vmin), 0.0, 1.0);
    output.color = textureSample(colormap_texture, colormap_sampler, value);

    return output;
}
