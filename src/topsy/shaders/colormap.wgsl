struct ColormapParams {
    vmin: f32,
    vmax: f32,
    density_vmin: f32, // used only in bivariate case
    density_vmax: f32, // used only in bivariate case
    window_aspect_ratio: f32,
    gamma: f32
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

#ifdef BIVARIATE
@group(0) @binding(2)
var colormap_texture: texture_2d<f32>;
#else
@group(0) @binding(2)
var colormap_texture: texture_1d<f32>;
#endif

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

fn log10(value: f32) -> f32 {
    return log(value)/2.30258509;
}

@fragment
fn fragment_main(input: VertexOutput) -> FragmentOutput {
    var output: FragmentOutput;

    var values = textureSample(image_texture, image_sampler, input.texcoord);

    var value : f32;

    // Note the following lines are selected by python before compile time
    // ultimately, this should be possible within wgsl itself by using a const, but this doesn't
    // seem to be supported at present

    #ifdef BIVARIATE
        var value_2d : vec2<f32>;
        value_2d.x = log10(values.r);
        value_2d.x -= colormap_params.density_vmin;
        value_2d.x /= (colormap_params.density_vmax - colormap_params.density_vmin);

        #ifdef WEIGHTED_MEAN
            value_2d.y = values.g / values.r;
        #else
            value_2d.y = values.r;
        #endif

        #ifdef LOG_SCALE
            value_2d.y = log10(value_2d.y);
        #endif

        value_2d.y -= colormap_params.vmin;
        value_2d.y /= (colormap_params.vmax - colormap_params.vmin);

        value_2d = clamp(value_2d, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
        output.color = textureSample(colormap_texture, colormap_sampler, value_2d);

    #else // not BIVARIATE

        #ifdef WEIGHTED_MEAN
            value = values.g/values.r;
        #else
            value = values.r;
        #endif

        #ifdef LOG_SCALE
            value = log10(value);
        #endif

        value = clamp((value-colormap_params.vmin)/(colormap_params.vmax-colormap_params.vmin), 0.0, 1.0);
        output.color = textureSample(colormap_texture, colormap_sampler, value);
    #endif // not BIVARIATE
    return output;
}

fn gamma_map(value: f32, vmin: f32, vmax: f32, gamma: f32) -> f32 {
    return pow(max((value - vmin)/(vmax - vmin), 0.0), gamma);
}

@fragment
fn fragment_main_tri(input: VertexOutput) -> FragmentOutput {
    var output: FragmentOutput;
    var value_r: f32;
    var value_g: f32;
    var value_b: f32;

    value_r = textureSample(image_texture, image_sampler, input.texcoord).r;
    value_g = textureSample(image_texture, image_sampler, input.texcoord).g;
    value_b = textureSample(image_texture, image_sampler, input.texcoord).b;

#ifdef LOG_SCALE
    value_r = log10(value_r);
    value_g = log10(value_g);
    value_b = log10(value_b);
#endif

    value_r = gamma_map(value_r, colormap_params.vmin, colormap_params.vmax, colormap_params.gamma);
    value_g = gamma_map(value_g, colormap_params.vmin, colormap_params.vmax, colormap_params.gamma);
    value_b = gamma_map(value_b, colormap_params.vmin, colormap_params.vmax, colormap_params.gamma);

    output.color = vec4<f32>(value_r, value_g, value_b, 1.0);

    return output;
}