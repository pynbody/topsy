struct LineRenderParams {
    transform: mat4x4<f32>,
    color: vec4<f32>,
    vp_size_pix: vec2<f32>,
    linewidth_pix: f32,
};

@group(0) @binding(0)
var<uniform> render_params: LineRenderParams;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};


@vertex
fn vertex_main(@location(0) instanceStart : vec4<f32>,
               @location(1) instanceEnd : vec4<f32>,
               @builtin(vertex_index) vertexIndex : u32) -> VertexOutput {

    var output: VertexOutput;

    var instanceStartTransformed : vec2<f32> = (render_params.transform * instanceStart).xy * render_params.vp_size_pix;
    var instanceEndTransformed : vec2<f32> = (render_params.transform * instanceEnd).xy * render_params.vp_size_pix;
    var normalizedOffsetVector : vec2<f32> = normalize(instanceEndTransformed - instanceStartTransformed);
    var normalToLine : vec2<f32> = vec2<f32>(-normalizedOffsetVector.y, normalizedOffsetVector.x);

    switch vertexIndex {
        case 0u {
           output.pos = vec4<f32>(instanceStartTransformed - normalToLine * render_params.linewidth_pix * 0.5, 0.0, 1.0);
        }
        case 1u: {
           output.pos = vec4<f32>(instanceStartTransformed + normalToLine * render_params.linewidth_pix * 0.5, 0.0, 1.0);
        }
        case 2u: {
           output.pos = vec4<f32>(instanceEndTransformed - normalToLine * render_params.linewidth_pix * 0.5, 0.0, 1.0);
        }
        case 3u: {
           output.pos = vec4<f32>(instanceEndTransformed + normalToLine * render_params.linewidth_pix * 0.5, 0.0, 1.0);
        }
        default: {
           output.pos = vec4<f32>(0.0);
        }
    }

    output.pos/=vec4<f32>(render_params.vp_size_pix, 1.0, 1.0);

    output.color = render_params.color;

    return output;
}



@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
   return input.color;
}
