// Vertex and fragment shaders for surface rendering

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
}

// Fragment shader
@group(0) @binding(0) var colorTexture: texture_2d<f32>;
@group(0) @binding(1) var textureSampler: sampler;

struct Uniforms {
    depthScale: f32,        // Scale factor for depth values
    lightDirection: vec3<f32>, // Direction to light source
    lightColor: vec3<f32>,     // Light color
    ambientColor: vec3<f32>,   // Ambient light color
    texelSize: vec2<f32>,      // 1.0 / texture dimensions
    windowAspectRatio: f32,    // Window aspect ratio for proper scaling
    vmin: f32,
    vmax: f32
}

@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@group(0) @binding(3)
var colormap_texture: texture_1d<f32>;


fn sampleDepth(coord: vec2<f32>) -> f32 {
    let samp = textureSample(colorTexture, textureSampler, coord);
    return samp.g * uniforms.depthScale; // Using alpha channel for depth
}

// Compute surface normal using finite differences
fn computeNormal(texCoord: vec2<f32>) -> vec3<f32> {
    let texelSize = uniforms.texelSize;

    // Sample depth at neighboring pixels

    let depthLeft   = sampleDepth(texCoord + vec2<f32>(-texelSize.x, 0.0));
    let depthRight  = sampleDepth(texCoord + vec2<f32>(texelSize.x, 0.0));
    let depthUp     = sampleDepth(texCoord + vec2<f32>(0.0, -texelSize.y));
    let depthDown   = sampleDepth(texCoord + vec2<f32>(0.0, texelSize.y));

    // Compute gradients using central differences
    let dX = (depthRight - depthLeft) * 0.5;
    let dY = (depthDown - depthUp) * 0.5;

    // Construct normal vector
    // The normal points "outward" from the surface
    let normal = normalize(vec3<f32>(-dX, -dY, texelSize.x));

    return normal;
}


fn computeLighting(texCoord: vec2<f32>, materialColor: vec3<f32>) -> vec3<f32> {
    let normal = computeNormal(texCoord);
    let depthCenter = sampleDepth(texCoord);
    let lightDir = uniforms.lightDirection;
    let NdotL = max(dot(normal, lightDir), 0.0);

    let diffuse = uniforms.lightColor * NdotL * materialColor;
    let ambient = uniforms.ambientColor * materialColor;

    return (diffuse + ambient)*clamp(depthCenter, 0.0, 0.5)*2.0;
}

@vertex
fn vertex_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var pos = array<vec2<f32>, 4>(
        vec2(-1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, -1.0),
        vec2(1.0, 1.0)
    );

    // Apply aspect ratio scaling
    if (uniforms.windowAspectRatio > 1.0) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            pos[i].y = pos[i].y * uniforms.windowAspectRatio;
        }
    } else {
        for (var i = 0u; i < 4u; i = i + 1u) {
            pos[i].x = pos[i].x / uniforms.windowAspectRatio;
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
    output.texCoord = texc[vertexIndex];

    return output;
}

fn log10(value: f32) -> f32 {
    return log(value)/2.30258509;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
#ifdef MATERIAL_COLORMAP
    var value = textureSample(colorTexture, textureSampler, input.texCoord).r;
    // NB above could be optimized by combining it with the g sample taken for the depth elsewhere

#ifdef MATERIAL_LOG
    value = log10(value);
#endif
    value = clamp((value - uniforms.vmin) / (uniforms.vmax - uniforms.vmin), 0.0, 1.0);
    let materialColor = textureSample(colormap_texture, textureSampler, value).rgb;
#else
    let materialColor = vec3<f32>(1.0, 1.0, 1.0);
#endif
    let lighting = computeLighting(input.texCoord, materialColor);
    return vec4<f32>(lighting, 1.0);
}
