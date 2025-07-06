// Vertex shader is shared with standard colormap

// Fragment shader
@group(0) @binding(0) var colorTexture: texture_2d<f32>;
@group(0) @binding(1) var textureSampler: sampler;

struct Uniforms {
    depthScale: f32,        // Scale factor for depth values
    lightDirection: vec3<f32>, // Direction to light source
    lightColor: vec3<f32>,     // Light color
    ambientColor: vec3<f32>,   // Ambient light color
    texelSize: vec2<f32>,      // 1.0 / texture dimensions
}

@group(0) @binding(2) var<uniform> uniforms: Uniforms;

fn sampleDepth(coord: vec2<f32>) -> f32 {
    let samp = textureSample(colorTexture, textureSampler, coord);
    return samp.g * uniforms.depthScale; // Using alpha channel for depth
}

// Compute surface normal using finite differences
fn computeNormal(texCoord: vec2<f32>) -> vec3<f32> {
    let texelSize = uniforms.texelSize;

    // Sample depth at neighboring pixels
    let depthCenter = sampleDepth(texCoord);
    let depthLeft   = sampleDepth(texCoord + vec2<f32>(-texelSize.x, 0.0));
    let depthRight  = sampleDepth(texCoord + vec2<f32>(texelSize.x, 0.0));
    let depthUp     = sampleDepth(texCoord + vec2<f32>(0.0, -texelSize.y));
    let depthDown   = sampleDepth(texCoord + vec2<f32>(0.0, texelSize.y));

    // Compute gradients using central differences
    let dX = (depthRight - depthLeft) * 0.5;
    let dY = (depthDown - depthUp) * 0.5;

    // Construct normal vector
    // The normal points "outward" from the surface
    let normal = normalize(vec3<f32>(-dX, -dY, 1.0));

    return normal;
}

// Simple Lambertian lighting
fn computeLighting(normal: vec3<f32>) -> vec3<f32> {
    let lightDir = uniforms.lightDirection;
    let NdotL = max(dot(normal, lightDir), 0.0);

    let diffuse = uniforms.lightColor * NdotL;
    let ambient = uniforms.ambientColor;

    return diffuse + ambient;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample the original color (RGB channels)
    let originalColor = textureSample(colorTexture, textureSampler, input.texCoord);

    // Compute surface normal from depth
    let normal = computeNormal(input.texCoord);

    // Compute lighting
    let lighting = computeLighting(normal);

    return vec4<f32>(lighting, 1.0);
}
