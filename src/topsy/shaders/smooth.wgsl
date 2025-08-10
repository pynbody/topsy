struct SmoothingParams {
    spatial_sigma: f32,
    range_sigma: f32,
    kernel_size: i32,
    padding: i32,  // For 16-byte alignment
}

@group(0) @binding(0) var input_depth: texture_2d<f32>;
@group(0) @binding(1) var output_depth: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> params: SmoothingParams;

@compute @workgroup_size(8, 8)
fn bilateral_filter_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);
    let center_sample = textureLoad(input_depth, coord, 0);
    let center_depth = center_sample.g;

    var weighted_sum = 0.0;
    var weight_sum = 0.0;

    let half_kernel = params.kernel_size / 2;

    let tex_size = textureDimensions(input_depth, 0);

    // Sample neighborhood
    for (var dy = -half_kernel; dy <= half_kernel; dy++) {
        for (var dx = -half_kernel; dx <= half_kernel; dx++) {    
            let sample_coord = clamp(coord + vec2<i32>(dx, dy), vec2<i32>(0, 0), vec2<i32>(tex_size) - vec2<i32>(1, 1));
            let sample_depth = textureLoad(input_depth, sample_coord, 0).g;

            // Spatial weight (Gaussian based on distance)
            let spatial_dist = sqrt(f32(dx*dx + dy*dy));
            let w_spatial = exp(-(spatial_dist * spatial_dist) / (2.0 * params.spatial_sigma * params.spatial_sigma));

            // Range weight (Gaussian based on depth difference)
            let depth_diff = abs(sample_depth - center_depth);
            let w_range = exp(-(depth_diff * depth_diff) / (2.0 * params.range_sigma * params.range_sigma));

            let total_weight = w_spatial * w_range;
            weighted_sum += sample_depth * total_weight;
            weight_sum += total_weight;
        }
    }


    let filtered_depth = weighted_sum / weight_sum;
    textureStore(output_depth, coord, vec4<f32>(center_sample.r, filtered_depth, 0.0, 1.0));
}
