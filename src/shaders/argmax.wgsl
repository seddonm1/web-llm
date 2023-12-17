@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, T, B]

@group(0) @binding(1) var<storage, read_write> x: array<f32>;         // (B, T, C)
@group(0) @binding(2) var<storage, read_write> y: array<u32>;         // (B, T, C)

const BLOCK_SIZE: u32 = 128u;

var<workgroup> sketch_max: array<f32, BLOCK_SIZE>;
var<workgroup> sketch_pos: array<u32, BLOCK_SIZE>;
var<workgroup> maximum: f32;

fn reduce_max(index: u32, stride: u32) {
    if index < stride {
        if sketch_max[index + stride] > sketch_max[index] {
            sketch_max[index] = sketch_max[index + stride];
            sketch_pos[index] = sketch_pos[index + stride];
        }
    }
    workgroupBarrier();
}

@compute @workgroup_size(128, 1, 1)
fn argmax(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;


    sketch_max[index] = 0.0;
    for (var i = index; i < shape[0]; i += BLOCK_SIZE) {
        if x[i] > sketch_max[index] {
            sketch_max[index] = x[i];
            sketch_pos[index] = i;
        }
    }
    workgroupBarrier();

    reduce_max(index, 64u);
    reduce_max(index, 32u);
    reduce_max(index, 16u);
    reduce_max(index, 8u);
    reduce_max(index, 4u);
    reduce_max(index, 2u);
    reduce_max(index, 1u);

    if index == 0u {
        y[0] = sketch_pos[0];
    }
}