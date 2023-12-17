@group(0) @binding(0) var<uniform> shape: vec4<u32>;

@group(0) @binding(1) var<storage, read_write> io: array<f32>;

const BLOCK_SIZE: u32 = 8u;

@compute @workgroup_size(8, 1, 1)
fn compute(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;
    _ = shape[0];

    let n_heads = 6u;
    let head_size = 48u;

    for (var head_index = index; head_index < n_heads; head_index += BLOCK_SIZE) {
        let head_size_offset = head_index * head_size;
        for (var i = 0u; i < head_size; i++) {
            io[head_size_offset + i] = f32(index);
        }
    }
}