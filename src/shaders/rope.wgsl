struct Params {
    dim: u32,
    head_size: u32,
    pos: u32,
    kv_dim: u32
};

struct View {
    stride: vec4<u32>,
    offset: vec4<u32>,
    shape: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: Params;

@group(0) @binding(1) var<uniform> q_view: View;
@group(0) @binding(2) var<uniform> k_view: View;

@group(0) @binding(3) var<storage, read_write> q: array<f32>;
@group(0) @binding(4) var<storage, read_write> k: array<f32>;

fn compute_index(view: View, batch: u32, token: u32, index: u32) -> u32 {
    let stride = view.stride.x / 1u;
    let offset = view.offset.x / 1u;
    return ((view.offset.z + batch) * view.stride.y + view.offset.y + token) * stride + offset + index;
}

// must be 2x workgroup[0] size due to the `index * 2u` indexing
const BLOCK_SIZE: u32 = 256u;

@compute @workgroup_size(128, 1, 1)
fn rope(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    let pos = f32(params.pos);
    let head_size_f32 = f32(params.head_size);

    for (var i = index * 2u; i < params.dim; i += BLOCK_SIZE) {
        let head_dim = f32(i % params.head_size);
        let freq = 1.0 / pow(10000.0, head_dim / head_size_f32);
        let val = pos * freq;
        let fcr = cos(val);
        let fci = sin(val);
        var rotn = 1;
        if i < params.kv_dim {
            rotn = 2;
        }
        for (var v = 0; v < rotn; v++) {
            if v == 0 {
                // rotate query
                let bti = compute_index(q_view, batch, token, i);
                let q0 = q[bti];
                let q1 = q[bti + 1u];
                q[bti] = q0 * fcr - q1 * fci;
                q[bti + 1u] = q0 * fci + q1 * fcr;
            } else {
                // rotate key
                let bti = compute_index(k_view, batch, token, i);
                let k0 = k[bti];
                let k1 = k[bti + 1u];
                k[bti] = k0 * fcr - k1 * fci;
                k[bti + 1u] = k0 * fci + k1 * fcr;
            }
        }
    }
}