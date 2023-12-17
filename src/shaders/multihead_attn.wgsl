struct Params {
    seq_len: u32,
    kv_dim: u32,
    kv_mul: u32,
    n_heads: u32,
    head_size: u32,
    pos: u32,
};

@group(0) @binding(0) var<uniform> params: Params;

@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> key_cache: array<f32>;
@group(0) @binding(3) var<storage, read> value_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> attn: array<f32>;
@group(0) @binding(5) var<storage, read_write> xb: array<f32>;

const BLOCK_SIZE: u32 = 6u;

@compute @workgroup_size(6, 1, 1)
fn multihead_attn(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    let head_size_sqrt = sqrt(f32(params.head_size));

    for (var head_index = index; head_index < params.n_heads; head_index += BLOCK_SIZE) {
        let head_offset = head_index * params.seq_len;
        let head_size_offset = head_index * params.head_size;

        // iterate over all pos, including the current one
        // also calculate the max value (for numerical stability) for the softmax step
        var max_val = 0.0;
        for (var t = 0u; t <= params.pos; t++) {
            // get the key vector for this head and at this pos
            let k_offset = t * params.kv_dim + (head_index / params.kv_mul) * params.head_size;

            var score = 0.0;
            for (var i = 0u; i < params.head_size; i++) {
                // calculate the attention score as the dot product of q and k
                score += q[head_size_offset + i] * key_cache[k_offset + i];
            }

            // save the score to the attention buffer
            attn[head_offset + t] = score / head_size_sqrt;

            // softmax max value
            if attn[head_offset + t] > max_val {
                max_val = attn[head_offset + t];
            }
        }


        // softmax the scores to get attention weights, from 0..pos inclusively
        // exp and sum
        var sum = 0.0;
        for (var t = 0u; t <= params.pos; t++) {
            attn[head_offset + t] = exp(attn[head_offset + t] - max_val);
            sum += attn[head_offset + t];
        }

        // normalize
        for (var t = 0u; t <= params.pos; t++) {
            attn[head_offset + t] /= sum;
        }

        // weighted sum of the values, store back into xb
        for (var i = 0u; i < params.head_size; i++) {
            xb[head_size_offset + i] = 0.0;
        }
        for (var t = 0u; t <= params.pos; t++) {
            // get the key vector for this head and at this pos
            let v_offset = t * params.kv_dim + (head_index / params.kv_mul) * params.head_size;
            // get the attention weight for this timestep
            let att = attn[head_offset + t];
            // accumulate the weighted value into xb
            for (var i = 0u; i < params.head_size; i++) {
                xb[head_size_offset + i] += att * value_cache[v_offset + i];
            }
        }
    }
}