#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
use std::{collections::HashMap, convert::Infallible, fs::File, mem::size_of, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use half::f16;
use serde_json::de::Read;
use web_rwkv_derive::{Deref, DerefMut};
use wgpu::{CommandEncoderDescriptor, ComputePassDescriptor};

use super::{loader::Loader, matrix::Matrix, FromBuilder, ModelBuilder, ModelInfo, StateBuilder};
use crate::{
    context::Context,
    num::Scalar,
    tensor::{
        cache::ResourceCache,
        ops::{TensorCommand, TensorOp, TensorPass},
        shape::Shape,
        DeepClone, ReadBack, ReadWrite, TensorCpu, TensorError, TensorGpu, TensorInit, TensorShape,
    },
};

#[derive(Debug)]
pub struct Model {
    context: Context,
    config: ModelConfig,
    tensor: ModelTensor,

    runtime_cache: ResourceCache<usize, Runtime>,
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
}

#[derive(Debug)]
struct ModelTensor {
    token_embedding: Matrix,
    layers: Vec<ResidualDecoderAttentionBlock>,
    rms_final_weight: RMSNorm,
    classifier: Matrix,
}

#[derive(Debug)]
pub struct ResidualDecoderAttentionBlock {
    attn: MultiHeadSelfAttention,
    ffn: FeedForwardNetwork,
    att_norm: RMSNorm,
    ffn_norm: RMSNorm,
}

#[derive(Debug)]
struct MultiHeadSelfAttention {
    w_q: Matrix,
    w_k: Matrix,
    w_v: Matrix,
    w_o: Matrix,
}

#[derive(Debug)]
pub struct FeedForwardNetwork {
    // w1
    gate: Matrix,
    // w2
    down: Matrix,
    // w3
    up: Matrix,
}

#[derive(Debug)]

struct RMSNorm {
    w: TensorGpu<f32, ReadWrite>,
}

/// Runtime buffers.
#[derive(Debug)]
struct Runtime {
    x: TensorGpu<f32, ReadWrite>,
    xb: TensorGpu<f32, ReadWrite>,
    xb2: TensorGpu<f32, ReadWrite>,

    attn: TensorGpu<f32, ReadWrite>,
    attn_q: TensorGpu<f32, ReadWrite>,

    ffn_gate: TensorGpu<f32, ReadWrite>,
    ffn_silu: TensorGpu<f32, ReadWrite>,
    ffn_up: TensorGpu<f32, ReadWrite>,

    logits: TensorGpu<f32, ReadWrite>,
    argmax: TensorGpu<u32, ReadWrite>,

    key_cache: HashMap<usize, TensorGpu<f32, ReadWrite>>,
    value_cache: HashMap<usize, TensorGpu<f32, ReadWrite>>,

    half_x: TensorGpu<f16, ReadWrite>,
    probe: TensorGpu<f32, ReadWrite>,
}

impl Runtime {
    pub fn new(context: &Context, config: &ModelConfig) -> Self {
        let dim_shape = Shape::new(config.dim, 1, 1, 1);
        let hidden_shape = Shape::new(config.hidden_dim, 1, 1, 1);
        let att_shape = Shape::new(config.n_heads * config.seq_len, 1, 1, 1);
        let logits_shape = Shape::new(config.vocab_size, 1, 1, 1);
        let cache_shape = Shape::new(config.seq_len * config.dim, 1, 1, 1);
        let argmax_shape = Shape::new(1, 1, 1, 1);

        Self {
            x: context.tensor_init(dim_shape),
            xb: context.tensor_init(dim_shape),
            xb2: context.tensor_init(dim_shape),

            attn: context.tensor_init(att_shape),
            attn_q: context.tensor_init(dim_shape),

            ffn_gate: context.tensor_init(hidden_shape),
            ffn_silu: context.tensor_init(hidden_shape),
            ffn_up: context.tensor_init(hidden_shape),

            logits: context.tensor_init(logits_shape),
            argmax: context.tensor_init(argmax_shape),

            key_cache: (0..config.n_layers)
                .map(|layer_index| (layer_index, context.tensor_init(cache_shape)))
                .collect(),
            value_cache: (0..config.n_layers)
                .map(|layer_index| (layer_index, context.tensor_init(cache_shape)))
                .collect(),

            // unused
            half_x: context.tensor_init(dim_shape),
            probe: context.tensor_init(Shape::new(768, 1, 1, 1)),
        }
    }
}

impl Model {
    #[inline]
    fn request_runtime(&self, num_token: usize) -> Arc<Runtime> {
        self.runtime_cache
            .request(num_token, || Runtime::new(&self.context, &self.config))
    }

    fn run_internal(&self, token: i64, state: &mut ModelState) -> Result<i64> {
        let context = &self.context;
        let tensor = &self.tensor;
        let config = &self.config;

        let token = token as usize;
        let buffer = self.request_runtime(0);
        let pos = state.pos();

        let dim = config.dim;
        let seq_len = config.seq_len;
        let n_heads = config.n_heads;
        let kv_dim = (dim * config.n_kv_heads) / config.n_heads;
        let kv_mul = config.n_heads / config.n_kv_heads;
        let head_size = dim / config.n_heads;

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut ops = Vec::with_capacity(64);

        // copy the token embedding for the token to the input buffer
        ops.extend(
            [match &tensor.token_embedding {
                Matrix::Fp32(tensor) => TensorOp::blit(
                    tensor.view(.., token..=token, .., ..)?,
                    buffer.x.view(.., .., .., ..)?,
                )?,
                Matrix::Fp16(tensor) => TensorOp::blit(
                    tensor.view(.., token..=token, .., ..)?,
                    buffer.x.view(.., .., .., ..)?,
                )?,
                _ => unimplemented!(),
            }]
            .into_iter(),
        );

        for (layer_index, layer) in tensor.layers.iter().enumerate() {
            // qkv projection
            ops.extend(
                [
                    // copy buffer to xb as layer_norm happens inplace
                    TensorOp::blit(
                        buffer.x.view(.., .., .., ..)?,
                        buffer.xb.view(.., .., .., ..)?,
                    )?,
                    // attention rmsnorm
                    TensorOp::rms_norm(&layer.att_norm.w, &buffer.xb)?,
                    // qkv matmuls for this position
                    layer.attn.w_q.matmul_vec_op(
                        buffer.half_x.view(.., .., .., ..)?,
                        buffer.xb.view(.., .., .., ..)?,
                        buffer.attn_q.view(.., .., .., ..)?,
                    )?,
                    // key and value point to the kv cache
                    layer.attn.w_k.matmul_vec_op(
                        buffer.half_x.view(.., .., .., ..)?,
                        buffer.xb.view(.., .., .., ..)?,
                        buffer.key_cache[&layer_index].view(
                            pos * dim..(pos + 1) * dim,
                            ..,
                            ..,
                            ..,
                        )?,
                    )?,
                    // if layer_index == 0 {
                    //     TensorOp::blit(
                    //         buffer.key_cache[&layer_index].view(
                    //             pos * dim..(pos + 1) * dim,
                    //             ..,
                    //             ..,
                    //             ..,
                    //         )?,
                    //         buffer.probe.view(.., .., .., ..)?,
                    //     )?
                    // } else {
                    //     TensorOp::NoOp
                    // },
                    layer.attn.w_v.matmul_vec_op(
                        buffer.half_x.view(.., .., .., ..)?,
                        buffer.xb.view(.., .., .., ..)?,
                        buffer.value_cache[&layer_index].view(
                            pos * kv_dim..(pos + 1) * kv_dim,
                            ..,
                            ..,
                            ..,
                        )?,
                    )?,
                ]
                .into_iter(),
            );

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            ops.extend(
                [TensorOp::relative_positional_encoding(
                    buffer.attn_q.view(.., .., .., ..)?,
                    buffer.key_cache[&layer_index].view(
                        pos * kv_dim..(pos + 1) * kv_dim,
                        ..,
                        ..,
                        ..,
                    )?,
                    dim,
                    head_size,
                    pos,
                    kv_dim,
                )?]
                .into_iter(),
            );

            ops.extend(
                [
                    // multi-head attention
                    TensorOp::multihead_attn(
                        buffer.attn_q.view(.., .., .., ..)?,
                        buffer.key_cache[&layer_index].view(.., .., .., ..)?,
                        buffer.value_cache[&layer_index].view(.., .., .., ..)?,
                        buffer.attn.view(.., .., .., ..)?,
                        buffer.xb.view(.., .., .., ..)?,
                        seq_len,
                        kv_dim,
                        kv_mul,
                        n_heads,
                        head_size,
                        pos,
                    )?,
                    // final matmul to get the output of the attention
                    layer.attn.w_o.matmul_vec_op(
                        buffer.half_x.view(.., .., .., ..)?,
                        buffer.xb.view(.., .., .., ..)?,
                        buffer.xb2.view(.., .., .., ..)?,
                    )?,
                    // residual connection back into x
                    TensorOp::add(
                        buffer.xb2.view(.., .., .., ..)?,
                        buffer.x.view(.., .., .., ..)?,
                    )?,
                ]
                .into_iter(),
            );

            // feedforward
            ops.extend(
                [
                    TensorOp::blit(
                        buffer.x.view(.., .., .., ..)?,
                        buffer.xb.view(.., .., .., ..)?,
                    )?,
                    TensorOp::rms_norm(&layer.ffn_norm.w, &buffer.xb)?,
                    layer.ffn.gate.matmul_vec_op(
                        buffer.half_x.view(.., .., .., ..)?,
                        buffer.xb.view(.., .., .., ..)?,
                        buffer.ffn_gate.view(.., .., .., ..)?,
                    )?,
                    TensorOp::silu(&buffer.ffn_gate, &buffer.ffn_silu)?,
                    layer.ffn.up.matmul_vec_op(
                        buffer.half_x.view(.., .., .., ..)?,
                        buffer.xb.view(.., .., .., ..)?,
                        buffer.ffn_up.view(.., .., .., ..)?,
                    )?,
                    TensorOp::elementwise(&buffer.ffn_silu, &buffer.ffn_up)?,
                    layer.ffn.down.matmul_vec_op(
                        buffer.half_x.view(.., .., .., ..)?,
                        buffer.ffn_up.view(.., .., .., ..)?,
                        buffer.xb.view(.., .., .., ..)?,
                    )?,
                    // residual connection back into x
                    TensorOp::add(
                        buffer.xb.view(.., .., .., ..)?,
                        buffer.x.view(.., .., .., ..)?,
                    )?,
                ]
                .into_iter(),
            );
        }

        ops.extend(
            [
                // final rmsnorm
                TensorOp::rms_norm(&tensor.rms_final_weight.w, &buffer.x)?,
                // classifier into logits
                tensor.classifier.matmul_vec_op(
                    buffer.half_x.view(.., .., .., ..)?,
                    buffer.x.view(.., .., .., ..)?,
                    buffer.logits.view(.., .., .., ..)?,
                )?,
                TensorOp::argmax(
                    buffer.logits.view(.., .., .., ..)?,
                    buffer.argmax.view(.., .., .., ..)?,
                )?,
            ]
            .into_iter(),
        );

        let ops = TensorOp::List(ops);
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.execute_tensor_op(&ops);
        drop(pass);

        // // debug get output
        // let probe_map = context.tensor_init(buffer.probe.shape());
        // encoder.copy_tensor(&buffer.probe, &probe_map)?;

        let argmax_map = context.tensor_init(buffer.argmax.shape());
        encoder.copy_tensor(&buffer.argmax, &argmax_map)?;
        context.queue.submit(Some(encoder.finish()));
        let argmax = Vec::from(argmax_map.back());

        // debug get output
        // let probe_host = probe_map.back();
        // let probe_host = Vec::from(probe_host);
        // println!(
        //     "\n{:?}\n{}",
        //     &buffer.probe.shape(),
        //     &probe_host
        //         .iter()
        //         .map(|v| format!("{:.4}", v))
        //         .collect::<Vec<_>>()
        //         .join(" ")
        //         .replace('"', "")
        // );
        // if pos == 16 {
        //     panic!()
        // }

        Ok(argmax[0] as i64)
    }
}

#[derive(Debug)]
struct Output {}

#[derive(Debug, Clone)]
pub struct ModelState {
    context: Context,
    pub pos: usize,
}

impl ModelState {
    fn pos(&self) -> usize {
        self.pos
    }

    fn pos_mut(&mut self) -> &mut usize {
        &mut self.pos
    }
}

impl super::ModelState for ModelState {
    fn context(&self) -> &Context {
        &self.context
    }
}

impl FromBuilder for ModelState {
    type Builder<'a> = StateBuilder;
    type Error = Infallible;

    fn from_builder(builder: Self::Builder<'_>) -> Result<Self, Self::Error> {
        let StateBuilder { context, .. } = builder;

        Ok(Self {
            context: context.clone(),
            pos: 0,
        })
    }
}

impl FromBuilder for Model {
    type Builder<'b> = ModelBuilder<'b>;
    type Error = anyhow::Error;

    fn from_builder(builder: Self::Builder<'_>) -> Result<Self, Self::Error> {
        let ModelBuilder {
            context,
            config,
            data,
            ..
        } = builder;

        let loader = Loader::new(&context, data, vec![])?;

        let layers =
            (0..config.n_layers)
                .map(|layer_idx| {
                    let attn = MultiHeadSelfAttention {
                        w_q: Matrix::Fp32(loader.load_matrix(format!(
                            "model.layers.{layer_idx}.self_attn.q_proj.weight"
                        ))?),
                        w_k: Matrix::Fp32(loader.load_matrix(format!(
                            "model.layers.{layer_idx}.self_attn.k_proj.weight"
                        ))?),
                        w_v: Matrix::Fp32(loader.load_matrix(format!(
                            "model.layers.{layer_idx}.self_attn.v_proj.weight"
                        ))?),
                        w_o: Matrix::Fp32(loader.load_matrix(format!(
                            "model.layers.{layer_idx}.self_attn.o_proj.weight"
                        ))?),
                    };

                    let ffn =
                        FeedForwardNetwork {
                            gate: Matrix::Fp32(loader.load_matrix(format!(
                                "model.layers.{layer_idx}.mlp.gate_proj.weight"
                            ))?),
                            down: Matrix::Fp32(loader.load_matrix(format!(
                                "model.layers.{layer_idx}.mlp.down_proj.weight"
                            ))?),
                            up: Matrix::Fp32(loader.load_matrix(format!(
                                "model.layers.{layer_idx}.mlp.up_proj.weight"
                            ))?),
                        };

                    // init LayerNorm with 0.0 bias
                    let w = loader.load_vector::<f32>(format!(
                        "model.layers.{layer_idx}.input_layernorm.weight"
                    ))?;
                    let att_norm = RMSNorm { w };

                    // init LayerNorm with 0.0 bias
                    let w = loader.load_vector::<f32>(format!(
                        "model.layers.{layer_idx}.post_attention_layernorm.weight"
                    ))?;
                    let ffn_norm = RMSNorm { w };

                    let block = ResidualDecoderAttentionBlock {
                        attn,
                        ffn,
                        att_norm,
                        ffn_norm,
                    };

                    context.queue.submit(None);
                    context.device.poll(wgpu::MaintainBase::Wait);

                    Ok(block)
                })
                .collect::<Result<Vec<_>>>()?;

        let token_embedding =
            Matrix::Fp32(loader.load_matrix(format!("model.embed_tokens.weight"))?);
        let output_norm_w = loader.load_vector::<f32>(format!("model.norm.weight"))?;
        let classifier = Matrix::Fp32(loader.load_matrix(format!("lm_head.weight"))?);

        context.queue.submit(None);
        context.device.poll(wgpu::MaintainBase::Wait);

        let tensor = ModelTensor {
            token_embedding,
            layers,
            rms_final_weight: RMSNorm { w: output_norm_w },
            classifier,
        };

        Ok(Self {
            context,
            config,
            tensor,
            runtime_cache: ResourceCache::new(1),
        })
    }
}

#[async_trait]
impl super::Model for Model {
    type ModelState = ModelState;

    #[inline]
    fn context(&self) -> &Context {
        &self.context
    }

    #[inline]
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    async fn run(&self, token: i64, state: &mut Self::ModelState) -> Result<i64> {
        let mut state = state.clone();
        let logits = self.run_internal(token, &mut state)?;

        Ok(logits)
    }
}
