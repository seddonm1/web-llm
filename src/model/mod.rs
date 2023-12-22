#![allow(dead_code)]
use std::convert::Infallible;

use anyhow::Result;
use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use web_rwkv_derive::{Deref, DerefMut};

use crate::context::Context;

use self::llama::ModelConfig;

pub mod llama;
pub mod loader;

pub const RESCALE_LAYER: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelVersion {
    V4,
    V5,
    V6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelError {
    InvalidVersion,
    InvalidChunkSize(usize),
    BatchSize(usize, usize),
    BatchOutOfRange { batch: usize, max: usize },
    EmptyInput,
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::InvalidVersion => write!(f, "invalid model version"),
            ModelError::InvalidChunkSize(size) => write!(f, "chunk size {size} not power of 2"),
            ModelError::BatchSize(lhs, rhs) => write!(f, "input batch size {lhs} not match {rhs}"),
            ModelError::BatchOutOfRange { batch, max } => {
                write!(f, "batch {batch} out of range of max {max}")
            }
            ModelError::EmptyInput => write!(f, "input is empty"),
        }
    }
}

impl std::error::Error for ModelError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelInfo {
    pub version: ModelVersion,
    pub num_layer: usize,
    pub num_emb: usize,
    pub num_hidden: usize,
    pub num_vocab: usize,
    pub num_head: usize,
}

pub trait FromBuilder: Sized {
    type Builder<'a>;
    type Error;

    fn from_builder(builder: Self::Builder<'_>) -> Result<Self, Self::Error>;
}

pub trait BackedState: Send {
    fn max_batch(&self) -> usize;
    fn num_layer(&self) -> usize;

    /// Extract the embedding from a given layer of the state.
    fn embed(&self, batch: usize, layer: usize) -> Vec<f32>;
}

#[async_trait]
pub trait ModelState {
    fn context(&self) -> &Context;
}

#[async_trait]
pub trait Model {
    type ModelState: ModelState;

    fn context(&self) -> &Context;
    fn config(&self) -> &ModelConfig;

    /// Run the model for a token as input.
    async fn run(&self, token: i64, state: &mut Self::ModelState) -> Result<i64>;
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Quant {
    /// No quantization.
    #[default]
    None,
    /// Use `Int8` quantization.
    Int8,
    /// Use `NF4` quantization.
    NF4,
}

#[derive(Debug, Clone)]
pub struct Lora {
    pub data: Vec<u8>,
    pub blend: LoraBlend,
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct LoraBlend(pub Vec<LoraBlendPattern>);

impl LoraBlend {
    pub fn full(alpha: f32) -> Self {
        let pattern = LoraBlendPattern::new(r"blocks\.[0-9]+\.([0-9a-zA-Z\.\_]+)", alpha)
            .expect("default blend pattern");
        Self(vec![pattern])
    }
}

impl Default for LoraBlend {
    fn default() -> Self {
        Self::full(1.0)
    }
}

#[derive(Debug, Clone)]
pub struct LoraBlendPattern {
    /// A regex pattern that matches tensors in the model.
    pattern: Regex,
    /// The blend factor.
    alpha: f32,
}

impl LoraBlendPattern {
    #[inline]
    pub fn new(pattern: &str, alpha: f32) -> Result<Self> {
        Ok(Self {
            pattern: Regex::new(pattern)?,
            alpha,
        })
    }

    #[inline]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

pub struct ModelBuilder<'a> {
    context: Context,
    config: ModelConfig,
    data: &'a [u8],
}

impl<'a> ModelBuilder<'a> {
    pub fn new(context: &Context, config: &ModelConfig, data: &'a [u8]) -> Self {
        Self {
            context: context.clone(),
            data,
            config: config.to_owned(),
        }
    }

    pub fn build<M>(self) -> Result<M>
    where
        M: Model + FromBuilder<Builder<'a> = Self, Error = anyhow::Error>,
    {
        M::from_builder(self)
    }
}

/// Create a model state.
pub struct StateBuilder {
    context: Context,
}

impl<'a> StateBuilder {
    pub fn new(context: &Context) -> Self {
        Self {
            context: context.clone(),
        }
    }

    pub fn build<S>(self) -> S
    where
        S: ModelState + FromBuilder<Builder<'a> = Self, Error = Infallible>,
    {
        S::from_builder(self).expect("build model state")
    }
}
