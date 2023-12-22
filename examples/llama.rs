use anyhow::Result;
use clap::Parser;
use half::f16;
use memmap2::Mmap;
use rust_tokenizers::{
    tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy},
    vocab::Vocab,
};
use std::{fs::File, io::Write, time::Instant};
use web_llm::{
    context::{Context, ContextBuilder, Instance},
    model::{
        llama::{self, ModelConfig},
        Model, ModelBuilder, StateBuilder,
    },
};

async fn create_context() -> Result<Context> {
    let instance = Instance::new();
    let adapter = instance.adapter(wgpu::PowerPreference::LowPower).await?;
    let context = ContextBuilder::new(adapter)
        .with_default_pipelines()
        .build()
        .await?;
    println!("{:#?}", context.adapter.get_info());
    Ok(context)
}

async fn run(args: Args) -> Result<()> {
    let context = create_context().await?;

    let file = File::open(&args.path)?;
    let data = unsafe { Mmap::map(&file)? };

    let path_string = args.path.to_string_lossy().to_string();
    let config = match path_string {
        // stories-15M
        p if p.contains("15M") => ModelConfig {
            dim: 288,
            hidden_dim: 768,
            n_layers: 6,
            n_heads: 6,
            n_kv_heads: 6,
            vocab_size: 32000,
            seq_len: 256,
        },
        // stories-42M
        p if p.contains("42M") => ModelConfig {
            dim: 512,
            hidden_dim: 1376,
            n_layers: 8,
            n_heads: 8,
            n_kv_heads: 8,
            vocab_size: 32000,
            seq_len: 1024,
        },
        // stories-110M
        p if p.contains("110M") => ModelConfig {
            dim: 768,
            hidden_dim: 2048,
            n_layers: 12,
            n_heads: 12,
            n_kv_heads: 12,
            vocab_size: 32000,
            seq_len: 1024,
        },
        _ => unimplemented!(),
    };

    let tokenizer = LlamaTokenizer::new("tokenizer.model").unwrap();
    let model: Box<dyn Model<ModelState = llama::ModelState>> =
        match args.path.to_string_lossy().to_string() {
            p if p.contains("f16") => Box::new(
                ModelBuilder::new(&context, &config, &data)
                    .build::<llama::TypedModel<f16, f32>>()?,
            ),
            p if p.contains("f32") => Box::new(
                ModelBuilder::new(&context, &config, &data)
                    .build::<llama::TypedModel<f32, f32>>()?,
            ),
            _ => unimplemented!(),
        };

    let mut state = StateBuilder::new(&context).build::<llama::ModelState>();
    println!("Model loaded");

    let prompt = "Once upon a time";
    let encoded_tokens = tokenizer.encode(prompt, true, false);
    let num_prompt_tokens = encoded_tokens.len();
    let mut token = encoded_tokens[0];

    let steps = 128;
    let start = Instant::now();
    while state.pos < steps {
        token = model.run(token, &mut state).await?;
        // if we are still processing the input prompt, force the next prompt token
        state.pos += 1;
        if state.pos < num_prompt_tokens {
            token = encoded_tokens[state.pos]
        }

        let decoded = tokenizer.decode(&[token], false, false);
        match token {
            1 => {
                print!("\n");
                break;
            }
            13 => print!("\n"),
            _ => print!("{}", decoded),
        }
        std::io::stdout().flush()?;
    }
    std::io::stdout().flush()?;

    println!(
        "\nachieved tok/s: {:.2}",
        (steps as f64 / start.elapsed().as_millis() as f64) * 1000.0
    );

    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The path to the file to read
    path: std::path::PathBuf,
}

fn main() {
    let cli = Args::parse();
    pollster::block_on(run(cli)).unwrap();
}

const BOS_TOKEN_ID: i64 = 1;
const EOS_TOKEN_ID: i64 = 2;

pub struct LlamaTokenizer {
    spm: SentencePieceBpeTokenizer,
}

impl LlamaTokenizer {
    pub fn new(tokenizer_path: &str) -> Result<Self> {
        let lower_case = false;
        Ok(
            SentencePieceBpeTokenizer::from_file(tokenizer_path, lower_case)
                .map(|spm| Self { spm })?,
        )
    }

    pub fn encode(&self, text: &str, include_bos: bool, include_eos: bool) -> Vec<i64> {
        let pre = if include_bos {
            vec![BOS_TOKEN_ID]
        } else {
            vec![]
        };

        let post = if include_eos {
            vec![EOS_TOKEN_ID]
        } else {
            vec![]
        };

        let token_ids = self
            .spm
            .encode(
                text,
                None,
                std::usize::MAX,
                &TruncationStrategy::LongestFirst,
                0,
            )
            .token_ids;

        [pre, token_ids, post]
            .into_iter()
            .flat_map(|v| v.into_iter())
            .collect()
    }

    pub fn decode(&self, tokens: &[i64], skip_special_tokens: bool, clean_spaces: bool) -> String {
        self.spm.decode(tokens, skip_special_tokens, clean_spaces)
    }

    pub fn vocab_size(&self, include_special_tokens: bool) -> usize {
        let vocab = self.spm.vocab();
        if include_special_tokens {
            vocab.values().len() + vocab.special_values().len()
        } else {
            vocab.values().len()
        }
    }
}
