# Web-LLM

This is an implementation of https://github.com/karpathy/llama2.c based on the excellent https://github.com/cryscan/web-rwkv project in pure Rust and WebGPU.

It is currently very slow and inefficient and is mainly a learning project and demonstration of capability.


## How to use

1. Export a model using `hf_export` using the conversion script in https://github.com/karpathy/llama2.c repository.
2. Convert the huggingface model to safetensors:

```bash
python3 convert_safetensors.py --input models/stories15M/pytorch_model.bin --output stories15M.f32.safetensors
```

3. Run the model:

```bash
cargo run --release --example llama ./stories15M.f32.safetensors
```

## Credits
- Based on the https://github.com/cryscan/web-rwkv and uses their design.
