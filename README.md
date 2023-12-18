# Web-LLM

This is an implementation of https://github.com/karpathy/llama2.c based on the excellent https://github.com/cryscan/web-rwkv project in pure Rust and WebGPU.

It is currently very slow and inefficient and is mainly a learning project and demonstration of capability.


## How to use

1. Export a model using `export.py` from the https://github.com/karpathy/llama2.c repository. The `.pt` (checkpoint) files are available from here: https://huggingface.co/karpathy/tinyllamas.

```bash
mkdir -p models/stories15M
python3 export.py --version -1 --dtype fp32 --checkpoint stories15M.pt models/stories15M
```

2. Convert the huggingface `pytorch_model.bin` to `safetensors`:

```bash
python3 convert_safetensors.py --input models/stories15M/pytorch_model.bin --config models/stories15M/config.json --output models/stories15M/model.safetensors
```

3. Run the model:

```bash
cargo run --release --example llama models/stories15M/model.safetensors
```

## Credits
- Based on the https://github.com/cryscan/web-rwkv and uses their design.
