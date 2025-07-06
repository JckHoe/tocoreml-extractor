# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Environment Setup

#### Using uv (recommended)
```bash
# Install dependencies and create virtual environment
uv sync

# Run commands in the virtual environment
uv run python llama_coreml_converter.py
uv run python llama_coreml_inference.py
```

#### Using pip (alternative)
```bash
# Create and activate virtual environment
python -m venv llama_coreml_env
source llama_coreml_env/bin/activate  # On Windows: llama_coreml_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Model Conversion
```bash
# Basic conversion
uv run python llama_coreml_converter.py

# With custom parameters
uv run python llama_coreml_converter.py --model_name meta-llama/Llama-3.2-1B --max_length 512 --output my_model.mlpackage

# Optimized conversion with quantization
uv run python llama_coreml_converter.py --optimized --quantize --output optimized_model.mlpackage
```

### Model Inference
```bash
# Basic inference
uv run python llama_coreml_inference.py

# Custom inference
uv run python llama_coreml_inference.py --model my_model.mlpackage --tokenizer tokenizer --prompt "Your prompt here" --max_tokens 100
```

## Architecture

This is a CoreML conversion and inference toolkit for Llama 3.2 1B models. The codebase consists of two main components:

### Converter (`llama_coreml_converter.py`)
- Loads Llama 3.2 1B from Hugging Face transformers
- Converts PyTorch models to CoreML format using torch.jit.trace
- Supports two conversion modes:
  - Basic: Direct conversion with float16 precision
  - Optimized: Includes post-training quantization (int8) for smaller model size
- Outputs `.mlpackage` files compatible with iOS 16+ and macOS
- Saves tokenizer separately for inference use

### Inference Engine (`llama_coreml_inference.py`)
- `LlamaCoreMLInference` class handles CoreML model loading and text generation
- Implements autoregressive text generation with temperature and top-k sampling
- Manages input padding/truncation to fit model's fixed sequence length
- Uses sliding window approach for longer generation sequences

### Key Implementation Details
- Models are traced with fixed sequence length (default 512 tokens)
- KV cache disabled for static conversion compatibility
- Input tensors are int32, output logits are float32
- Tokenizer padding token set to EOS token if not available
- Generation uses numpy-based sampling with manual softmax implementation