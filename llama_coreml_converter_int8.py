#!/usr/bin/env python3

import torch
import coremltools as ct
from transformers import LlamaForCausalLM, AutoTokenizer
import numpy as np
from pathlib import Path
import argparse


class LogitsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids):
        return self.model(input_ids).logits


def create_int4_quantized_model(model_name="meta-llama/Llama-3.2-1B", max_length=512):
    
    print(f"Loading model for int4 quantization: {model_name}")
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        use_cache=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    wrapped_model = LogitsWrapper(model)
    wrapped_model.eval()
    
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, max_length), dtype=torch.long)
    
    print("Applying int4 quantization...")
    from coremltools.optimize.torch import quantization
    
    quantizer = quantization.LinearQuantizer(wrapped_model)
    
    wrapped_model = quantizer.prepare(example_inputs=(dummy_input,))
    
    # Calibration with more samples for better int4 accuracy
    print("Calibrating model...")
    with torch.no_grad():
        for i in range(20):
            cal_input = torch.randint(0, tokenizer.vocab_size, (1, max_length), dtype=torch.long)
            wrapped_model(cal_input)
            if i % 5 == 0:
                print(f"Calibration step {i+1}/20")
    
    wrapped_model = quantizer.finalize(wrapped_model)
    
    print("Tracing int4 quantized model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapped_model, dummy_input, strict=False)
    
    return traced_model, tokenizer


def convert_to_coreml_int4(traced_model, tokenizer, output_path="llama_3_2_1b_int4.mlpackage", max_length=512):
    
    print("Converting to CoreML with int4 quantization...")
    
    input_shape = ct.Shape(shape=(1, max_length))
    
    # Use int4 quantization in CoreML conversion
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_ids", shape=input_shape, dtype=np.int32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL
    )
    
    # Apply post-training quantization to int4
    print("Applying post-training int4 quantization...")
    from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig
    
    # Configure int4 quantization
    op_config = OpPalettizerConfig(
        mode="kmeans",
        nbits=4,
        granularity="per_tensor"
    )
    
    config = OptimizationConfig(
        global_config=op_config
    )
    
    # Apply the quantization
    mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config)
    
    mlmodel.short_description = "Llama 3.2 1B text generation model with int4 quantization"
    mlmodel.author = "Meta (converted to CoreML with int4 quantization)"
    mlmodel.license = "Custom"
    mlmodel.version = "1.0"
    
    mlmodel.input_description["input_ids"] = "Input token IDs"
    mlmodel.output_description["logits"] = "Output logits for next token prediction"
    
    output_path = Path(output_path)
    print(f"Saving int4 quantized CoreML model to: {output_path}")
    mlmodel.save(str(output_path))
    
    return mlmodel


def main():
    parser = argparse.ArgumentParser(description="Convert Llama 3.2 1B to CoreML with int4 quantization")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-1B", help="Hugging Face model name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--output", default="llama_3_2_1b_int4.mlpackage", help="Output CoreML model path")
    
    args = parser.parse_args()
    
    try:
        traced_model, tokenizer = create_int4_quantized_model(args.model_name, args.max_length)
        
        mlmodel = convert_to_coreml_int4(traced_model, tokenizer, args.output, args.max_length)
        
        print(f"✅ Successfully converted {args.model_name} to CoreML with int4 quantization!")
        print(f"📁 Model saved to: {args.output}")
        print(f"📊 Model size: {Path(args.output).stat().st_size / (1024*1024):.1f} MB")
        
        tokenizer_path = Path(args.output).parent / "tokenizer_int4"
        tokenizer.save_pretrained(str(tokenizer_path))
        print(f"💾 Tokenizer saved to: {tokenizer_path}")
        
        print("\n🔧 Usage:")
        print(f"uv run python llama_coreml_inference.py --model {args.output} --tokenizer {tokenizer_path}")
        
    except Exception as e:
        print(f"❌ Error during int4 conversion: {e}")
        raise


if __name__ == "__main__":
    main()