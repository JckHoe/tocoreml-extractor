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


def create_traced_model(model_name="meta-llama/Llama-3.2-1B", max_length=512):
    
    print(f"Loading model: {model_name}")
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
    
    print("Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapped_model, dummy_input, strict=False)
    
    return traced_model, tokenizer


def convert_to_coreml(traced_model, tokenizer, output_path="llama_3_2_1b.mlpackage", max_length=512):
    
    print("Converting to CoreML...")
    
    input_shape = ct.Shape(shape=(1, max_length))
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_ids", shape=input_shape, dtype=np.int32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL
    )
    
    mlmodel.short_description = "Llama 3.2 1B text generation model"
    mlmodel.author = "Meta (converted to CoreML)"
    mlmodel.license = "Custom"
    mlmodel.version = "1.0"
    
    mlmodel.input_description["input_ids"] = "Input token IDs"
    mlmodel.output_description["logits"] = "Output logits for next token prediction"
    output_path = Path(output_path)
    print(f"Saving CoreML model to: {output_path}")
    mlmodel.save(str(output_path))
    
    return mlmodel


def create_optimized_model(model_name="meta-llama/Llama-3.2-1B", max_length=512, quantize=True):
    
    print(f"Loading model for optimization: {model_name}")
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
    
    if quantize:
        print("Applying quantization...")
        from coremltools.optimize.torch import quantization
        quantizer = quantization.LinearQuantizer(
            global_config=quantization.LinearQuantizerConfig(
                quantization_scheme=quantization.ObserverType.min_max,
                dtype=torch.qint8,
                mode=quantization.QuantizationMode.linear_quantization
            )
        )
        
        wrapped_model = quantizer.prepare(wrapped_model, example_inputs=(dummy_input,))
        
        with torch.no_grad():
            for _ in range(10):
                cal_input = torch.randint(0, tokenizer.vocab_size, (1, max_length), dtype=torch.long)
                wrapped_model(cal_input)
        
        wrapped_model = quantizer.finalize(wrapped_model)
    
    print("Tracing quantized model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapped_model, dummy_input, strict=False)
    
    return traced_model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Convert Llama 3.2 1B to CoreML")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-1B", help="Hugging Face model name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--output", default="llama_3_2_1b.mlpackage", help="Output CoreML model path")
    parser.add_argument("--quantize", action="store_true", help="Apply quantization")
    parser.add_argument("--optimized", action="store_true", help="Create optimized model")
    
    args = parser.parse_args()
    
    try:
        if args.optimized:
            traced_model, tokenizer = create_optimized_model(
                args.model_name, args.max_length, args.quantize
            )
        else:
            traced_model, tokenizer = create_traced_model(args.model_name, args.max_length)
        
        mlmodel = convert_to_coreml(traced_model, tokenizer, args.output, args.max_length)
        
        print(f"✅ Successfully converted {args.model_name} to CoreML!")
        print(f"📁 Model saved to: {args.output}")
        print(f"📊 Model size: {Path(args.output).stat().st_size / (1024*1024):.1f} MB")
        
        tokenizer_path = Path(args.output).parent / "tokenizer"
        tokenizer.save_pretrained(str(tokenizer_path))
        print(f"💾 Tokenizer saved to: {tokenizer_path}")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        raise


if __name__ == "__main__":
    main()