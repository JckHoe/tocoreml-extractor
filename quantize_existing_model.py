#!/usr/bin/env python3
"""
Post-conversion quantization script for existing CoreML models.
This script takes an existing CoreML model and applies quantization to reduce its size.
"""

import argparse
import os
import coremltools as ct
from coremltools.optimize.coreml import linear_quantize_weights, OptimizationConfig, OpLinearQuantizerConfig


def quantize_coreml_model(input_model_path, output_model_path, quantization_bits=8):
    """
    Quantize an existing CoreML model to reduce its size.
    
    Args:
        input_model_path: Path to the input .mlpackage or .mlmodel file
        output_model_path: Path to save the quantized model
        quantization_bits: Number of bits for quantization (4, 8, or 16)
    """
    print(f"Loading model from: {input_model_path}")
    
    # Load the CoreML model
    model = ct.models.MLModel(input_model_path)
    print(f"Original model size: {get_model_size(input_model_path):.2f} MB")
    
    # Apply linear quantization
    print(f"Applying {quantization_bits}-bit quantization...")
    
    if quantization_bits == 4:
        # For 4-bit quantization
        config = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4")
        )
    elif quantization_bits == 8:
        # For 8-bit quantization
        config = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        )
    else:
        # For 16-bit quantization - use default float16
        config = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        )
    
    quantized_model = linear_quantize_weights(model, config)
    
    # Save the quantized model
    print(f"Saving quantized model to: {output_model_path}")
    quantized_model.save(output_model_path)
    
    print(f"Quantized model size: {get_model_size(output_model_path):.2f} MB")
    
    # Calculate compression ratio
    original_size = get_model_size(input_model_path)
    quantized_size = get_model_size(output_model_path)
    compression_ratio = original_size / quantized_size
    
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Size reduction: {((original_size - quantized_size) / original_size) * 100:.1f}%")
    
    return quantized_model


def get_model_size(model_path):
    """Get the size of a model in MB."""
    if os.path.isdir(model_path):
        # For .mlpackage directories
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(model_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # Convert to MB
    else:
        # For .mlmodel files
        return os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB


def main():
    parser = argparse.ArgumentParser(description="Quantize an existing CoreML model")
    parser.add_argument(
        "--input", 
        type=str, 
        default="llama_3_2_1b.mlpackage",
        help="Path to input CoreML model (.mlpackage or .mlmodel)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="llama_3_2_1b_quantized.mlpackage",
        help="Path to save quantized model"
    )
    parser.add_argument(
        "--bits", 
        type=int, 
        choices=[4, 8, 16],
        default=8,
        help="Number of bits for quantization (4, 8, or 16)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Error: Input model not found at {args.input}")
        return
    
    try:
        quantized_model = quantize_coreml_model(
            args.input, 
            args.output, 
            args.bits
        )
        print(f"✅ Successfully quantized model saved to {args.output}")
        
        # Display model information
        print(f"\nModel Information:")
        print(f"Input: {quantized_model.input_description}")
        print(f"Output: {quantized_model.output_description}")
        
    except Exception as e:
        print(f"❌ Error during quantization: {e}")


if __name__ == "__main__":
    main()