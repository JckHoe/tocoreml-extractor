#!/usr/bin/env python3
"""
Inference script for CoreML Llama 3.2 1B model
"""

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer
from pathlib import Path
import argparse


class LlamaCoreMLInference:
    def __init__(self, model_path, tokenizer_path):
        """Initialize CoreML Llama inference"""
        
        print(f"Loading CoreML model from: {model_path}")
        self.model = ct.models.MLModel(model_path)
        
        print(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get model input shape
        input_spec = self.model.get_spec().description.input[0]
        self.max_length = input_spec.type.multiArrayType.shape[1]
        print(f"Model max length: {self.max_length}")
    
    def generate_text(self, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
        """Generate text using the CoreML model"""
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Ensure input fits in model
        if input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, -self.max_length:]
            print(f"Warning: Input truncated to {self.max_length} tokens")
        
        # Pad to max length
        if input_ids.shape[1] < self.max_length:
            padding = self.max_length - input_ids.shape[1]
            input_ids = np.pad(input_ids, ((0, 0), (0, padding)), 
                             constant_values=self.tokenizer.pad_token_id)
        
        generated_tokens = input_ids[0].tolist()
        original_length = len(self.tokenizer.encode(prompt))
        
        print(f"Generating {max_new_tokens} tokens...")
        
        for step in range(max_new_tokens):
            # Prepare input for CoreML
            input_dict = {"input_ids": input_ids.astype(np.int32)}
            
            # Run inference
            output = self.model.predict(input_dict)
            logits = output["logits"]
            
            # Get next token logits (last position)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < np.sort(next_token_logits)[-top_k]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            probs = self._softmax(next_token_logits)
            next_token = np.random.choice(len(probs), p=probs)
            
            # Check for EOS token
            if next_token == self.tokenizer.eos_token_id:
                break
            
            # Add to generated sequence
            generated_tokens.append(next_token)
            
            # Update input for next iteration (shift left, add new token)
            input_ids = np.roll(input_ids, -1, axis=1)
            input_ids[0, -1] = next_token
            
            # Print progress
            if step % 10 == 0:
                partial_text = self.tokenizer.decode(generated_tokens[original_length:])
                print(f"Step {step}: {partial_text}")
        
        # Decode final output
        generated_text = self.tokenizer.decode(generated_tokens[original_length:], skip_special_tokens=True)
        return generated_text
    
    def _softmax(self, x):
        """Compute softmax probabilities"""
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)


def main():
    parser = argparse.ArgumentParser(description="Run inference with CoreML Llama 3.2 1B")
    parser.add_argument("--model", default="llama_3_2_1b.mlpackage", help="CoreML model path")
    parser.add_argument("--tokenizer", default="tokenizer", help="Tokenizer path")
    parser.add_argument("--prompt", default="The future of AI is", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("Run the converter script first to create the model.")
        return
    
    # Check if tokenizer exists
    if not Path(args.tokenizer).exists():
        print(f"❌ Tokenizer not found: {args.tokenizer}")
        print("Run the converter script first to create the tokenizer.")
        return
    
    try:
        # Initialize inference
        inference = LlamaCoreMLInference(args.model, args.tokenizer)
        
        # Generate text
        print(f"\n🤖 Prompt: {args.prompt}")
        print("=" * 50)
        
        generated_text = inference.generate_text(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        print(f"\n📝 Generated text:")
        print(generated_text)
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()