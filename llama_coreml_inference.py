#!/usr/bin/env python3

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer
from pathlib import Path
import argparse


class LlamaInference:
    def __init__(self, model_path, tokenizer_path):
        
        print(f"Loading CoreML model from: {model_path}")
        self.model = ct.models.MLModel(model_path)
        
        print(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        spec = self.model.get_spec().description.input[0]
        self.max_length = spec.type.multiArrayType.shape[1]
        print(f"Model max length: {self.max_length}")
    
    def generate_text(self, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
        
        tokens = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if tokens.shape[1] > self.max_length:
            tokens = tokens[:, -self.max_length:]
            print(f"Warning: Input truncated to {self.max_length} tokens")
        
        if tokens.shape[1] < self.max_length:
            padding = self.max_length - tokens.shape[1]
            tokens = np.pad(tokens, ((0, 0), (0, padding)), 
                             constant_values=self.tokenizer.pad_token_id)
        
        generated = tokens[0].tolist()
        orig_len = len(self.tokenizer.encode(prompt))
        
        print(f"Generating {max_new_tokens} tokens...")
        
        for step in range(max_new_tokens):
            input_dict = {"input_ids": tokens.astype(np.int32)}
            
            output = self.model.predict(input_dict)
            logits = output["logits"]
            
            next_logits = logits[0, -1, :] / temperature
            
            if top_k > 0:
                indices_to_remove = next_logits < np.sort(next_logits)[-top_k]
                next_logits[indices_to_remove] = -float('inf')
            
            probs = self._softmax(next_logits)
            next_token = np.random.choice(len(probs), p=probs)
            
            if next_token == self.tokenizer.eos_token_id:
                break
            
            generated.append(next_token)
            
            tokens = np.roll(tokens, -1, axis=1)
            tokens[0, -1] = next_token
            
            if step % 10 == 0:
                partial = self.tokenizer.decode(generated[orig_len:])
                print(f"Step {step}: {partial}")
        
        text = self.tokenizer.decode(generated[orig_len:], skip_special_tokens=True)
        return text
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


def main():
    parser = argparse.ArgumentParser(description="Run inference with CoreML Llama 3.2 1B")
    parser.add_argument("--model", default="llama_3_2_1b.mlpackage", help="CoreML model path")
    parser.add_argument("--tokenizer", default="tokenizer", help="Tokenizer path")
    parser.add_argument("--prompt", default="Hello there", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=30, help="Top-k sampling")
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("Run the converter script first to create the model.")
        return
    
    if not Path(args.tokenizer).exists():
        print(f"❌ Tokenizer not found: {args.tokenizer}")
        print("Run the converter script first to create the tokenizer.")
        return
    
    try:
        inference = LlamaInference(args.model, args.tokenizer)
        
        print(f"\n🤖 Prompt: {args.prompt}")
        print("=" * 50)
        
        result = inference.generate_text(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        print(f"\n📝 Generated text:")
        print(result)
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()
