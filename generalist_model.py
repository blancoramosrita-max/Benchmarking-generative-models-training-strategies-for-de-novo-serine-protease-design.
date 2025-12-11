#!/usr/bin/env python3
"""
ProtGPT2 Fine-tuning for Serine Protease Generation
Distilled ProtGPT2 checkpoint fine-tuned on serine protease dataset
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from Bio import SeqIO
import argparse
import logging
from typing import List, Dict, Optional
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProteaseDataset(Dataset):
    """Dataset for ProtGPT2 fine-tuning on serine protease sequences"""
    
    def __init__(self, sequences: List[str], tokenizer: GPT2Tokenizer, max_length: int = 600):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Add special tokens if not present
        if '<|start|>' not in tokenizer.additional_special_tokens:
            tokenizer.add_special_tokens({'additional_special_tokens': ['<|start|>', '<|end|>']})
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Format sequence for ProtGPT2
        formatted_seq = f"<|start|>{seq}<|end|>"
        
        encoding = self.tokenizer(
            formatted_seq,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()  # For language modeling
        }

class ProtGPT2FineTuner:
    """Fine-tune distilled ProtGPT2 on serine protease dataset"""
    
    def __init__(self, model_name: str = "littleworth/protgpt2-distilled-small"):
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = None
        
    def load_dataset(self, fasta_path: str, max_seqs: int = 50000) -> List[str]:
        """Load serine protease sequences from FASTA file"""
        sequences = []
        
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
            
        try:
            with open(fasta_path, 'r') as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    seq_str = str(record.seq).upper()
                    # Filter for valid amino acids and length
                    if self._is_valid_sequence(seq_str):
                        sequences.append(seq_str)
                        if len(sequences) >= max_seqs:
                            break
                            
            logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
            
        except Exception as e:
            logger.error(f"Error loading FASTA file: {e}")
            # Fallback to plain text format
            sequences = self._load_plain_text(fasta_path, max_seqs)
            
        return sequences
    
    def _is_valid_sequence(self, seq: str) -> bool:
        """Check if sequence contains only standard amino acids"""
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        return all(aa in valid_aas for aa in seq) and 150 <= len(seq) <= 600
    
    def _load_plain_text(self, file_path: str, max_seqs: int) -> List[str]:
        """Load sequences from plain text file"""
        sequences = []
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip().upper()
                if line and not line.startswith('>'):
                    if all(aa in valid_aas for aa in line) and 150 <= len(line) <= 600:
                        sequences.append(line)
                        if len(sequences) >= max_seqs:
                            break
        
        logger.info(f"Loaded {len(sequences)} sequences from plain text file")
        return sequences
    
    def fine_tune(self, sequences: List[str], output_dir: str = "./protgpt2_finetuned"):
        """Fine-tune ProtGPT2 on serine protease sequences"""
        
        # Prepare dataset
        dataset = ProteaseDataset(sequences, self.tokenizer)
        
        # Load model and resize token embeddings
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=10,
            per_device_train_batch_size=32,
            save_steps=1000,
            save_total_limit=2,
            prediction_loss_only=True,
            learning_rate=5e-3,
            warmup_steps=100,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="no",
            seed=SEED,
            fp16=True if torch.cuda.is_available() else False,
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # We're doing causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        # Fine-tune the model
        logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
        
        return trainer

class ProtGPT2Generator:
    """Generate serine protease sequences using fine-tuned ProtGPT2"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
    @torch.no_grad()
    def generate_sequence(
        self, 
        min_length: int = 150,
        max_length: int = 600,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 950,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> List[str]:
        """Generate serine protease sequences using nucleus sampling"""
        
        # Prepare prompt
        prompt = "<|start|>"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate sequences
        generated_sequences = []
        
        for _ in range(num_return_sequences):
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.encode("<|end|>")[0] if "<|end|>" in self.tokenizer.additional_special_tokens else self.tokenizer.eos_token_id,
                early_stopping=True
            )
            
            # Decode generated sequence
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Clean up the sequence (remove start/end tokens if present)
            generated_text = generated_text.replace("<|start|>", "").replace("<|end|>", "")
            
            # Validate sequence
            if self._is_valid_generated_sequence(generated_text, min_length, max_length):
                generated_sequences.append(generated_text)
        
        return generated_sequences
    
    def _is_valid_generated_sequence(self, seq: str, min_len: int, max_len: int) -> bool:
        """Check if generated sequence is valid"""
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        return (
            min_len <= len(seq) <= max_len and
            all(aa in valid_aas for aa in seq) and
            all(aa in seq for aa in "SDH")  # Must contain catalytic triad
        )
    
    def generate_batch(
        self, 
        n: int = 200, 
        min_length: int = 150,
        max_length: int = 600,
        **generation_kwargs
    ) -> Dict:
        """Generate a batch of sequences and evaluate them"""
        
        sequences = []
        
        for _ in range(n):
            seqs = self.generate_sequence(
                min_length=min_length,
                max_length=max_length,
                **generation_kwargs
            )
            
            if seqs:
                sequences.extend(seqs)
        
        # Evaluate sequences
        results = []
        for seq in sequences:
            results.append({
                "seq": seq,
                "len": len(seq),
                "uniq": len(set(seq)),
                "triad": all(aa in seq for aa in "SDH"),
                "m_pct": seq.count("M") / len(seq) * 100
            })
        
        df = pd.DataFrame(results)
        
        return {
            "df": df,
            "triad_rate": df["triad"].mean(),
            "avg_uniq": df["uniq"].mean(),
            "avg_len": df["len"].mean()
        }

def main():
    """Main pipeline for ProtGPT2 fine-tuning and generation"""
    
    parser = argparse.ArgumentParser(description='Fine-tune ProtGPT2 on serine protease sequences')
    parser.add_argument('--fasta_path', type=str, required=True, help='Path to serine protease FASTA file')
    parser.add_argument('--output_dir', type=str, default='./protgpt2_finetuned', help='Output directory for fine-tuned model')
    parser.add_argument('--generate', action='store_true', help='Generate sequences after fine-tuning')
    parser.add_argument('--n_generate', type=int, default=200, help='Number of sequences to generate')
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    fine_tuner = ProtGPT2FineTuner()
    
    # Load dataset
    logger.info("Loading serine protease sequences...")
    sequences = fine_tuner.load_dataset(args.fasta_path)
    
    # Fine-tune model
    logger.info("Starting ProtGPT2 fine-tuning...")
    trainer = fine_tuner.fine_tune(sequences, args.output_dir)
    
    # Generate sequences if requested
    if args.generate:
        logger.info("Generating sequences with fine-tuned model...")
        generator = ProtGPT2Generator(args.output_dir)
        
        metrics = generator.generate_batch(
            n=args.n_generate,
            min_length=150,
            max_length=600,
            temperature=0.8,
            top_p=0.95,
            top_k=950,
            repetition_penalty=1.2
        )
        
        # Save results
        os.makedirs("outputs", exist_ok=True)
        
        with open("outputs/protgpt2_generated.txt", "w") as f:
            for row in metrics["df"].itertuples():
                f.write(f"{row.seq}\n")
        
        with open("outputs/protgpt2_generated.json", "w") as f:
            json.dump([{
                "sequence": row.seq,
                "length": row.len,
                "uniq_aa": row.uniq,
                "has_catalytic_triad": row.triad
            } for row in metrics["df"].itertuples()], f, indent=2)
        
        logger.info(f"Generated {len(metrics['df'])} sequences")
        logger.info(f"Catalytic triad rate: {metrics['triad_rate']:.3f}")
        logger.info(f"Average unique amino acids: {metrics['avg_uniq']:.2f}")
        logger.info(f"Average length: {metrics['avg_len']:.2f}")
        
        # Save best sequences
        best = metrics["df"].query("triad & uniq>=18").head(5)
        best.to_json("outputs/protgpt2_best.json", orient="records", indent=2)

if __name__ == "__main__":
    main()