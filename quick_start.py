#!/usr/bin/env python3
"""
Quick start example for serine protease sequence generation
This script demonstrates the basic usage of both specialist and generalist models
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from specialist_model import (
    SerineProteaseTransformer, 
    SerineProteaseDataset, 
    Generator as SpecialistGenerator,
    Trainer
)
from generalist_model import ProtGPT2Generator

def main():
    print("="*60)
    print("SERINE PROTEASE GENERATION - QUICK START")
    print("="*60)
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    data_dir = Path("sp_data")
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # 1. Load Dataset
    print("\n1. Loading serine protease dataset...")
    try:
        dataset = SerineProteaseDataset.from_fasta(data_dir / "serine_proteases_final.fasta", max_seqs=1000)
        print(f"Loaded {len(dataset)} sequences")
    except FileNotFoundError:
        print("Dataset not found. Please run data_curation.py first.")
        return
    
    # 2. Initialize Specialist Model
    print("\n2. Initializing Specialist Model...")
    specialist_model = SerineProteaseTransformer(
        vocab_size=len(dataset.VOCAB),
        d_model=128,
        nhead=4,
        num_layers=6
    ).to(device)
    
    # Try to load pre-trained weights
    model_path = outputs_dir / "best_early.pt"
    if model_path.exists():
        print("Loading pre-trained specialist model...")
        specialist_model.load_state_dict(torch.load(model_path))
    else:
        print("Pre-trained model not found. Please train the model first.")
        print("Skipping specialist model generation...")
        specialist_sequences = []
    
    # 3. Generate Sequences with Specialist Model
    if model_path.exists():
        print("\n3. Generating sequences with Specialist Model...")
        specialist_generator = SpecialistGenerator(specialist_model, dataset)
        
        specialist_results = specialist_generator.eval_batch(
            n=10,  # Generate 10 sequences for demo
            use_beam=True,
            beam=3,
            min_len=150,
            max_len=400,
            temp=0.8,
            penalty=0.3,
            random_start=True
        )
        
        specialist_sequences = specialist_results["df"]["seq"].tolist()
        print(f"Generated {len(specialist_sequences)} specialist sequences")
        
        # Print some statistics
        print(f"Catalytic triad rate: {specialist_results['triad_rate']:.1%}")
        print(f"Average length: {specialist_results['avg_len']:.1f}")
        print(f"Average unique amino acids: {specialist_results['avg_uniq']:.1f}")
    
    # 4. Initialize Generalist Model
    print("\n4. Initializing Generalist Model...")
    generalist_model_path = outputs_dir / "protgpt2_finetuned"
    
    if generalist_model_path.exists():
        print("Loading fine-tuned ProtGPT2 model...")
        try:
            generalist_generator = ProtGPT2Generator(str(generalist_model_path))
            
            # Generate sequences with generalist model
            print("\n5. Generating sequences with Generalist Model...")
            generalist_results = generalist_generator.generate_batch(
                n=10,  # Generate 10 sequences for demo
                min_length=150,
                max_length=400,
                temperature=0.8,
                top_p=0.95,
                top_k=950,
                repetition_penalty=1.2
            )
            
            generalist_sequences = generalist_results["df"]["seq"].tolist()
            print(f"Generated {len(generalist_sequences)} generalist sequences")
            
            # Print some statistics
            print(f"Catalytic triad rate: {generalist_results['triad_rate']:.1%}")
            print(f"Average length: {generalist_results['avg_len']:.1f}")
            print(f"Average unique amino acids: {generalist_results['avg_uniq']:.1f}")
            
        except Exception as e:
            print(f"Error loading generalist model: {e}")
            print("Skipping generalist model generation...")
            generalist_sequences = []
    else:
        print("Fine-tuned ProtGPT2 model not found.")
        print("Skipping generalist model generation...")
        generalist_sequences = []
    
    # 6. Compare Sequences
    print("\n6. Comparing generated sequences...")
    
    if specialist_sequences:
        print("\nSample Specialist Sequences:")
        for i, seq in enumerate(specialist_sequences[:3]):
            print(f"  {i+1}. Length: {len(seq)}, Triad: {'SDH' if all(aa in seq for aa in 'SDH') else 'Missing'}")
            print(f"     {seq[:50]}{'...' if len(seq) > 50 else ''}")
    
    if generalist_sequences:
        print("\nSample Generalist Sequences:")
        for i, seq in enumerate(generalist_sequences[:3]):
            print(f"  {i+1}. Length: {len(seq)}, Triad: {'SDH' if all(aa in seq for aa in 'SDH') else 'Missing'}")
            print(f"     {seq[:50]}{'...' if len(seq) > 50 else ''}")
    
    # 7. Save Results
    print("\n7. Saving results...")
    
    if specialist_sequences:
        specialist_file = outputs_dir / "demo_specialist_sequences.fasta"
        with open(specialist_file, "w") as f:
            for i, seq in enumerate(specialist_sequences):
                f.write(f">specialist_{i+1}\n{seq}\n")
        print(f"Specialist sequences saved to: {specialist_file}")
    
    if generalist_sequences:
        generalist_file = outputs_dir / "demo_generalist_sequences.fasta"
        with open(generalist_file, "w") as f:
            for i, seq in enumerate(generalist_sequences):
                f.write(f">generalist_{i+1}\n{seq}\n")
        print(f"Generalist sequences saved to: {generalist_file}")
    
    print("\n" + "="*60)
    print("QUICK START COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train models with more data for better results")
    print("2. Run comprehensive evaluation with evaluation.py")
    print("3. Analyze structural quality with AlphaFold2")
    print("4. Validate functional properties experimentally")

if __name__ == "__main__":
    main()