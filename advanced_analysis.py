#!/usr/bin/env python3
"""
Advanced analysis example for serine protease sequences
This script demonstrates detailed analysis and comparison of generated sequences
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation import DiversityAnalyzer, HomologyDetector

def calculate_amino_acid_composition(sequences: List[str]) -> Dict[str, float]:
    """Calculate average amino acid composition across sequences"""
    all_counts = Counter()
    total_length = 0
    
    for seq in sequences:
        all_counts.update(seq)
        total_length += len(seq)
    
    # Convert to percentages
    composition = {aa: (count / total_length) * 100 
                   for aa, count in all_counts.items()}
    
    return composition

def analyze_catalytic_triad_positions(sequences: List[str]) -> Dict:
    """Analyze positions of catalytic triad residues"""
    ser_positions = []
    his_positions = []
    asp_positions = []
    
    for seq in sequences:
        # Find all positions of each residue
        ser_pos = [i for i, aa in enumerate(seq) if aa == 'S']
        his_pos = [i for i, aa in enumerate(seq) if aa == 'H']
        asp_pos = [i for i, aa in enumerate(seq) if aa == 'D']
        
        ser_positions.extend(ser_pos)
        his_positions.extend(his_pos)
        asp_positions.extend(asp_pos)
    
    return {
        'serine_positions': ser_positions,
        'histidine_positions': his_positions,
        'aspartate_positions': asp_positions,
        'serine_avg': np.mean(ser_positions) if ser_positions else 0,
        'histidine_avg': np.mean(his_positions) if his_positions else 0,
        'aspartate_avg': np.mean(asp_positions) if asp_positions else 0
    }

def find_common_motifs(sequences: List[str], motif_length: int = 4) -> Dict[str, int]:
    """Find common amino acid motifs in sequences"""
    all_motifs = Counter()
    
    for seq in sequences:
        for i in range(len(seq) - motif_length + 1):
            motif = seq[i:i+motif_length]
            all_motifs[motif] += 1
    
    # Filter for motifs that appear in at least 10% of sequences
    min_count = len(sequences) * 0.1
    common_motifs = {motif: count for motif, count in all_motifs.items() 
                     if count >= min_count}
    
    return common_motifs

def plot_length_distribution(specialist_lengths: List[int], 
                            generalist_lengths: List[int], 
                            natural_lengths: List[int],
                            output_file: Path):
    """Plot sequence length distributions"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Specialist
    axes[0].hist(specialist_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_title('Specialist Model')
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(np.mean(specialist_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(specialist_lengths):.0f}')
    axes[0].legend()
    
    # Generalist
    axes[1].hist(generalist_lengths, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].set_title('Generalist Model')
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(np.mean(generalist_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(generalist_lengths):.0f}')
    axes[1].legend()
    
    # Natural
    axes[2].hist(natural_lengths, bins=30, alpha=0.7, color='salmon', edgecolor='black')
    axes[2].set_title('Natural Sequences')
    axes[2].set_xlabel('Sequence Length')
    axes[2].set_ylabel('Frequency')
    axes[2].axvline(np.mean(natural_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(natural_lengths):.0f}')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_amino_acid_composition(compositions: Dict[str, Dict[str, float]], output_file: Path):
    """Plot amino acid composition comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    datasets = list(compositions.keys())
    amino_acids = list(compositions[datasets[0]].keys())
    
    x = np.arange(len(amino_acids))
    width = 0.25
    
    for i, dataset in enumerate(datasets):
        values = [compositions[dataset].get(aa, 0) for aa in amino_acids]
        ax.bar(x + i*width, values, width, label=dataset)
    
    ax.set_xlabel('Amino Acid')
    ax.set_ylabel('Frequency (%)')
    ax.set_title('Amino Acid Composition Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(amino_acids)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_catalytic_triad_analysis(triad_analysis: Dict[str, Dict], output_file: Path):
    """Plot catalytic triad position analysis"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = list(triad_analysis.keys())
    
    for i, dataset in enumerate(datasets):
        data = triad_analysis[dataset]
        
        # Create histogram of positions
        axes[i].hist(data['serine_positions'], bins=30, alpha=0.5, label='Serine', color='blue')
        axes[i].hist(data['histidine_positions'], bins=30, alpha=0.5, label='Histidine', color='green')
        axes[i].hist(data['aspartate_positions'], bins=30, alpha=0.5, label='Aspartate', color='red')
        
        axes[i].set_title(f'{dataset.title()} Model')
        axes[i].set_xlabel('Position in Sequence')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        
        # Add vertical lines for average positions
        axes[i].axvline(data['serine_avg'], color='blue', linestyle='--', alpha=0.8)
        axes[i].axvline(data['histidine_avg'], color='green', linestyle='--', alpha=0.8)
        axes[i].axvline(data['aspartate_avg'], color='red', linestyle='--', alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("="*60)
    print("ADVANCED ANALYSIS OF SERINE PROTEASE SEQUENCES")
    print("="*60)
    
    # Configuration
    outputs_dir = Path("outputs")
    analysis_dir = outputs_dir / "advanced_analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Load sequences
    print("\n1. Loading sequence datasets...")
    
    def load_sequences(fasta_file: Path) -> List[str]:
        """Load sequences from FASTA file"""
        if not fasta_file.exists():
            print(f"File not found: {fasta_file}")
            return []
        
        sequences = []
        with open(fasta_file, 'r') as f:
            current_seq = ""
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                    current_seq = ""
                else:
                    current_seq += line
            if current_seq:
                sequences.append(current_seq)
        
        return sequences
    
    # Load all datasets
    specialist_file = outputs_dir / "demo_specialist_sequences.fasta"
    generalist_file = outputs_dir / "demo_generalist_sequences.fasta"
    natural_file = outputs_dir / "natural_sequences.fasta"  # You need to provide this
    
    specialist_seqs = load_sequences(specialist_file)
    generalist_seqs = load_sequences(generalist_file)
    natural_seqs = load_sequences(natural_file)
    
    print(f"Loaded {len(specialist_seqs)} specialist sequences")
    print(f"Loaded {len(generalist_seqs)} generalist sequences")
    print(f"Loaded {len(natural_seqs)} natural sequences")
    
    if not specialist_seqs and not generalist_seqs:
        print("No generated sequences found. Please run quick_start.py first.")
        return
    
    # Analysis results storage
    results = {}
    
    # 2. Basic statistics
    print("\n2. Calculating basic statistics...")
    
    datasets = {
        'Specialist': specialist_seqs,
        'Generalist': generalist_seqs,
        'Natural': natural_seqs
    }
    
    for name, sequences in datasets.items():
        if not sequences:
            continue
            
        lengths = [len(seq) for seq in sequences]
        results[name] = {
            'n_sequences': len(sequences),
            'avg_length': np.mean(lengths),
            'length_std': np.std(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'triad_rate': sum(all(aa in seq for aa in 'SDH') for seq in sequences) / len(sequences)
        }
        
        print(f"{name}: {len(sequences)} sequences, avg length: {np.mean(lengths):.1f}, triad rate: {results[name]['triad_rate']:.1%}")
    
    # 3. Amino acid composition
    print("\n3. Analyzing amino acid composition...")
    
    compositions = {}
    for name, sequences in datasets.items():
        if not sequences:
            continue
        compositions[name] = calculate_amino_acid_composition(sequences)
    
    # Save composition data
    with open(analysis_dir / "amino_acid_composition.json", 'w') as f:
        json.dump(compositions, f, indent=2)
    
    # 4. Catalytic triad analysis
    print("\n4. Analyzing catalytic triad positions...")
    
    triad_analysis = {}
    for name, sequences in datasets.items():
        if not sequences:
            continue
        triad_analysis[name] = analyze_catalytic_triad_positions(sequences)
    
    # Save triad analysis
    with open(analysis_dir / "catalytic_triad_analysis.json", 'w') as f:
        json.dump(triad_analysis, f, indent=2, default=str)
    
    # 5. Common motifs
    print("\n5. Finding common motifs...")
    
    motifs = {}
    for name, sequences in datasets.items():
        if not sequences:
            continue
        motifs[name] = find_common_motifs(sequences, motif_length=4)
    
    # Save motifs
    with open(analysis_dir / "common_motifs.json", 'w') as f:
        json.dump(motifs, f, indent=2)
    
    # 6. Diversity analysis
    print("\n6. Performing diversity analysis...")
    
    diversity_analyzer = DiversityAnalyzer()
    diversity_results = {}
    
    for name, sequences in datasets.items():
        if not sequences:
            continue
        diversity_results[name] = diversity_analyzer.analyze_diversity(sequences)
    
    # Save diversity results
    with open(analysis_dir / "diversity_analysis.json", 'w') as f:
        json.dump(diversity_results, f, indent=2, default=str)
    
    # 7. Create visualizations
    print("\n7. Creating visualizations...")
    
    # Length distribution
    specialist_lengths = [len(seq) for seq in specialist_seqs] if specialist_seqs else []
    generalist_lengths = [len(seq) for seq in generalist_seqs] if generalist_seqs else []
    natural_lengths = [len(seq) for seq in natural_seqs] if natural_seqs else []
    
    if specialist_lengths or generalist_lengths or natural_lengths:
        plot_length_distribution(
            specialist_lengths, generalist_lengths, natural_lengths,
            analysis_dir / "length_distribution.png"
        )
    
    # Amino acid composition
    if compositions:
        plot_amino_acid_composition(compositions, analysis_dir / "amino_acid_composition.png")
    
    # Catalytic triad positions
    if triad_analysis:
        plot_catalytic_triad_analysis(triad_analysis, analysis_dir / "catalytic_triad_positions.png")
    
    # 8. Generate summary report
    print("\n8. Generating summary report...")
    
    summary_report = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'datasets_analyzed': list(results.keys()),
        'basic_statistics': results,
        'common_motifs': {name: dict(list(motifs[name].items())[:10]) for name in motifs},
        'key_findings': []
    }
    
    # Add key findings
    if len(results) > 1:
        summary_report['key_findings'].append("Comparative analysis completed for multiple datasets")
    
    for name, stats in results.items():
        if stats['triad_rate'] > 0.8:
            summary_report['key_findings'].append(f"{name} model shows high catalytic triad conservation ({stats['triad_rate']:.1%})")
    
    # Save summary
    with open(analysis_dir / "summary_report.json", 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print("\n" + "="*60)
    print("ADVANCED ANALYSIS COMPLETE!")
    print("="*60)
    
    print(f"\nResults saved to: {analysis_dir}")
    print("\nFiles generated:")
    print("- amino_acid_composition.json")
    print("- catalytic_triad_analysis.json")
    print("- common_motifs.json")
    print("- diversity_analysis.json")
    print("- length_distribution.png")
    print("- amino_acid_composition.png")
    print("- catalytic_triad_positions.png")
    print("- summary_report.json")
    
    print("\nKey Findings:")
    for finding in summary_report['key_findings']:
        print(f"- {finding}")

if __name__ == "__main__":
    main()