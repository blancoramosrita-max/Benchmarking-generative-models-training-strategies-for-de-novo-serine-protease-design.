# Examples and Tutorials

This directory contains example scripts and tutorials for using the serine protease generation pipeline.

## Files

### Quick Start
- **`quick_start.py`**: Basic example demonstrating how to generate sequences with both specialist and generalist models
- **Purpose**: Get started quickly with sequence generation
- **Usage**: `python quick_start.py`

### Advanced Analysis
- **`advanced_analysis.py`**: Comprehensive analysis of generated sequences including:
  - Amino acid composition analysis
  - Catalytic triad position analysis
  - Common motif identification
  - Diversity metrics
  - Visualization of results
- **Purpose**: Detailed comparison of different models
- **Usage**: `python advanced_analysis.py` (requires sequences from quick_start.py)

## Getting Started

### 1. Generate Sequences

First, run the quick start script to generate some example sequences:

```bash
cd examples
python quick_start.py
```

This will:
- Load the curated dataset
- Initialize both models
- Generate 10 sequences from each model (if models are trained)
- Save the sequences to FASTA files

### 2. Analyze Results

After generating sequences, run the advanced analysis:

```bash
python advanced_analysis.py
```

This will:
- Load all sequence datasets
- Perform comprehensive analysis
- Generate visualizations
- Create a summary report

## Expected Output

### Quick Start Output
```
============================================================
SERINE PROTEASE GENERATION - QUICK START
============================================================

1. Loading serine protease dataset...
Loaded 1000 sequences

2. Initializing Specialist Model...
Loading pre-trained specialist model...

3. Generating sequences with Specialist Model...
Generated 10 specialist sequences
Catalytic triad rate: 80.0%
Average length: 287.4
Average unique amino acids: 19.2

4. Initializing Generalist Model...
Loading fine-tuned ProtGPT2 model...

5. Generating sequences with Generalist Model...
Generated 10 generalist sequences
Catalytic triad rate: 70.0%
Average length: 298.1
Average unique amino acids: 18.8

6. Comparing generated sequences...

Sample Specialist Sequences:
  1. Length: 289, Triad: SDH
     MKTLL... (truncated)
  2. Length: 275, Triad: SDH
     MIVLQ... (truncated)
  3. Length: 298, Triad: Missing
     MKTLV... (truncated)

Sample Generalist Sequences:
  1. Length: 301, Triad: SDH
     MKTLV... (truncated)
  2. Length: 295, Triad: SDH
     MIVLL... (truncated)
  3. Length: 298, Triad: Missing
     MKTLA... (truncated)

7. Saving results...
Specialist sequences saved to: outputs/demo_specialist_sequences.fasta
Generalist sequences saved to: outputs/demo_generalist_sequences.fasta

============================================================
QUICK START COMPLETE!
============================================================
```

### Advanced Analysis Output
The advanced analysis will create:
- JSON files with detailed statistics
- PNG files with visualizations
- A summary report with key findings

## Customization

### Modifying Sequence Generation

Edit `quick_start.py` to change generation parameters:

```python
# Specialist model parameters
specialist_results = specialist_generator.eval_batch(
    n=50,  # Generate 50 sequences instead of 10
    use_beam=True,
    beam=5,  # Use beam search with width 5
    min_len=200,  # Minimum length 200
    max_len=500,  # Maximum length 500
    temp=0.9,  # Higher temperature for more diversity
    penalty=0.2,  # Lower repeat penalty
    random_start=True
)
```

### Adding Your Own Analysis

Extend `advanced_analysis.py` with custom analysis functions:

```python
def my_custom_analysis(sequences: List[str]) -> Dict:
    """Perform custom analysis on sequences"""
    results = {}
    # Your analysis code here
    return results

# Add to main analysis pipeline
results['custom'] = my_custom_analysis(sequences)
```

## Jupyter Notebook Version

For interactive analysis, you can convert these scripts to Jupyter notebooks:

```bash
jupyter notebook
```

Then create a new notebook and import the functions:

```python
from examples.quick_start import *
from examples.advanced_analysis import *

# Run interactively
sequences = generate_sequences(...)
analysis_results = analyze_sequences(sequences)
```

## Troubleshooting

### Common Issues

1. **"File not found" errors**
   - Make sure you've run `data_curation.py` first
   - Check that model files exist in the `outputs/` directory

2. **CUDA out of memory**
   - Reduce batch size in generation parameters
   - Use smaller sequences (reduce max_len)
   - Use CPU instead of GPU (slower but more memory)

3. **Slow generation**
   - Reduce the number of sequences to generate
   - Use smaller beam width for beam search
   - Disable random start for faster initialization

### Performance Tips

1. **For faster generation:**
   ```python
   # Use greedy decoding instead of sampling
   temp=0.1
   use_beam=False
   random_start=False
   ```

2. **For better quality:**
   ```python
   # Use larger beam width and lower temperature
   beam=10
   temp=0.6
   penalty=0.5
   ```

3. **For more diversity:**
   ```python
   # Use higher temperature and random start
   temp=1.2
   random_start=True
   penalty=0.1
   ```

## Next Steps

After running these examples, you might want to:

1. **Train models on larger datasets** using the main scripts
2. **Run comprehensive evaluation** with `evaluation.py`
3. **Analyze structural quality** with AlphaFold2/ColabFold
4. **Validate sequences experimentally** if you have access to wet lab facilities
5. **Extend the models** for other protein families

## Support

For questions about these examples:
- Check the main README.md for detailed documentation
- Review the source code comments for parameter explanations
- Open an issue on GitHub if you encounter problems