#!/usr/bin/env python3
"""
Serine-protease sequence harvester
– UniProt keyword + MEROPS S01 family
– CD-HIT 90 % redundancy removal
– Optional hhblits against UniProt30 for homologous expansion
Writes FASTA files ready for your Dataset class.
"""
import os
import subprocess
import argparse
import gzip
import json
import shutil
import tempfile
from typing import List
from pathlib import Path
from Bio import SeqIO, Seq, SeqRecord
import requests
import io
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AA = "ACDEFGHIKLMNPQRSTVWY"
MIN_LEN, MAX_LEN = 150, 600
CDHIT_ID = 0.90  # 90 % identity
THREADS = os.cpu_count()

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def fetch_uniprot(query: str, review: bool = False) -> List[SeqRecord.SeqRecord]:
    """Download UniProtKB entries that match `query`."""
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "compressed": "true",
        "format": "fasta",
        "query": f"({query}) AND (length:[{MIN_LEN} TO {MAX_LEN}]",
    }
    if review:
        params["query"] += " AND (reviewed:true)"
    
    logger.info(f"Fetching UniProt sequences with query: {query}")
    r = requests.get(url, params=params, timeout=600)
    r.raise_for_status()
    
    with gzip.open(io.BytesIO(r.content), "rt") as fh:
        records = list(SeqIO.parse(fh, "fasta"))
        logger.info(f"Found {len(records)} sequences")
        return records

def fetch_merops() -> List[SeqRecord.SeqRecord]:
    """MEROPS S01 family (trypsin-like) – pre-packaged FASTA."""
    url = "https://www.ebi.ac.uk/merops/cgi-bin/seq_by_fam?fam=S01&format=fasta"
    
    logger.info("Fetching MEROPS S01 family sequences")
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    
    records = []
    for rec in SeqIO.parse(io.StringIO(r.text), "fasta"):
        if MIN_LEN <= len(rec.seq) <= MAX_LEN and set(rec.seq).issubset(AA):
            records.append(rec)
    
    logger.info(f"Found {len(records)} MEROPS sequences")
    return records

def run_cdhit(infasta: Path, outfasta: Path, identity: float = CDHIT_ID):
    """Run CD-HIT on FASTA."""
    cdhit = shutil.which("cd-hit")
    if not cdhit:
        raise FileNotFoundError("cd-hit not in PATH – install with conda install cd-hit")
    
    logger.info(f"Running CD-HIT with {identity*100:.0f}% identity threshold")
    cmd = [
        cdhit, "-i", str(infasta), "-o", str(outfasta),
        "-c", str(identity), "-T", str(THREADS), "-M", "0"
    ]
    subprocess.run(cmd, check=True)
    
    # Count output sequences
    with open(outfasta, 'r') as f:
        output_count = sum(1 for line in f if line.startswith('>'))
    
    logger.info(f"CD-HIT complete. {output_count} sequences retained")
    return Path(str(outfasta).replace(".fasta", ""))

def hhblits_expand(input_fasta: Path, output_fasta: Path,
                   db: str = "uniclust30_2018_08", e_val: float = 1e-3, iterations: int = 3):
    """Expand input with hhblits hits."""
    hhblits = shutil.which("hhblits")
    if not hhblits:
        raise FileNotFoundError("hhblits not found – install HH-suite3 and download UniProt30")
    
    logger.info(f"Running hhblits with {iterations} iterations against {db}")
    tmpdir = Path(tempfile.mkdtemp())
    a3m = tmpdir / "out.a3m"
    
    cmd = [
        hhblits, "-i", str(input_fasta), "-d", db, "-oa3m", str(a3m),
        "-n", str(iterations), "-e", str(e_val), "-cpu", str(THREADS)
    ]
    subprocess.run(cmd, check=True)
    
    # Convert a3m to FASTA (remove lowercase inserts)
    seqs = []
    for rec in SeqIO.parse(a3m, "fasta"):
        rec.seq = Seq.Seq(str(rec.seq).upper().replace("-", ""))
        if MIN_LEN <= len(rec.seq) <= MAX_LEN and set(rec.seq).issubset(AA):
            rec.id = rec.id.split("|")[0]  # keep unique ID
            rec.description = ""
            seqs.append(rec)
    
    SeqIO.write(seqs, output_fasta, "fasta")
    shutil.rmtree(tmpdir)
    
    logger.info(f"hhblits complete. {len(seqs)} sequences added")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Curate serine protease sequences')
    parser.add_argument('--hhblits', action='store_true', help='Run hhblits expansion (needs HH-suite)')
    parser.add_argument('--reviewed', action='store_true', help='Only use reviewed UniProt entries')
    parser.add_argument('--output_dir', type=str, default='sp_data', help='Output directory')
    parser.add_argument('--max_seqs', type=int, default=50000, help='Maximum sequences to process')
    
    args = parser.parse_args()
    
    # Create output directory
    outdir = Path(args.output_dir)
    outdir.mkdir(exist_ok=True)
    
    # Define file paths
    raw = outdir / "raw.fasta"
    nr90 = outdir / "serine_proteases_nr90.fasta"
    hhb = outdir / "serine_proteases_hhblits.fasta"
    final = outdir / "serine_proteases_final.fasta"
    stats_file = outdir / "dataset_stats.json"
    
    # 1. Download sequences
    logger.info("Downloading UniProt + MEROPS sequences...")
    
    # UniProt queries for serine proteases
    uniprot_queries = [
        'keyword:"Serine protease"',
        'family:"Trypsin"',
        'family:"Subtilisin"',
        'family:"Chymotrypsin"',
        'ec:3.4.21*',
        'ec:3.4.21.1',
        'ec:3.4.21.4',
        'ec:3.4.21.5'
    ]
    
    all_records = []
    
    for query in uniprot_queries:
        try:
            records = fetch_uniprot(query, review=args.reviewed)
            all_records.extend(records)
        except Exception as e:
            logger.warning(f"Failed to fetch UniProt sequences for query '{query}': {e}")
    
    # Add MEROPS sequences
    try:
        merops_records = fetch_merops()
        all_records.extend(merops_records)
    except Exception as e:
        logger.warning(f"Failed to fetch MEROPS sequences: {e}")
    
    # Remove duplicates based on sequence
    seen_sequences = set()
    unique_records = []
    
    for record in all_records:
        seq_str = str(record.seq).upper()
        if seq_str not in seen_sequences and MIN_LEN <= len(seq_str) <= MAX_LEN:
            seen_sequences.add(seq_str)
            # Clean sequence
            clean_seq = ''.join(c for c in seq_str if c in AA)
            if clean_seq and len(clean_seq) >= MIN_LEN:
                record.seq = Seq.Seq(clean_seq)
                unique_records.append(record)
        
        if len(unique_records) >= args.max_seqs:
            break
    
    logger.info(f"Total unique sequences: {len(unique_records)}")
    
    # Save raw sequences
    SeqIO.write(unique_records, raw, "fasta")
    
    # 2. CD-HIT 90% redundancy removal
    logger.info("Running CD-HIT for redundancy removal...")
    
    if len(unique_records) > 100:  # Only run CD-HIT if we have enough sequences
        run_cdhit(raw, nr90)
    else:
        logger.info("Too few sequences for CD-HIT, skipping redundancy removal")
        shutil.copy(raw, nr90)
    
    # 3. Optional hhblits expansion
    if args.hhblits:
        logger.info("Expanding dataset with hhblits...")
        
        try:
            hhblits_expand(nr90, hhb)
            final_input = hhb
        except Exception as e:
            logger.warning(f"hhblits failed: {e}")
            logger.info("Using CD-HIT output as final dataset")
            final_input = nr90
    else:
        logger.info("Skipping hhblits expansion")
        shutil.copy(nr90, hhb)
        final_input = nr90
    
    # Create final dataset
    final_records = list(SeqIO.parse(final_input, "fasta"))
    
    # Remove any remaining invalid sequences
    valid_records = []
    for record in final_records:
        seq_str = str(record.seq).upper()
        if (MIN_LEN <= len(seq_str) <= MAX_LEN and 
            all(aa in AA for aa in seq_str)):
            valid_records.append(record)
    
    logger.info(f"Final dataset contains {len(valid_records)} sequences")
    
    # Save final dataset
    SeqIO.write(valid_records, final, "fasta")
    
    # Generate statistics
    lens = [len(r) for r in valid_records]
    triad = [all(a in str(r.seq) for a in "SDH") for r in valid_records]
    
    stats = {
        "file": str(final.resolve()),
        "n_seqs": len(valid_records),
        "avg_len": round(sum(lens)/len(lens)) if lens else 0,
        "min_len": min(lens) if lens else 0,
        "max_len": max(lens) if lens else 0,
        "catalytic_triad_rate": round(sum(triad)/len(triad), 3) if triad else 0,
        "cdhit_threshold": CDHIT_ID,
        "hhblits_used": args.hhblits,
        "reviewed_only": args.reviewed
    }
    
    # Save statistics
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("Dataset curation complete!")
    logger.info(f"Final dataset saved to: {final}")
    logger.info(f"Statistics saved to: {stats_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET CURATION SUMMARY")
    print("="*60)
    print(f"Total sequences: {stats['n_seqs']}")
    print(f"Average length: {stats['avg_len']} aa")
    print(f"Length range: {stats['min_len']} - {stats['max_len']} aa")
    print(f"Catalytic triad rate: {stats['catalytic_triad_rate']:.1%}")
    print(f"CD-HIT threshold: {stats['cdhit_threshold']*100:.0f}%")
    print(f"hhblits expansion: {'Yes' if stats['hhblits_used'] else 'No'}")
    print(f"Reviewed UniProt only: {'Yes' if stats['reviewed_only'] else 'No'}")
    print("="*60)

if __name__ == "__main__":
    main()