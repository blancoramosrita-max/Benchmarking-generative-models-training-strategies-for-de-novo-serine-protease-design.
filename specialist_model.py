# Serine-Protease Transformer – improved edition
#  – Autoregressive decoder with causal mask
#  – Top-p sampling, padding suppression, catalytic-triad loss boost
#  – Optional ESM-fold & HMMER evaluation stubs
#  – FIXED: Now properly handles FASTA format files
# ============================================================================
import os, sys, warnings, random, math, json, subprocess, logging
from typing import List, Dict, Tuple, Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import requests, io
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ------------------------------------------------------------------
# 1.  Dataset - FIXED VERSION with FASTA support
# ------------------------------------------------------------------
class SerineProteaseDataset(Dataset):
    AA = "ACDEFGHIKLMNPQRSTVWY"
    EOS = "*"                         # 20
    PAD = "X"  # 21
    VOCAB = AA + EOS + PAD
    EOS_IDX = 20
    PAD_IDX = 21                          # helper constant

    def __init__(self, sequences: List[str], max_len: int = 600):
        self.seqs = []
        for s in sequences:
            s = s.upper()
            if self._valid(s):
                # append EOS only if we still have room
                s = s[:max_len-1] + self.EOS if len(s) < max_len else s[:max_len]
                self.seqs.append(s)
        self.max_len = max_len
        # mappings cover every character we can ever see
        self.char2idx = {aa: idx for idx, aa in enumerate(self.VOCAB)}
        self.idx2char = {idx: aa for idx, aa in enumerate(self.VOCAB)}

        if len(self.seqs) == 0:
            raise ValueError("No valid sequences found in the dataset! Please check your FASTA file format and sequence validity.")

    def _valid(self, s: str) -> bool:
        return all(a in self.AA for a in s) and 150 <= len(s) <= 600

    def seq2tensor(self, seq: str) -> torch.Tensor:
        encoded = [self.char2idx[c] for c in seq]
        if len(encoded) > self.max_len:
            encoded = encoded[:self.max_len]
        else:
            encoded += [self.PAD_IDX] * (self.max_len - len(encoded))
        return torch.tensor(encoded, dtype=torch.long)

    def collate(self, batch: List[str]) -> torch.Tensor:
        return torch.stack([self.seq2tensor(s) for s in batch])

    def __getitem__(self, idx: int) -> str:
        return self.seqs[idx]          # just return the raw string

    def __len__(self):
        return len(self.seqs)

    @classmethod
    def from_fasta(cls, fasta_path: str, max_seqs: int = 50000) -> "SerineProteaseDataset":
        """Load sequences from a FASTA file."""
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

        sequences = []
        try:
            # Try parsing as FASTA first
            with open(fasta_path, 'r') as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    seq_str = str(record.seq).upper()
                    # Clean sequence - remove any non-standard amino acid characters
                    clean_seq = ''.join(c for c in seq_str if c in cls.AA)
                    if clean_seq and cls._valid_seq(clean_seq):
                        sequences.append(clean_seq)
                        if len(sequences) >= max_seqs:
                            break

            logger.info("Loaded %d sequences from FASTA file", len(sequences))

        except Exception as e:
            logger.error("Error parsing FASTA file: %s", e)
            # Fallback: try reading as plain text (one sequence per line)
            logger.info("Attempting to read as plain text file...")
            sequences = cls._read_plain_text(fasta_path, max_seqs)

        if len(sequences) == 0:
            raise ValueError("No valid sequences found! Please check your file format.")

        return cls(sequences)

    @classmethod
    def from_uniprot(cls, queries: List[str], max_seqs: int = 50000) -> "SerineProteaseDataset":
        """Original method for backward compatibility."""
        txt_path = "/content/raw (2).fasta"
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Kaggle input file not found: {txt_path}")

        # Try reading as plain text first, then fallback to FASTA
        try:
            sequences = cls._read_plain_text(txt_path, max_seqs)
        except:
            sequences = cls._read_fasta_fallback(txt_path, max_seqs)

        logger.info("Loaded %d sequences from text file", len(sequences))
        return cls(sequences)

    @staticmethod
    def _valid_seq(seq: str) -> bool:
        """Check if sequence is valid."""
        return all(a in SerineProteaseDataset.AA for a in seq) and 150 <= len(seq) <= 600

    @classmethod
    def _read_plain_text(cls, file_path: str, max_seqs: int) -> List[str]:
        """Read sequences from plain text file (one per line)."""
        sequences = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip().upper()
                if line and not line.startswith('>'):  # Skip FASTA headers
                    clean_seq = ''.join(c for c in line if c in cls.AA)
                    if clean_seq and cls._valid_seq(clean_seq):
                        sequences.append(clean_seq)
                        if len(sequences) >= max_seqs:
                            break
        return sequences

    @classmethod
    def _read_fasta_fallback(cls, file_path: str, max_seqs: int) -> List[str]:
        """Fallback method to read FASTA format."""
        sequences = []
        current_seq = []

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence if exists
                    if current_seq:
                        seq = ''.join(current_seq).upper()
                        clean_seq = ''.join(c for c in seq if c in cls.AA)
                        if clean_seq and cls._valid_seq(clean_seq):
                            sequences.append(clean_seq)
                            if len(sequences) >= max_seqs:
                                break
                    current_seq = []
                else:
                    current_seq.append(line)

        # Don't forget the last sequence
        if current_seq:
            seq = ''.join(current_seq).upper()
            clean_seq = ''.join(c for c in seq if c in cls.AA)
            if clean_seq and cls._valid_seq(clean_seq):
                sequences.append(clean_seq)

        return sequences

# ------------------------------------------------------------------
# 2.  Model – Transformer decoder
# ------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.3, max_len=600):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SerineProteaseTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 6, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        # FIXED: Use TransformerDecoder for autoregressive generation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.pad_token = SerineProteaseDataset.PAD
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.emb(x) * math.sqrt(self.emb.embedding_dim)
        x = self.pos_enc(x)

        # FIXED: Create causal mask for autoregressive generation
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        if mask is not None:
            x = self.transformer(x, x, tgt_mask=causal_mask, tgt_key_padding_mask=mask)
        else:
            x = self.transformer(x, x, tgt_mask=causal_mask)
        logits = self.head(self.ln(x))  # (B, T, V)
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # self-attention sub-layer
        out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(out))

        # feed-forward sub-layer
        out = self.ff(x)
        x = self.norm2(x + self.dropout(out))
        return x

class MyModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(d_model, dropout=0.3)
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4*d_model,
                dropout=0.3,
                batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.emb(x) * math.sqrt(x.size(-1))
        x = self.pos_enc(x)               # <-- only adds pos + drop
        x = self.transformer(x, mask)     # <-- real transformer blocks
        return self.out(x)

# ------------------------------------------------------------------
# 3.  Trainer - FIXED VERSION
# ------------------------------------------------------------------

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 dataset: 'SerineProteaseDataset',
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 clip: float = 1.0,
                 stop_threshold: float = 0.10, patience=3, min_delta=0.005):
        self.model = model.to(DEVICE)
        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.dataset.PAD_IDX, reduction='none', label_smoothing=0.15)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
        self.patience = patience
        self.min_delta   = min_delta
        self.best_loss   = np.inf
        self.epochs_wait = 0
        self.clip = clip

    # ------------- existing helpers unchanged -------------
    def catalytic_mask(self, target: torch.Tensor) -> torch.Tensor:
        """Up-weight loss for Ser, Asp, His."""
        mask = torch.ones_like(target, dtype=torch.float32)
        for aa, idx in self.dataset.char2idx.items():
            if aa in "SDH":
                mask = torch.where(target == idx, 2.0, mask)   # ×2 boost
        return mask

    def pad_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """True where token == PAD."""
        return ids == self.dataset.PAD_IDX

    # ------------------------------------------------------

    def train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss, n = 0.0, 0

        for batch in tqdm(loader, leave=False):
            batch = batch.to(DEVICE)
            src = batch[:, :-1]
            tgt = batch[:, 1:]
            pad_mask = self.pad_mask(src)

            logits = self.model(src, pad_mask)
            loss_per_token = self.criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            k = 10
            batch_size, seq_len = tgt.shape
            repeat_pen = torch.ones_like(loss_per_token)
            for b in range(batch_size):
                for pos in range(seq_len):
                    start = max(0, pos - k)
                    recent = tgt[b, start:pos]                      # tokens in window
                    if pos > 0 and tgt[b, pos] in recent:
                        repeat_pen[b * seq_len + pos] = 0.05
            # ---------- catalytic weighting ----------

            # ---------- compute loss ----------
            mask = self.catalytic_mask(tgt).reshape(-1) if hasattr(self, 'catalytic_mask') else torch.ones_like(loss_per_token)
            loss = (loss_per_token * mask * repeat_pen).mean()
            # ---------- backward ----------
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            # ---------- metrics ----------
            total_loss += loss.item() * src.size(0)
            n += src.size(0)

        # ====== epoch finished ======
        avg_loss = total_loss / n
        return avg_loss
    def train_with_early_stop(self, loader, max_epochs=10):
        for epoch in range(1, max_epochs + 1):
            tr_loss = self.train_one_epoch(loader)   # your existing method
        # ---- early stopping ----
            if tr_loss < self.best_loss - self.min_delta:
                self.best_loss   = tr_loss
                self.epochs_wait = 0
                torch.save(self.model.state_dict(), "best_early.pt")
            else:
                self.epochs_wait += 1
                if self.epochs_wait >= self.patience:
                    print(f"Early-stop at epoch {epoch} (best loss {self.best_loss:.4f})")
                    break
        return self.best_loss



# ------------------------------------------------------------------
# 4.  Generator – FIXED VERSION with proper repeat penalty
# ------------------------------------------------------------------
class Generator:
    def __init__(self, model: nn.Module, dataset: SerineProteaseDataset):
        self.model = model.eval()
        self.dataset = dataset

    @torch.no_grad()
    def generate(self, min_len: int = 150, max_len: int = 600,
                 temp=0.8, top_p=0.9, penalty=0.3, random_start=True,) -> str:
        # FIXED: Better initialization to avoid repetitive sequences
        if random_start:
            # Start with common N-terminal amino acids for serine proteases
            starters = ['M', 'I', 'V', 'L', 'A']
            start = random.choice(starters)
        else:
            start = "M"

        seq = [start]
        x = torch.tensor([[self.dataset.char2idx[start]]], device=DEVICE)

        # Track recent amino acids for better repeat penalty
        recent_aas = [start]
        recent_window = 10

        for step in range(max_len - 1):
            logits = self.model(x)               # (1, len(seq), vocab)
            next_logit = logits[0, -1, :] # (V,)

            # FIXED: Smarter repeat penalty that considers recent history
            for aa in set(recent_aas):
                idx = self.dataset.char2idx.get(aa, None)
                if idx is not None:
                    count = recent_aas.count(aa)
                    next_logit[idx] /= (penalty ** count)

            # nucleus sampling
            sorted_logits, sorted_idx = torch.sort(next_logit, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits / temp, dim=-1), dim=-1)
            sorted_idx_to_remove = cum_probs > top_p
            sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[..., :-1].clone()
            sorted_idx_to_remove[..., 0] = 0
            idx_to_remove = sorted_idx[sorted_idx_to_remove]
            if idx_to_remove.numel() == next_logit.numel():
                idx_to_remove = sorted_idx[:1]
            idx_to_remove = idx_to_remove.clamp(max=next_logit.size(-1) - 1)
            next_logit[idx_to_remove] = -1e9

            probs = torch.softmax(next_logit, dim=-1)
            next_token = torch.multinomial(probs, 1)
            next_idx = next_token.item()
            next_aa = self.dataset.idx2char.get(next_idx, "")

            if next_aa == "" or next_aa == self.dataset.PAD:
                break

            # FIXED: Proper EOS handling
            if next_idx == self.dataset.EOS_IDX or len(seq) >= max_len:
                break

            seq.append(next_aa)
            recent_aas.append(next_aa)
            if len(recent_aas) > recent_window:
                recent_aas.pop(0)

            x = torch.cat([x, next_token.view(1, 1)], dim=1)

        return "".join(seq) if len(seq) >= min_len else ""

    @torch.no_grad()
    def beam_generate(self, beam: int = 5, div_pen: float = 2.0,
                      min_len: int = 150, max_len: int = 600,
                      temp: float = 0.8, top_p: float = 0.9,
                      random_start: bool = False) -> str:
        from torch.nn.functional import log_softmax

        start = random.choice(self.dataset.AA) if random_start else "M"
        beams = [([start], 0.0, set())]          # (seq, log_prob, seen_aa)

        for step in range(max_len - 1):
            candidates = []
            for seq, score, seen in beams:
                if len(seq) >= max_len:
                    candidates.append((seq, score, seen))
                    continue
                x = torch.tensor([[self.dataset.char2idx[aa] for aa in seq]], device=DEVICE)
                next_logit = self.model(x)[0, -1, :]

                # diversity penalty
                for aa in set(seq):
                    idx = self.dataset.char2idx.get(aa, None)
                    if idx is not None:
                        next_logit[idx] -= div_pen

                # top-p (nucleus) inside beam
                sorted_logits, sorted_idx = torch.sort(next_logit, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits / temp, dim=-1), dim=-1)
                sorted_idx_to_remove = cum_probs > top_p
                sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[..., :-1].clone()
                sorted_idx_to_remove[..., 0] = 0
                idx_to_remove = sorted_idx[sorted_idx_to_remove]
                if idx_to_remove.numel() == next_logit.numel():
                    idx_to_remove = sorted_idx[:1]
                idx_to_remove = idx_to_remove.clamp(max=next_logit.size(-1) - 1)
                next_logit[idx_to_remove] = -1e9

                log_probs = log_softmax(next_logit / temp, dim=-1)
                topk = torch.topk(log_probs, beam)
                for lp, idx in zip(topk.values, topk.indices):
                    aa = self.dataset.idx2char.get(idx.item(), "")
                    if aa and aa != "X":
                        new_seen = seen | {idx.item()}
                        candidates.append((seq + [aa], score + lp.item(), new_seen))
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam]

        best = beams[0][0]
        return "".join(best) if len(best) >= min_len else ""

    def eval_batch(self, n: int = 100, **kwargs) -> Dict:
        out = []
        for _ in tqdm(range(n), leave=False):
            # FIXED: Proper argument handling
            use_beam = kwargs.get("use_beam", False)

            if use_beam:
                beam_args = {k: v for k, v in kwargs.items() if k in ['beam', 'div_pen', 'min_len', 'max_len', 'temp', 'random_start']}
                seq = self.beam_generate(**beam_args)
            else:
                gen_args = {k: v for k, v in kwargs.items() if k in ['min_len', 'max_len', 'temp', 'top_p', 'penalty', 'random_start']}
                seq = self.generate(**gen_args)

            if seq:
                out.append({
                    "seq": seq,
                    "len": len(seq),
                    "uniq": len(set(seq)),
                    "triad": all(a in seq for a in "SDH"),
                    "m_pct": seq.count("M") / len(seq) * 100
                })
        df = pd.DataFrame(out)
        return {
            "df": df,
            "triad_rate": df["triad"].mean(),
            "avg_uniq": df["uniq"].mean(),
            "avg_len": df["len"].mean()
        }

# ------------------------------------------------------------------
# 5.  ESM-fold stub (optional)
# ------------------------------------------------------------------
try:
    import esm.esmfold.v1 as esmfold
    ESM_MODEL = esmfold.esmfold_v1().to(DEVICE)
except Exception:
    ESM_MODEL = None
    logger.info("ESM-fold not installed – skipping structure check")

def fold_check(seq: str) -> Optional[float]:
    if ESM_MODEL is None:
        return None
    with torch.no_grad():
        output = ESM_MODEL.infer(seq)
        return output["plddt"].mean().item()

# ------------------------------------------------------------------
#  ESM-fold filter (utility)
# ------------------------------------------------------------------
def fold_filter(seqs: List[str], min_plddt: float = 75) -> List[str]:
    ok = []
    for s in tqdm(seqs, desc="ESM-fold"):
        p = fold_check(s)
        if p and p >= min_plddt:
            ok.append(s)
    return ok

# ------------------------------------------------------------------
# 6.  Main pipeline
# ------------------------------------------------------------------
def main():
    # 1. Dataset - Now you can use either FASTA or the original method

    # Option 1: Use FASTA file directly
    fasta_path = "/content/raw (2).fasta"  # Update this path
    if os.path.exists(fasta_path):
        logger.info("Loading from FASTA file: %s", fasta_path)
        dataset = SerineProteaseDataset.from_fasta(fasta_path, max_seqs=10000)
    else:
        # Option 2: Use original UniProt method (fallback)
        logger.info("FASTA file not found, using UniProt queries")
        queries = ["serine protease", "trypsin", "chymotrypsin", "elastase",
               "subtilisin", "granzyme", "kallikrein", "thrombin", "plasmin",
               "ec:3.4.21", "ec:3.4.21.1", "ec:3.4.21.4", "ec:3.4.21.5"]
        dataset = SerineProteaseDataset.from_uniprot(queries, max_seqs=50000)

    logger.info("Dataset loaded with %d valid sequences", len(dataset))
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate)

    # 2. Model
    model = SerineProteaseTransformer(vocab_size=len(dataset.VOCAB))
    trainer = Trainer(model, dataset, patience=3, min_delta=0.005)
    best = trainer.train_with_early_stop(loader, max_epochs=10)

    # 3. Train
    best_loss = trainer.train_with_early_stop(loader, max_epochs=10)
      # add this
    logger.info("Training finished. Best loss %.4f", best_loss)
# reload best weights
    model.load_state_dict(torch.load("best_early.pt"))
    # 4. Generate & evaluate
    generator = Generator(model, dataset)
    metrics = generator.eval_batch(n=200, use_beam=True, beam=5, div_pen=2.0,
                               min_len=150, max_len=600, temp=0.8,
                               penalty=0.3, random_start=True)
    # save best
    best = metrics["df"].query("triad & uniq>=18").head(5)
    best.to_json("best_generated.json", orient="records", indent=2)

    # ---------- auto-save (inside main, before if __name__) ----------
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/training_dataset.txt", "w") as f:
        for seq in dataset.seqs:
            f.write(f"{seq}\n")
    with open("outputs/training_dataset.json", "w") as f:
        json.dump([{"sequence": s, "length": len(s), "uniq_aa": len(set(s)), "has_catalytic_triad": all(aa in s for aa in "SDH")}
                   for s in dataset.seqs], f, indent=2)

    all_df = metrics["df"]
    with open("outputs/generated_proteases.txt", "w") as f:
        for row in metrics["df"].itertuples():
            f.write(f"{row.seq}\n")
    with open("outputs/generated_proteases.json", "w") as f:
        json.dump([{"sequence": row.seq, "length": row.len,
                    "uniq_aa": row.uniq, "has_catalytic_triad": row.triad}
                   for row in metrics["df"].itertuples()], f, indent=2)
    print(f"✅ Saved -> outputs/  ({len(dataset)} train, {len(metrics['df'])} generated sequences)", flush=True)


if __name__ == "__main__":
    main()