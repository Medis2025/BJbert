#!/usr/bin/env python3
"""
Interactive validator: type sentences and get word-level NER labels from a
multi-head BioLinkBERT model. Supports using a fine-tuned checkpoint (.pt)
and optionally comparing against the unfine-tuned base model.

Usage examples:

  # Minimal (uses checkpoint if present)
  python UserSentenceValidator.py \
      --backbone /cluster/home/gw/Backend_project/models/BioLinkBERT-base \
      --save_dir /cluster/home/gw/Backend_project/NER/tuned \
      --ckpt best_model.pt

  # One-off input
  python UserSentenceValidator.py --input "BRCA1 mutations cause breast cancer."

  # Compare checkpoint vs base predictions
  python UserSentenceValidator.py --compare_base --input "EGFR L858R is a driver in NSCLC."

Press Ctrl+C to exit the interactive prompt.
"""

import os
import io
import json
import sys
import argparse
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

SentenceTokens = List[str]

# -------------------------- Utilities -------------------------- #

def read_label_space(path: str) -> Optional[List[str]]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def try_load_labels(save_dir: str) -> Dict[str, List[str]]:
    """Try common sources for label spaces.
    Priority: explicit JSON files in save_dir → state dict extras → fallbacks.
    """
    labels_dis = read_label_space(os.path.join(save_dir, "labels_dis.json"))
    labels_gene = read_label_space(os.path.join(save_dir, "labels_gene.json"))

    if labels_dis is not None and labels_gene is not None:
        return {"dis": labels_dis, "gene": labels_gene}

    # If jsons are missing, try to peek state dict if provided
    return {"dis": None, "gene": None}


class MultiHeadTokenClassifier(nn.Module):
    def __init__(self, backbone_name: str, num_dis: int, num_gene: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier_dis = nn.Linear(hidden, num_dis)
        self.classifier_gene = nn.Linear(hidden, num_gene)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(out.last_hidden_state)
        logits_dis = self.classifier_dis(seq)
        logits_gene = self.classifier_gene(seq)
        return {"logits_dis": logits_dis, "logits_gene": logits_gene}


@torch.no_grad()
def decode_word_level(
    tokenizer: AutoTokenizer,
    words: SentenceTokens,
    logits_dis: torch.Tensor,
    logits_gene: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    id2label_dis: Dict[int, str],
    id2label_gene: Dict[int, str],
):
    # Re-encode without tensors to get word_ids mapping for each token
    enc = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=input_ids.size(1),
        padding=False,
    )
    word_ids = enc.word_ids()

    pred_dis_tok = logits_dis.argmax(-1).squeeze(0).tolist()
    pred_gene_tok = logits_gene.argmax(-1).squeeze(0).tolist()
    attn = attention_mask.squeeze(0).tolist()

    word_seen = set()
    out_rows = []  # (word, pred_gene, pred_dis)
    for idx, wid in enumerate(word_ids):
        if wid is None or attn[idx] == 0 or wid in word_seen:
            continue
        word_seen.add(wid)
        w = words[wid]
        pd = id2label_dis.get(pred_dis_tok[idx], "O")
        pg = id2label_gene.get(pred_gene_tok[idx], "O")
        out_rows.append((w, pg, pd))
    return out_rows


def pretty_print(rows, header_note: str = ""):
    if header_note:
        print(header_note)
    print(f"{'WORD':20s} | {'PRED-GENE':12s} | {'PRED-DIS':12s}")
    print("=" * 60)
    for w, pg, pd in rows:
        print(f"{w[:20]:20s} | {pg:12s} | {pd:12s}")
    print("-" * 60)


def pretty_print_compare(rows_ckpt, rows_base, header_note: str = ""):
    # Align by word index (we assume same tokenization by whitespace words)
    n = min(len(rows_ckpt), len(rows_base))
    if header_note:
        print(header_note)
    print(f"{'WORD':16s} | {'CKPT-GENE':10s} | {'BASE-GENE':10s} | {'CKPT-DIS':10s} | {'BASE-DIS':10s}")
    print("=" * 70)
    for i in range(n):
        w = rows_ckpt[i][0][:16]
        pg_c, pd_c = rows_ckpt[i][1], rows_ckpt[i][2]
        pg_b, pd_b = rows_base[i][1], rows_base[i][2]
        print(f"{w:16s} | {pg_c:10s} | {pg_b:10s} | {pd_c:10s} | {pd_b:10s}")
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Interactive sentence validator for multi-head NER.")
    parser.add_argument("--backbone", type=str, required=True,
                        help="Path or HF id to the base transformer (e.g., BioLinkBERT-base).")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory containing labels_*.json (optional).")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint filename inside save_dir or absolute path to .pt. If omitted, runs base only.")
    parser.add_argument("--compare_base", action="store_true",
                        help="If set, also run predictions with the unfine-tuned base model.")
    parser.add_argument("--input", type=str, default=None,
                        help="One-off input sentence. If omitted, starts interactive mode.")
    parser.add_argument("--max_length", type=int, default=None, help="Override max sequence length.")
    args = parser.parse_args()

    # ---- seed for reproducibility (non-critical here) ----
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    model_max = (
        args.max_length
        if args.max_length is not None
        else (tokenizer.model_max_length if tokenizer.model_max_length and tokenizer.model_max_length < 10_000 else 512)
    )

    # ---- labels ----
    labels = try_load_labels(args.save_dir)

    # If checkpoint has embedded labels, try to fetch them
    state = None
    ckpt_path = None
    if args.ckpt:
        ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(args.save_dir, args.ckpt)
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            for k in ("labels_dis", "labels_gene"):
                if labels.get("dis" if k == "labels_dis" else "gene") is None and k in state:
                    labels["dis" if k == "labels_dis" else "gene"] = state[k]

    # Sensible fallbacks if still missing (common BIO tag sets)
    if labels["dis"] is None:
        labels["dis"] = ["O", "B-Disease", "I-Disease"]
    if labels["gene"] is None:
        labels["gene"] = ["O", "B-Gene", "I-Gene"]

    id2label_dis = {i: t for i, t in enumerate(labels["dis"])}
    id2label_gene = {i: t for i, t in enumerate(labels["gene"])}

    # ---- build models ----
    def build_model():
        return MultiHeadTokenClassifier(args.backbone, len(labels["dis"]), len(labels["gene"]))

    model_ckpt = None
    if ckpt_path and state is not None:
        model_ckpt = build_model()
        model_ckpt.load_state_dict(state.get("model_state", state))
        model_ckpt.to(device).eval()

    model_base = None
    if args.compare_base or model_ckpt is None:
        model_base = build_model()
        # leave as random-initialized heads over pre-trained encoder
        model_base.to(device).eval()

    if model_ckpt is None and model_base is None:
        print("Nothing to run: provide --ckpt or --compare_base.")
        sys.exit(1)

    def run_once(text: str, note: str = ""):
        # Very simple whitespace tokenization to get original 'words'
        words = text.strip().split()
        if not words:
            print("(empty input)")
            return
        enc = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=model_max,
            padding=False,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        def predict_rows(m):
            out = m(enc["input_ids"], enc["attention_mask"])
            return decode_word_level(
                tokenizer,
                words,
                out["logits_dis"],
                out["logits_gene"],
                enc["input_ids"],
                enc["attention_mask"],
                id2label_dis,
                id2label_gene,
            )

        rows_ckpt = predict_rows(model_ckpt) if model_ckpt is not None else None
        rows_base = predict_rows(model_base) if model_base is not None else None

        if rows_ckpt is not None and rows_base is not None:
            pretty_print_compare(rows_ckpt, rows_base, header_note=note)
        elif rows_ckpt is not None:
            pretty_print(rows_ckpt, header_note=note or "[Checkpoint]")
        else:
            pretty_print(rows_base, header_note=note or "[Base]")

    # One-shot or interactive
    if args.input is not None:
        run_once(args.input, note="Input: " + args.input)
    else:
        print("\nType a sentence to label (Ctrl+C to exit).\n")
        try:
            while True:
                text = input(">> ")
                if not text.strip():
                    continue
                run_once(text)
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")


if __name__ == "__main__":
    main()
