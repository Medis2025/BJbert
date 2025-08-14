#!/usr/bin/env python3
"""
Validation script for the router-first multi-head model.

- Loads the **best model** from: /cluster/home/gw/Backend_project/NER/tuned/intention
- Prints router intention probs [p_gene, p_disease]
- Prints token-level NER tags for both Gene and Disease heads
- Randomly samples sentences from:
    /cluster/home/gw/Backend_project/NER/dataset/data/merged_data1.translated.annotated.csv

Usage:
  python Validate_RouterFirst_Intention_and_NER.py \
    --model_dir /cluster/home/gw/Backend_project/NER/tuned/intention \
    --backbone /cluster/home/gw/Backend_project/models/BioLinkBERT-base \
    --annot_csv /cluster/home/gw/Backend_project/NER/dataset/data/merged_data1.translated.annotated.csv \
    --num 8 --seed 0
"""

import os
import io
import json
import random
from typing import List, Tuple, Dict, Optional

import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

# -------------------------- Model -------------------------- #
class RouterFirstMultiHead(nn.Module):
    """Shared encoder + intention head (router) + two token classifiers."""
    def __init__(self, backbone_name: str, num_dis: int, num_gene: int, hidden_router: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hid = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # Intention (router) head — small MLP → 2 outputs
        self.router = nn.Sequential(
            nn.Linear(hid, hidden_router),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_router, 2),
        )
        # Token classifiers
        self.classifier_dis  = nn.Linear(hid, num_dis)
        self.classifier_gene = nn.Linear(hid, num_gene)

    def forward(self, input_ids, attention_mask):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        H = self.dropout(enc.last_hidden_state)  # (B,T,h)
        CLS = H[:, 0]                             # [CLS]
        router_logits = self.router(CLS)
        router_probs  = torch.sigmoid(router_logits)
        logits_dis  = self.classifier_dis(H)
        logits_gene = self.classifier_gene(H)
        return {"router_probs": router_probs, "logits_dis": logits_dis, "logits_gene": logits_gene}

# ----------------------- Helpers --------------------------- #
SentenceTokens = List[str]

@torch.no_grad()
def decode_tokens_to_word_tags(tokenizer, words: SentenceTokens, logits_dis, logits_gene, input_ids, attention_mask, id2label_dis, id2label_gene):
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=input_ids.size(1), padding=False)
    word_ids = enc.word_ids()
    pred_dis_tok  = logits_dis.argmax(-1).squeeze(0).tolist()
    pred_gene_tok = logits_gene.argmax(-1).squeeze(0).tolist()
    attn = attention_mask.squeeze(0).tolist()
    out_rows = []  # (word, gene_tag, dis_tag)
    used = set()
    for ti, wid in enumerate(word_ids):
        if wid is None or attn[ti] == 0 or wid in used:
            continue
        used.add(wid)
        w  = words[wid]
        tg = id2label_gene.get(pred_gene_tok[ti], "O")
        td = id2label_dis.get(pred_dis_tok[ti],  "O")
        out_rows.append((w, tg, td))
    return out_rows

@torch.no_grad()
def bio_spans(rows, which: str = "gene"):
    """Extract BIO spans from rows [(word, gene_tag, dis_tag), ...]."""
    idx = 1 if which == "gene" else 2
    ents = []
    cur = []
    for w, tg, td in rows:
        t = tg if which == "gene" else td
        if t.startswith("B-"):
            if cur:
                ents.append(" ".join(cur)); cur = []
            cur.append(w)
        elif t.startswith("I-"):
            if cur: cur.append(w)
            else:   cur = [w]
        else:
            if cur:
                ents.append(" ".join(cur)); cur = []
    if cur: ents.append(" ".join(cur))
    return ents

# ------------------------ Main ----------------------------- #
@torch.no_grad()
def main(model_dir: str,
         backbone: str,
         annot_csv: str,
         n_samples: int = 8,
         seed: int = 0,
         max_length: int = 256):

    random.seed(seed)

    # Load label spaces
    with open(os.path.join(model_dir, "labels_dis.json"), "r", encoding="utf-8") as f:
        labels_dis = json.load(f)
    with open(os.path.join(model_dir, "labels_gene.json"), "r", encoding="utf-8") as f:
        labels_gene = json.load(f)
    id2label_dis  = {i: t for i, t in enumerate(labels_dis)}
    id2label_gene = {i: t for i, t in enumerate(labels_gene)}

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
    model = RouterFirstMultiHead(backbone, len(labels_dis), len(labels_gene))

    # Load checkpoint (best)
    ckpt_path = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(model_dir, "last_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("No checkpoint found in model_dir")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Load annotated CSV and sample
    df = pd.read_csv(annot_csv)
    if "text_en" not in df.columns:
        raise ValueError("Annotated CSV must contain a 'text_en' column.")
    # Filter reasonably long, non-null sentences
    cand = df[df["text_en"].astype(str).str.len() > 1].sample(n=max(1, n_samples), random_state=seed)

    for idx, row in cand.iterrows():
        text = str(row["text_en"]) 
        words = text.strip().split()
        if not words:
            continue
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length, padding=False, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])

        # Router intention
        probs = out["router_probs"][0].detach().cpu().tolist()  # [p_gene, p_disease]

        # NER rows
        rows = decode_tokens_to_word_tags(tokenizer, words, out["logits_dis"], out["logits_gene"], enc["input_ids"], enc["attention_mask"], id2label_dis, id2label_gene)
        gene_ents = bio_spans(rows, which="gene")
        dis_ents  = bio_spans(rows, which="disease")

        # Print nicely
        print("\n================ SAMPLE ================")
        print(f"Text: {text}")
        print(f"Intention probs [p_gene, p_disease]: [{probs[0]:.3f}, {probs[1]:.3f}]")
        print("--- Tokens (word | GENE | DIS) ---")
        for w, tg, td in rows:
            print(f"{w:20s} | {tg:12s} | {td:12s}")
        print("--- Spans ---")
        print(f"GENE spans   : {gene_ents if gene_ents else 'None'}")
        print(f"DISEASE spans: {dis_ents  if dis_ents  else 'None'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/cluster/home/gw/Backend_project/NER/tuned/intention")
    parser.add_argument("--backbone", type=str, default="/cluster/home/gw/Backend_project/models/BioLinkBERT-base")
    parser.add_argument("--annot_csv", type=str, default="/cluster/home/gw/Backend_project/NER/dataset/data/merged_data1.translated.annotated.csv")
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    main(model_dir=args.model_dir,
         backbone=args.backbone,
         annot_csv=args.annot_csv,
         n_samples=args.num,
         seed=args.seed,
         max_length=args.max_length)
