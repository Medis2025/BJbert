#!/usr/bin/env python3
"""
Extended validation script for the router-first multi-head model.

New feature: Evaluate on a plain-text file containing **negative** sentences (no gene/disease).

For negative evaluation:
- Counts all predicted gene/disease entities as false positives.
- Computes precision, recall, and F1 for both gene and disease detection.
- In ideal case, all counts should be zero predictions (perfect precision & recall).

Usage:
  python /cluster/home/gw/Backend_project/NER/val/txt_ng.py    \
    --model_dir /cluster/home/gw/Backend_project/NER/tuned/intention  \   
    --backbone /cluster/home/gw/Backend_project/models/BioLinkBERT-base \
    --neg_eval /cluster/home/gw/Backend_project/NER/dataset/data/test_phrases2/negative.txt
"""

import os
import io
import json
import random
import string
from typing import List, Tuple, Dict, Optional

import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

class RouterFirstMultiHead(nn.Module):
    def __init__(self, backbone_name: str, num_dis: int, num_gene: int, hidden_router: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hid = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.router = nn.Sequential(
            nn.Linear(hid, hidden_router),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_router, 2),
        )
        self.classifier_dis  = nn.Linear(hid, num_dis)
        self.classifier_gene = nn.Linear(hid, num_gene)

    def forward(self, input_ids, attention_mask):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        H = self.dropout(enc.last_hidden_state)
        CLS = H[:, 0]
        router_logits = self.router(CLS)
        router_probs  = torch.sigmoid(router_logits)
        logits_dis  = self.classifier_dis(H)
        logits_gene = self.classifier_gene(H)
        return {"router_probs": router_probs, "logits_dis": logits_dis, "logits_gene": logits_gene}

SentenceTokens = List[str]

def decode_tokens_to_word_tags(tokenizer, words: SentenceTokens, logits_dis, logits_gene, input_ids, attention_mask, id2label_dis, id2label_gene):
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=input_ids.size(1), padding=False)
    word_ids = enc.word_ids()
    pred_dis_tok  = logits_dis.argmax(-1).squeeze(0).tolist()
    pred_gene_tok = logits_gene.argmax(-1).squeeze(0).tolist()
    attn = attention_mask.squeeze(0).tolist()
    out_rows = []
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

def bio_spans(rows, which: str = "gene"):
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

def eval_negative_file(model, tokenizer, txt_path: str, device: torch.device, id2label_dis, id2label_gene, max_length: int = 256):
    with open(txt_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    FP_gene = FP_dis = 0
    total = len(sentences)

    for sent in sentences:
        words = sent.split()
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length, padding=False, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])

        rows = decode_tokens_to_word_tags(tokenizer, words, out["logits_dis"], out["logits_gene"], enc["input_ids"], enc["attention_mask"], id2label_dis, id2label_gene)
        pred_genes = bio_spans(rows, which="gene")
        pred_dises = bio_spans(rows, which="disease")

        FP_gene += len(pred_genes)
        FP_dis += len(pred_dises)

        print(f"Sentence: {sent}")
        print(f"Pred genes: {pred_genes if pred_genes else 'None'} | Pred diseases: {pred_dises if pred_dises else 'None'}")

    P_gene = 0.0 if FP_gene > 0 else 1.0
    R_gene = 0.0
    F1_gene = 0.0
    P_dis = 0.0 if FP_dis > 0 else 1.0
    R_dis = 0.0
    F1_dis = 0.0

    print("\n============= NEGATIVE SET RESULTS =============")
    print(f"Gene FP={FP_gene} | Precision={P_gene:.4f} Recall={R_gene:.4f} F1={F1_gene:.4f}")
    print(f"Disease FP={FP_dis} | Precision={P_dis:.4f} Recall={R_dis:.4f} F1={F1_dis:.4f}")

def main(model_dir: str,
         backbone: str,
         neg_eval: Optional[str] = None):

    with open(os.path.join(model_dir, "labels_dis.json"), "r", encoding="utf-8") as f:
        labels_dis = json.load(f)
    with open(os.path.join(model_dir, "labels_gene.json"), "r", encoding="utf-8") as f:
        labels_gene = json.load(f)
    id2label_dis  = {i: t for i, t in enumerate(labels_dis)}
    id2label_gene = {i: t for i, t in enumerate(labels_gene)}

    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
    model = RouterFirstMultiHead(backbone, len(labels_dis), len(labels_gene))

    ckpt_path = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(model_dir, "last_model.pt")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    if neg_eval:
        eval_negative_file(model, tokenizer, txt_path=neg_eval, device=device,
                           id2label_dis=id2label_dis, id2label_gene=id2label_gene)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--neg_eval", type=str, default=None)
    args = parser.parse_args()

    main(model_dir=args.model_dir, backbone=args.backbone, neg_eval=args.neg_eval)
