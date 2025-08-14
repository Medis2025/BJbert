#!/usr/bin/env python3
"""
Validation and JSON evaluation for the router-first multi-head model.

Adds JSON-based eval on pairs (gene, disease, sentence) where the JSON is a dict:
{
  "OPA1": ["Pseudohypoaldosteronism type IIB",
            "Clinical trials targeting the OPA1 pathway show promising results in halting Pseudohypoaldosteronism type IIB progression in animal models."]
  ...
}

What this script does now:
1) Original sample print from annotated CSV (unchanged, optional)
2) NEW: Evaluate on a JSON file and compute TP, FP, FN, precision, recall, F1, and pair accuracy.
   - Gene metrics are computed against the gene key.
   - Disease metrics are computed against the provided disease string.
   - Pair accuracy counts a sample as correct only if BOTH the correct gene and disease are detected in the sentence.

Usage examples:
  python Validate_RouterFirst_Intention_and_NER.py \
    --model_dir /cluster/home/gw/Backend_project/NER/tuned/intention \
    --backbone /cluster/home/gw/Backend_project/models/BioLinkBERT-base \
    --json_eval /cluster/home/gw/Backend_project/NER/dataset/data/test_phrases2/d_g_sentences.json

  # (Optional) keep the original CSV sampling output too
  python Validate_RouterFirst_Intention_and_NER.py \
    --model_dir /cluster/home/gw/Backend_project/NER/tuned/intention \
    --backbone /cluster/home/gw/Backend_project/models/BioLinkBERT-base \
    --annot_csv /cluster/home/gw/Backend_Project/NER/dataset/data/merged_data1.translated.annotated.csv \
    --num 8 --seed 0 --json_eval /path/to/d_g_sentences.json
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

# -------------------------- Model -------------------------- #
class RouterFirstMultiHead(nn.Module):
    """Shared encoder + intention head (router) + two token classifiers."""
    def __init__(self, backbone_name: str, num_dis: int, num_gene: int, hidden_router: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hid = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # Intention (router) head — small MLP → 2 outputs (gene vs disease routing score)
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

# ---------- Text normalization & matching for eval ---------- #
_punct_tbl = str.maketrans({p: " " for p in string.punctuation})

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = s.translate(_punct_tbl)
    s = " ".join(s.split())
    return s

def _match_any(pred_list: List[str], gold: str) -> Tuple[bool, int]:
    """Return (matched?, num_exact_matches). A match is counted if
    any normalized prediction equals the normalized gold OR one contains the other.
    """
    g = _norm(gold)
    if not g:
        return False, 0
    cnt = 0
    hit = False
    for p in pred_list:
        pn = _norm(p)
        if not pn:
            continue
        if pn == g or pn in g or g in pn:
            hit = True
            cnt += 1
    return hit, cnt

# ------------------------ CSV Demo ------------------------- #
@torch.no_grad()
def demo_from_csv(model, tokenizer, annot_csv: str, device: torch.device, id2label_dis, id2label_gene, n_samples: int = 8, seed: int = 0, max_length: int = 256):
    df = pd.read_csv(annot_csv)
    if "text_en" not in df.columns:
        raise ValueError("Annotated CSV must contain a 'text_en' column.")
    cand = df[df["text_en"].astype(str).str.len() > 1].sample(n=max(1, n_samples), random_state=seed)

    for _, row in cand.iterrows():
        text = str(row["text_en"]) 
        words = text.strip().split()
        if not words:
            continue
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length, padding=False, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])

        probs = out["router_probs"][0].detach().cpu().tolist()  # [p_gene, p_disease]
        rows = decode_tokens_to_word_tags(tokenizer, words, out["logits_dis"], out["logits_gene"], enc["input_ids"], enc["attention_mask"], id2label_dis, id2label_gene)
        gene_ents = bio_spans(rows, which="gene")
        dis_ents  = bio_spans(rows, which="disease")

        print("\n================ SAMPLE ================")
        print(f"Text: {text}")
        print(f"Intention probs [p_gene, p_disease]: [{probs[0]:.3f}, {probs[1]:.3f}]")
        print("--- Tokens (word | GENE | DIS) ---")
        for w, tg, td in rows:
            print(f"{w:20s} | {tg:12s} | {td:12s}")
        print("--- Spans ---")
        print(f"GENE spans   : {gene_ents if gene_ents else 'None'}")
        print(f"DISEASE spans: {dis_ents  if dis_ents  else 'None'}")

# ------------------------ JSON Eval ------------------------ #
@torch.no_grad()
def eval_json_pairs(model, tokenizer, json_path: str, device: torch.device, id2label_dis, id2label_gene, max_length: int = 256):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON must be a dict mapping gene -> [disease, sentence]")

    # Micro counters
    TP_gene = FP_gene = FN_gene = 0
    TP_dis  = FP_dis  = FN_dis  = 0
    pair_correct = 0
    total = 0

    print("\n================ JSON EVAL ================")
    for gene_key, payload in data.items():
        if not isinstance(payload, list) or len(payload) < 2:
            continue
        disease_str, sentence = payload[0], payload[1]
        total += 1

        words = str(sentence).strip().split()
        if not words:
            # If sentence is empty, count as FN for both
            FN_gene += 1
            FN_dis  += 1
            continue

        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length, padding=False, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])

        rows = decode_tokens_to_word_tags(tokenizer, words, out["logits_dis"], out["logits_gene"], enc["input_ids"], enc["attention_mask"], id2label_dis, id2label_gene)
        pred_genes = bio_spans(rows, which="gene")
        pred_dises = bio_spans(rows, which="disease")

        # --- Gene matching ---
        hit_gene, nmatch_gene = _match_any(pred_genes, gene_key)
        if hit_gene:
            TP_gene += 1
            # count FPs as all non-matching predictions besides the matched ones
            FP_gene += max(0, len(pred_genes) - nmatch_gene)
        else:
            FN_gene += 1
            FP_gene += len(pred_genes)  # all predictions are wrong in this case

        # --- Disease matching ---
        hit_dis, nmatch_dis = _match_any(pred_dises, disease_str)
        if hit_dis:
            TP_dis += 1
            FP_dis += max(0, len(pred_dises) - nmatch_dis)
        else:
            FN_dis += 1
            FP_dis += len(pred_dises)

        both = hit_gene and hit_dis
        pair_correct += 1 if both else 0

        # Per-sample debug print
        print("\n--- SAMPLE ---")
        print(f"Sentence: {sentence}")
        print(f"GOLD gene: {gene_key} | Pred genes: {pred_genes if pred_genes else 'None'} | Hit: {hit_gene}")
        print(f"GOLD dis : {disease_str} | Pred dise.: {pred_dises if pred_dises else 'None'} | Hit: {hit_dis}")

    def _prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        return prec, rec, f1

    P_gene, R_gene, F1_gene = _prf(TP_gene, FP_gene, FN_gene)
    P_dis,  R_dis,  F1_dis  = _prf(TP_dis,  FP_dis,  FN_dis)

    pair_acc = pair_correct / total if total > 0 else 0.0

    print("\n============= MICRO SCORES =============")
    print("Gene:")
    print(f"  TP={TP_gene} FP={FP_gene} FN={FN_gene}")
    print(f"  Precision={P_gene:.4f} Recall={R_gene:.4f} F1={F1_gene:.4f}")
    print("Disease:")
    print(f"  TP={TP_dis} FP={FP_dis} FN={FN_dis}")
    print(f"  Precision={P_dis:.4f} Recall={R_dis:.4f} F1={F1_dis:.4f}")
    print("Pair:")
    print(f"  Correct pairs={pair_correct}/{total} | Pair Accuracy={pair_acc:.4f}")

# ------------------------ Main ----------------------------- #
@torch.no_grad()
def main(model_dir: str,
         backbone: str,
         annot_csv: Optional[str] = None,
         n_samples: int = 8,
         seed: int = 0,
         max_length: int = 256,
         json_eval: Optional[str] = None):

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

    # Optional CSV sampling demo
    if annot_csv:
        demo_from_csv(model, tokenizer, annot_csv=annot_csv, device=device,
                      id2label_dis=id2label_dis, id2label_gene=id2label_gene,
                      n_samples=n_samples, seed=seed, max_length=max_length)

    # JSON eval
    if json_eval:
        eval_json_pairs(model, tokenizer, json_path=json_eval, device=device,
                        id2label_dis=id2label_dis, id2label_gene=id2label_gene,
                        max_length=max_length)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/cluster/home/gw/Backend_project/NER/tuned/intention")
    parser.add_argument("--backbone", type=str, default="/cluster/home/gw/Backend_project/models/BioLinkBERT-base")
    parser.add_argument("--annot_csv", type=str, default=None)
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--json_eval", type=str, default="/cluster/home/gw/Backend_project/NER/dataset/data/test_phrases2/d_g_sentences.json")
    args = parser.parse_args()

    main(model_dir=args.model_dir,
         backbone=args.backbone,
         annot_csv=args.annot_csv,
         n_samples=args.num,
         seed=args.seed,
         max_length=args.max_length,
         json_eval=args.json_eval)
