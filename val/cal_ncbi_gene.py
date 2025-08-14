#!/usr/bin/env python3
"""
Validation on the HuNER Gene test set for a Router-First Multi-Head model.

Input (CoNLL format):
  /cluster/home/gw/Backend_project/NER/dataset/data/huner_gene_nlm_gene/
    SciSpacySentenceSplitter_core_sci_sm_0.5.1_SciSpacyTokenizer_core_sci_sm_0.5.1_test.conll

Format example:
  token<TAB>tag

This script evaluates GENE recognition (span-level) and reports:
  - Micro Precision / Recall / F1 over gene mentions (exact match via BIO tags)
  - Token-level accuracy
  - Optional token table printouts for qualitative inspection

Usage:
  python Validate_HuNER_Gene_Test_RouterFirst_MultiHead.py \
    --model_dir /cluster/home/gw/Backend_project/NER/tuned/intention \
    --backbone  /cluster/home/gw/Backend_project/models/BioLinkBERT-base \
    --gene_test /cluster/home/gw/Backend_project/NER/dataset/data/huner_gene_nlm_gene/SciSpacySentenceSplitter_core_sci_sm_0.5.1_SciSpacyTokenizer_core_sci_sm_0.5.1_test.conll \
    --print_samples 5
"""

import os
import io
import json
import argparse
from typing import List, Tuple, Dict

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

# -------------------------- Model -------------------------- #
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
        logits_dis  = self.classifier_dis(H)
        logits_gene = self.classifier_gene(H)
        return {"router_logits": router_logits, "logits_dis": logits_dis, "logits_gene": logits_gene}

# -------------------------- Data --------------------------- #
def load_conll_sentences(path: str) -> List[Tuple[List[str], List[str]]]:
    sentences = []
    tokens, tags = [], []
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            token, tag = parts[0], parts[1]
            tokens.append(token)
            tags.append(tag)
    if tokens:
        sentences.append((tokens, tags))
    return sentences

# ------------------------- Utils --------------------------- #
@torch.no_grad()
def decode_tokens_to_word_tags(tokenizer, words: List[str], logits_gene, input_ids, attention_mask, id2label_gene):
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=input_ids.size(1), padding=False)
    word_ids = enc.word_ids()
    pred_gene_tok = logits_gene.argmax(-1).squeeze(0).tolist()
    attn = attention_mask.squeeze(0).tolist()
    out_rows = []
    used = set()
    for ti, wid in enumerate(word_ids):
        if wid is None or attn[ti] == 0 or wid in used:
            continue
        used.add(wid)
        w = words[wid]
        tg = id2label_gene.get(pred_gene_tok[ti], "O")
        out_rows.append((w, tg))
    return out_rows

def bio_spans(rows: List[Tuple[str, str]]) -> List[str]:
    spans, cur = [], []
    for w, tg in rows:
        if tg.startswith("B-"):
            if cur:
                spans.append(" ".join(cur)); cur = []
            cur.append(w)
        elif tg.startswith("I-"):
            if cur:
                cur.append(w)
            else:
                cur = [w]
        else:
            if cur:
                spans.append(" ".join(cur)); cur = []
    if cur:
        spans.append(" ".join(cur))
    return spans

# ------------------------ Eval ----------------------------- #
@torch.no_grad()
def evaluate_gene_test(model_dir: str, backbone: str, gene_test_path: str, print_samples: int = 5, max_length: int = 256):
    # Load label space (gene)
    labels_gene_path = os.path.join(model_dir, "labels_gene.json")
    if not os.path.exists(labels_gene_path):
        raise FileNotFoundError(f"Missing labels_gene.json in {model_dir}")
    with io.open(labels_gene_path, "r", encoding="utf-8") as f:
        labels_gene = json.load(f)
    id2label_gene = {i: t for i, t in enumerate(labels_gene)}

    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
    model = RouterFirstMultiHead(backbone, num_dis=3, num_gene=len(labels_gene))

    ckpt_path = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(model_dir, "last_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("No checkpoint found in model_dir")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    sentences = load_conll_sentences(gene_test_path)

    TP = FP = FN = 0
    token_correct = token_total = 0
    printed = 0

    for words, gold_tags in sentences:
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length, padding=False, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])

        rows = decode_tokens_to_word_tags(tokenizer, words, out["logits_gene"], enc["input_ids"], enc["attention_mask"], id2label_gene)
        pred_spans = bio_spans(rows)
        gold_spans = bio_spans(list(zip(words, gold_tags)))

        matched_gold = 0
        for g in gold_spans:
            if g in pred_spans:
                TP += 1
                matched_gold += 1
            else:
                FN += 1
        FP += max(0, len(pred_spans) - matched_gold)

        for (_, gt), (_, pt) in zip(list(zip(words, gold_tags)), rows):
            if gt != "O":
                token_total += 1
                if gt == pt:
                    token_correct += 1

        if printed < print_samples:
            printed += 1
            print("\n================ SAMPLE ================")
            print("WORDS:", words)
            print("GOLD TAGS:", gold_tags)
            print("PRED TAGS:", [t for _, t in rows])
            print("GOLD SPANS:", gold_spans)
            print("PRED SPANS:", pred_spans)

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))
    token_acc = token_correct / token_total if token_total > 0 else 0.0

    print("\n============= GENE TEST RESULTS =============")
    print(f"Sentences: {len(sentences)}")
    print(f"TP={TP} FP={FP} FN={FN}")
    print(f"Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")
    print(f"Token Accuracy={token_acc:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--backbone", type=str, required=True)
    ap.add_argument("--gene_test", type=str, required=True)
    ap.add_argument("--print_samples", type=int, default=5)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    evaluate_gene_test(
        model_dir=args.model_dir,
        backbone=args.backbone,
        gene_test_path=args.gene_test,
        print_samples=args.print_samples,
        max_length=args.max_length,
    )

if __name__ == "__main__":
    main()
