#!/usr/bin/env python3
"""
Validation on the NCBI Disease TEST set for a Router-First Multi-Head model.

Input (NCBI test original file):
  /cluster/home/gw/Backend_project/NER/dataset/data/ncbi_disease/original/NCBItestset_corpus.txt

Format example:
  9949209|t|Genetic mapping of the copper toxicosis locus ...
  9949209|a|Abnormal hepatic copper accumulation is recognized ...
  9949209\t23\t39\tcopper toxicosis\tModifier\tOMIM:215600
  ...

This script evaluates DISEASE recognition (span-level) and reports:
  - Micro Precision / Recall / F1 over disease mentions
  - Document-level accuracy (a document is correct if at least one gold disease is detected)
  - Optional token table printouts for qualitative inspection

Matching modes:
  * normalized: casefold + punctuation stripped + whitespace squeezed; match if exact OR containment
  * exact: exact string equality only (stricter)

Usage:
  python Validate_NCBI_Test_RouterFirst_MultiHead.py \
    --model_dir /cluster/home/gw/Backend_project/NER/tuned/intention \
    --backbone  /cluster/home/gw/Backend_project/models/BioLinkBERT-base \
    --ncbi_test /cluster/home/gw/Backend_project/NER/dataset/data/ncbi_disease/original/NCBItestset_corpus.txt \
    --print_samples 5 --match_mode normalized
"""

import os
import io
import json
import argparse
import string
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

# -------------------------- Model -------------------------- #
class RouterFirstMultiHead(nn.Module):
    """Shared encoder + intention head (router) + two token classifiers (disease, gene).
    Only the DISEASE head is used for NCBI evaluation.
    """
    def __init__(self, backbone_name: str, num_dis: int, num_gene: int = 3, hidden_router: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hid = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # Intention (router) head — optional for this evaluation
        self.router = nn.Sequential(
            nn.Linear(hid, hidden_router),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_router, 2),  # [gene, disease]
        )
        # Token classifiers
        self.classifier_dis  = nn.Linear(hid, num_dis)
        self.classifier_gene = nn.Linear(hid, num_gene)

    def forward(self, input_ids, attention_mask):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        H = self.dropout(enc.last_hidden_state)  # (B,T,h)
        CLS = H[:, 0]
        router_logits = self.router(CLS)
        logits_dis  = self.classifier_dis(H)
        logits_gene = self.classifier_gene(H)
        return {"router_logits": router_logits, "logits_dis": logits_dis, "logits_gene": logits_gene}

# ----------------------- Helpers --------------------------- #
SentenceTokens = List[str]

@torch.no_grad()
def decode_tokens_to_word_tags(tokenizer, words: SentenceTokens, logits_dis, input_ids, attention_mask, id2label_dis):
    """Map token predictions back to unique words using word_ids from a fresh encode.
    Returns rows: List[(word, disease_tag)].
    """
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=input_ids.size(1), padding=False)
    word_ids = enc.word_ids()
    pred_dis_tok  = logits_dis.argmax(-1).squeeze(0).tolist()
    attn = attention_mask.squeeze(0).tolist()
    out_rows = []  # (word, dis_tag)
    used = set()
    for ti, wid in enumerate(word_ids):
        if wid is None or attn[ti] == 0 or wid in used:
            continue
        used.add(wid)
        w  = words[wid]
        td = id2label_dis.get(pred_dis_tok[ti],  "O")
        out_rows.append((w, td))
    return out_rows

@torch.no_grad()
def bio_spans(rows):
    """Extract BIO spans from rows [(word, dis_tag)]. Returns list of strings."""
    ents = []
    cur = []
    for w, td in rows:
        if td.startswith("B-"):
            if cur:
                ents.append(" ".join(cur)); cur = []
            cur.append(w)
        elif td.startswith("I-"):
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

def _match_any(pred_list: List[str], gold: str, mode: str = "normalized") -> Tuple[bool, int]:
    """Return (matched?, num_matches). mode in {normalized, exact}.
    normalized: equal OR containment after normalization.
    exact: raw-string exact equality only.
    """
    if mode not in {"normalized", "exact"}:
        mode = "normalized"
    g_raw = gold or ""
    g = _norm(g_raw) if mode == "normalized" else g_raw
    if not g:
        return False, 0
    cnt = 0
    hit = False
    for p in pred_list:
        pn_raw = p or ""
        pn = _norm(pn_raw) if mode == "normalized" else pn_raw
        if not pn:
            continue
        if mode == "normalized":
            if pn == g or pn in g or g in pn:
                hit = True
                cnt += 1
        else:
            if pn_raw == g_raw:
                hit = True
                cnt += 1
    return hit, cnt

# ------------------------ NCBI TEST ------------------------ #
def load_ncbi_split(path: str) -> Dict[str, Dict[str, List[str]]]:
    """Parse an original NCBI split file (dev or test).
    Returns dict pmid -> {"text": full_text, "diseases": [mention strings]}.

    We concatenate title and abstract with a single space. Gold mentions are taken from the file.
    Offsets are NOT used; we evaluate by string match against the text.
    """
    docs: Dict[str, Dict[str, List[str]]] = {}
    titles: Dict[str, str] = {}
    abstracts: Dict[str, str] = {}
    diseases: Dict[str, List[str]] = {}

    with io.open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            if "|t|" in line:
                pmid, _, title = line.partition("|t|")
                titles[pmid] = title
                diseases.setdefault(pmid, [])
            elif "|a|" in line:
                pmid, _, abstract = line.partition("|a|")
                abstracts[pmid] = abstract
                diseases.setdefault(pmid, [])
            else:
                # annotation row: pmid\tstart\tend\tmention\ttype\tmesh
                parts = line.split("\t")
                if len(parts) >= 4:
                    pmid = parts[0]
                    mention = parts[3]
                    diseases.setdefault(pmid, []).append(mention)

    for pmid in set(list(titles.keys()) + list(abstracts.keys()) + list(diseases.keys())):
        t = titles.get(pmid, "")
        a = abstracts.get(pmid, "")
        full = (t + " " + a).strip()
        docs[pmid] = {"text": full, "diseases": diseases.get(pmid, [])}
    return docs

# ------------------------ Eval ----------------------------- #
@torch.no_grad()
def evaluate_ncbi_test(model_dir: str, backbone: str, ncbi_test_path: str, print_samples: int = 5, max_length: int = 256, match_mode: str = "normalized"):
    # Load label space (disease)
    labels_dis_path = os.path.join(model_dir, "labels_dis.json")
    if not os.path.exists(labels_dis_path):
        raise FileNotFoundError(f"Missing labels_dis.json in {model_dir}")
    with io.open(labels_dis_path, "r", encoding="utf-8") as f:
        labels_dis = json.load(f)
    id2label_dis = {i: t for i, t in enumerate(labels_dis)}

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
    model = RouterFirstMultiHead(backbone, num_dis=len(labels_dis), num_gene=3)

    # Load checkpoint
    ckpt_path = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(model_dir, "last_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("No checkpoint found in model_dir")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Load NCBI test
    docs = load_ncbi_split(ncbi_test_path)
    pmids = list(docs.keys())

    # Counters
    TP = FP = FN = 0
    doc_correct = 0

    printed = 0

    for pmid in pmids:
        text = docs[pmid]["text"]
        gold_spans = docs[pmid]["diseases"]

        words = text.split()
        if not words:
            if gold_spans:
                FN += len(gold_spans)
            continue

        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length, padding=False, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])

        rows = decode_tokens_to_word_tags(tokenizer, words, out["logits_dis"], enc["input_ids"], enc["attention_mask"], id2label_dis)
        pred_dises = bio_spans(rows)

        # Span-level matching (micro)
        hit_any = False
        matched_gold = 0
        for g in gold_spans:
            hit, nmatch = _match_any(pred_dises, g, mode=match_mode)
            if hit:
                TP += 1
                matched_gold += 1
                hit_any = True
            else:
                FN += 1
        # Count FPs as predictions that didn't match any gold
        FP += max(0, len(pred_dises) - matched_gold)
        if hit_any:
            doc_correct += 1

        # Print a few samples with token-level table
        if printed < print_samples:
            printed += 1
            print("\n================ SAMPLE ===============")
            print(f"PMID: {pmid}")
            print("--- GOLD DISEASE MENTIONS ---")
            print(gold_spans if gold_spans else "None")
            print("--- PREDICTED DISEASE SPANS ---")
            print(pred_dises if pred_dises else "None")
            print("--- TOKENS word | PRED_TAG ---")
            for w, tag in rows[:200]:  # avoid flooding
                print(f"{w:20s} | {tag:10s}")

    # Metrics
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))
    doc_acc = doc_correct / len(pmids) if pmids else 0.0

    print("\n============= NCBI TEST RESULTS =============")
    print(f"Documents: {len(pmids)}")
    print(f"TP={TP} FP={FP} FN={FN}")
    print(f"Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")
    print(f"Document Accuracy={doc_acc:.4f}  # fraction of docs with at least one correct disease detected")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True,
                    help="Directory containing best_model.pt/last_model.pt and labels_dis.json")
    ap.add_argument("--backbone", type=str, required=True,
                    help="HF path to the encoder backbone (e.g., BioLinkBERT-base)")
    ap.add_argument("--ncbi_test", type=str, required=True,
                    help="Path to NCBItestset_corpus.txt")
    ap.add_argument("--print_samples", type=int, default=5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--match_mode", type=str, default="normalized", choices=["normalized", "exact"],
                    help="Span matching: normalized (default) or exact")
    args = ap.parse_args()

    evaluate_ncbi_test(
        model_dir=args.model_dir,
        backbone=args.backbone,
        ncbi_test_path=args.ncbi_test,
        print_samples=args.print_samples,
        max_length=args.max_length,
        match_mode=args.match_mode,
    )

if __name__ == "__main__":
    main()
