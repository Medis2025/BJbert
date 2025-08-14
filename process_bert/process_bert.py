#!/usr/bin/env python3
"""
Batch NER filter for cleaned CSV using a trained multi-head BioLinkBERT model.

- Loads a multi-head token classifier (Gene head + Disease head)
- Reads the cleaned CSV at INPUT_CSV (default: merged_data1.translated.clean.csv)
- Runs word-level NER on the English text (text_en)
- Extracts BIO entities for Gene and Disease
- Appends boolean flags and entity lists to each row
- Writes a combined annotated CSV, plus two filtered CSVs:
    * gene_filtered.csv: rows where a Gene entity was found; if none found in a row, the row is appended to BOTH gene and disease CSVs (per spec)
    * disease_filtered.csv: rows where a Disease entity was found; plus the "none found" rows

Usage example:

python Batch_NER_Filter_with_Multihead_BioLinkBERT.py \
  --backbone /cluster/home/gw/Backend_project/models/BioLinkBERT-base \
  --save_dir /cluster/home/gw/Backend_project/NER/tuned \
  --ckpt best_model.pt \
  --input_csv /cluster/home/gw/Backend_project/NER/dataset/data/merged_data1.translated.clean.csv \
  --out_dir /cluster/home/gw/Backend_project/NER/dataset/data

"""

import os
import re
import io
import json
import argparse
from typing import List, Tuple, Dict, Optional

import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

SentenceTokens = List[str]

# -------------------------- Utilities -------------------------- #

def read_label_space(path: str) -> Optional[List[str]]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def try_load_labels(save_dir: str, state: Optional[dict]) -> Dict[str, List[str]]:
    """Try common sources for label spaces.
    Priority: explicit JSON files in save_dir → state dict extras → fallbacks.
    """
    labels_dis = read_label_space(os.path.join(save_dir, "labels_dis.json"))
    labels_gene = read_label_space(os.path.join(save_dir, "labels_gene.json"))

    if labels_dis is None and state is not None and "labels_dis" in state:
        labels_dis = state["labels_dis"]
    if labels_gene is None and state is not None and "labels_gene" in state:
        labels_gene = state["labels_gene"]

    # Sensible fallbacks if still missing (common BIO sets)
    if labels_dis is None:
        labels_dis = ["O", "B-Disease", "I-Disease"]
    if labels_gene is None:
        labels_gene = ["O", "B-Gene", "I-Gene"]

    return {"dis": labels_dis, "gene": labels_gene}


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
    """Return list of (word, gene_tag, dis_tag)."""
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


def bio_spans(words_with_tags: List[Tuple[str, str]]) -> List[str]:
    """Extract entity strings from a list of (word, tag) using BIO.
       Returns a list of entity surface strings (joined by spaces).
    """
    ents = []
    cur = []
    for w, t in words_with_tags:
        if t.startswith("B-"):
            if cur:
                ents.append(" ".join(cur))
                cur = []
            cur.append(w)
        elif t.startswith("I-"):
            if cur:
                cur.append(w)
            else:
                # orphan I- -> start new span
                cur = [w]
        else:  # O
            if cur:
                ents.append(" ".join(cur))
                cur = []
    if cur:
        ents.append(" ".join(cur))
    return ents


def predict_sentence(model: nn.Module, tokenizer: AutoTokenizer, text: str, max_len: int, device: torch.device,
                     id2label_gene: Dict[int, str], id2label_dis: Dict[int, str]):
    words = text.strip().split()
    if not words:
        return [], [], []  # rows, gene_ents, dis_ents

    enc = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(enc["input_ids"], enc["attention_mask"])  # logits
    rows = decode_word_level(
        tokenizer,
        words,
        out["logits_dis"],
        out["logits_gene"],
        enc["input_ids"],
        enc["attention_mask"],
        id2label_dis,
        id2label_gene,
    )

    # rows: [(word, gene_tag, dis_tag), ...]
    gene_spans = bio_spans([(w, tg) for (w, tg, _) in rows])
    dis_spans = bio_spans([(w, td) for (w, _, td) in rows])
    return rows, gene_spans, dis_spans


# -------------------------- Main -------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", type=str, required=True,
                    help="HF id or local path for base encoder (e.g., BioLinkBERT-base)")
    ap.add_argument("--save_dir", type=str, required=True,
                    help="Directory containing labels_*.json and the checkpoint file")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Checkpoint filename in save_dir or absolute path to .pt")
    ap.add_argument("--input_csv", type=str, default="/cluster/home/gw/Backend_project/NER/dataset/data/merged_data1.translated.clean.csv")
    ap.add_argument("--out_dir", type=str, default="/cluster/home/gw/Backend_project/NER/dataset/data")
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    model_max = args.max_length

    # load checkpoint
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(args.save_dir, args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    labels = try_load_labels(args.save_dir, state)

    id2label_dis = {i: t for i, t in enumerate(labels["dis"])}
    id2label_gene = {i: t for i, t in enumerate(labels["gene"])}

    # build & load model
    model = MultiHeadTokenClassifier(args.backbone, len(labels["dis"]), len(labels["gene"]))
    # state may be either raw model state or under key "model_state"
    model_state = state.get("model_state", state)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing)
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected)
    model.to(device).eval()

    # read CSV
    df = pd.read_csv(args.input_csv)
    if "text_en" not in df.columns:
        # Accept alternative column name "text"
        if "text" in df.columns:
            df.rename(columns={"text": "text_en"}, inplace=True)
        else:
            raise ValueError("Input CSV must contain a 'text_en' (or 'text') column.")

    # ensure text_zh and score columns exist (create empty if missing)
    if "text_zh" not in df.columns:
        df["text_zh"] = ""
    if "score" not in df.columns:
        df["score"] = None

    gene_found_list: List[bool] = []
    dis_found_list: List[bool] = []
    gene_ents_list: List[str] = []
    dis_ents_list: List[str] = []

    print(f"Processing {len(df)} rows ...")
    for i in tqdm(range(len(df))):
        text = str(df.loc[i, "text_en"])
        rows, gene_spans, dis_spans = predict_sentence(
            model, tokenizer, text, model_max, device, id2label_gene, id2label_dis
        )
        gene_found = len(gene_spans) > 0
        dis_found = len(dis_spans) > 0

        gene_found_list.append(gene_found)
        dis_found_list.append(dis_found)
        gene_ents_list.append("; ".join(gene_spans))
        dis_ents_list.append("; ".join(dis_spans))

    # augment dataframe
    df["gene_found"] = gene_found_list
    df["disease_found"] = dis_found_list
    df["gene_entities"] = gene_ents_list
    df["disease_entities"] = dis_ents_list

    # write combined annotated CSV
    out_all = os.path.join(args.out_dir, "merged_data1.translated.annotated.csv")
    df.to_csv(out_all, index=False, encoding="utf-8-sig")

    # split per spec:
    #  - If any B/I of a type is found, append to that type CSV.
    #  - If none (neither gene nor disease), append to BOTH type CSVs.
    gene_rows = []
    disease_rows = []
    for _, row in df.iterrows():
        g = bool(row["gene_found"])  # type: ignore
        d = bool(row["disease_found"])  # type: ignore
        if g:
            gene_rows.append(row)
        if d:
            disease_rows.append(row)
        if not g and not d:
            gene_rows.append(row)
            disease_rows.append(row)

    gene_df = pd.DataFrame(gene_rows).reset_index(drop=True)
    disease_df = pd.DataFrame(disease_rows).reset_index(drop=True)

    out_gene = os.path.join(args.out_dir, "merged_data1.translated.gene.csv")
    out_dis = os.path.join(args.out_dir, "merged_data1.translated.disease.csv")
    gene_df.to_csv(out_gene, index=False, encoding="utf-8-sig")
    disease_df.to_csv(out_dis, index=False, encoding="utf-8-sig")

    print(f"Saved annotated CSV: {out_all}")
    print(f"Saved gene CSV     : {out_gene} (rows={len(gene_df)})")
    print(f"Saved disease CSV  : {out_dis} (rows={len(disease_df)})")


if __name__ == "__main__":
    main()
