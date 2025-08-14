import os
import io
import json
import random
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score

HF_BACKBONE = "/cluster/home/gw/Backend_project/models/BioLinkBERT-base"
SAVE_DIR = "/cluster/home/gw/Backend_project/NER/tuned"
BASE = "/cluster/home/gw/Backend_project/NER/dataset/data"
HUNER_DIR = os.path.join(BASE, "huner_gene_nlm_gene")
NCBI_DIR = os.path.join(BASE, "ncbi_disease")

HUNER_TRAIN = os.path.join(HUNER_DIR, "SciSpacySentenceSplitter_core_sci_sm_0.5.1_SciSpacyTokenizer_core_sci_sm_0.5.1_train.conll")
NCBI_TRAIN = os.path.join(NCBI_DIR, "SciSpacySentenceSplitter_core_sci_sm_0.5.1_SciSpacyTokenizer_core_sci_sm_0.5.1_train.conll")

SentenceTokens = List[str]
SentenceTags = List[str]
Pair = Tuple[SentenceTokens, SentenceTags]

def read_conll_sentences(path: str) -> List[Pair]:
    sent_tokens, sent_tags, all_sents = [], [], []
    with io.open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if sent_tokens:
                    all_sents.append((sent_tokens, sent_tags))
                    sent_tokens, sent_tags = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            token, tag = parts[0], parts[1]
            sent_tokens.append(token)
            sent_tags.append(tag)
    if sent_tokens:
        all_sents.append((sent_tokens, sent_tags))
    return all_sents

def load_label_space(save_dir: str, key: str) -> Optional[List[str]]:
    path = os.path.join(save_dir, f"{key}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

class MultiHeadTokenClassifier(nn.Module):
    def __init__(self, backbone_name, num_dis, num_gene, dropout=0.1):
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
def decode_word_level(tokenizer, words, logits_dis, logits_gene, input_ids, attention_mask, id2label_dis, id2label_gene):
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=input_ids.size(1), padding=False)
    word_ids = enc.word_ids()
    pred_dis_tok = logits_dis.argmax(-1).squeeze(0).tolist()
    pred_gene_tok = logits_gene.argmax(-1).squeeze(0).tolist()
    attn = attention_mask.squeeze(0).tolist()
    word_seen = set()
    out_rows = []
    for idx, wid in enumerate(word_ids):
        if wid is None or attn[idx] == 0 or wid in word_seen:
            continue
        word_seen.add(wid)
        w = words[wid]
        pd = id2label_dis.get(pred_dis_tok[idx], "O")
        pg = id2label_gene.get(pred_gene_tok[idx], "O")
        out_rows.append((w, pg, pd))
    return out_rows

@torch.no_grad()
def main(n_samples: int = 8, seed: int = 0):
    random.seed(seed)
    ncbi_train = read_conll_sentences(NCBI_TRAIN)
    huner_train = read_conll_sentences(HUNER_TRAIN)
    labels_dis = load_label_space(SAVE_DIR, "labels_dis")
    labels_gene = load_label_space(SAVE_DIR, "labels_gene")
    if labels_dis is None:
        labels_dis = ["O"] + sorted(set(t for _, ts in ncbi_train for t in ts if t and t != "O"))
    if labels_gene is None:
        labels_gene = ["O"] + sorted(set(t for _, ts in huner_train for t in ts if t and t != "O"))
    id2label_dis = {i: t for i, t in enumerate(labels_dis)}
    id2label_gene = {i: t for i, t in enumerate(labels_gene)}
    tokenizer = AutoTokenizer.from_pretrained(HF_BACKBONE, use_fast=True)
    model = MultiHeadTokenClassifier(HF_BACKBONE, len(labels_dis), len(labels_gene))
    ckpt_path = os.path.join(SAVE_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(SAVE_DIR, "last_model.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    pool = [(w, t, "disease") for (w, t) in ncbi_train] + [(w, t, "gene") for (w, t) in huner_train]
    random.shuffle(pool)
    pick = pool[: max(1, n_samples)]

    gold_gene_all, pred_gene_all = [], []
    gold_dis_all, pred_dis_all = [], []

    for i, (words, gold_tags, task) in enumerate(pick, start=1):
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=256, padding=False, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])
        rows = decode_word_level(tokenizer, words, out["logits_dis"], out["logits_gene"], enc["input_ids"], enc["attention_mask"], id2label_dis, id2label_gene)

        if task == "disease":
            gold_gene = ["O"] * len(gold_tags)
            gold_dis = gold_tags
        else:
            gold_gene = gold_tags
            gold_dis = ["O"] * len(gold_tags)

        pred_gene = [pg for _, pg, _ in rows]
        pred_dis = [pd for _, _, pd in rows]

        gold_gene_all.append(gold_gene)
        pred_gene_all.append(pred_gene)
        gold_dis_all.append(gold_dis)
        pred_dis_all.append(pred_dis)

        print(f"Sample {i} | Task={task}")
        print(f"{'WORD':20s} | {'GOLD-GENE':12s} | {'PRED-GENE':12s} | {'GOLD-DIS':12s} | {'PRED-DIS':12s}")
        print("=" * 80)
        for idx, (w, pg, pd) in enumerate(rows):
            g_gene = gold_gene[idx]
            g_dis = gold_dis[idx]
            print(f"{w[:20]:20s} | {g_gene:12s} | {pg:12s} | {g_dis:12s} | {pd:12s}")
        print("-" * 80)

    print("\n=== Metrics for Gene NER ===")
    print(f"Precision: {precision_score(gold_gene_all, pred_gene_all):.4f}")
    print(f"Recall:    {recall_score(gold_gene_all, pred_gene_all):.4f}")
    print(f"F1:        {f1_score(gold_gene_all, pred_gene_all):.4f}")
    print(classification_report(gold_gene_all, pred_gene_all))

    print("\n=== Metrics for Disease NER ===")
    print(f"Precision: {precision_score(gold_dis_all, pred_dis_all):.4f}")
    print(f"Recall:    {recall_score(gold_dis_all, pred_dis_all):.4f}")
    print(f"F1:        {f1_score(gold_dis_all, pred_dis_all):.4f}")
    print(classification_report(gold_dis_all, pred_dis_all))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate and print metrics.")
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(n_samples=args.num, seed=args.seed)
