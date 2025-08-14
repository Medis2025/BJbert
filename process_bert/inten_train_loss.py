#!/usr/bin/env python3
"""
Router-First Multi-Task NER Training (BioLinkBERT encoder + Intent Router + Dual NER heads)

What's included
- Shared BioLinkBERT encoder
- Router (intent) head trained with weak labels from CSVs (gene/disease scores in [0..10])
- Two token-classification heads (disease, gene)
- Alternating optimization: one NER step per batch + one Intent step per NER batch
- Comprehensive TensorBoard logging (losses, learning rate, grad norm, dev metrics)
- Dev evaluation:
  * Masked token accuracies for both tasks
  * Span-level micro Precision/Recall/F1 for disease (on NCBI dev) and gene (on HuNER test)
- Best/last checkpointing with label spaces persisted to disk

Example run
python train_router_first_multitask_v2.py \
  --backbone /cluster/home/gw/Backend_project/models/BioLinkBERT-base \
  --save_dir /cluster/home/gw/Backend_project/NER/tuned/intention_v2 \
  --huner_train /cluster/home/gw/Backend_project/NER/dataset/data/huner_gene_nlm_gene/SciSpacySentenceSplitter_core_sci_sm_0.5.1_SciSpacyTokenizer_core_sci_sm_0.5.1_train.conll \
  --huner_test  /cluster/home/gw/Backend_project/NER/dataset/data/huner_gene_nlm_gene/SciSpacySentenceSplitter_core_sci_sm_0.5.1_SciSpacyTokenizer_core_sci_sm_0.5.1_test.conll \
  --ncbi_train  /cluster/home/gw/Backend_project/NER/dataset/data/ncbi_disease/SciSpacySentenceSplitter_core_sci_sm_0.5.1_SciSpacyTokenizer_core_sci_sm_0.5.1_train.conll \
  --ncbi_dev    /cluster/home/gw/Backend_project/NER/dataset/data/ncbi_disease/SciSpacySentenceSplitter_core_sci_sm_0.5.1_SciSpacyTokenizer_core_sci_sm_0.5.1_dev.conll \
  --ncbi_test   /cluster/home/gw/Backend_project/NER/dataset/data/ncbi_disease/SciSpacySentenceSplitter_core_sci_sm_0.5.1_SciSpacyTokenizer_core_sci_sm_0.5.1_test.conll \
  --gene_csv    /cluster/home/gw/Backend_project/NER/dataset/data/merged_data1.translated.gene.csv \
  --disease_csv /cluster/home/gw/Backend_project/NER/dataset/data/merged_data1.translated.disease.csv \
  --epochs 6 --batch_size 16 --intent_batch_size 32 --max_len 256 --lr 2e-5
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use CUDA:1
import io
import json
import math
import random
import argparse
import string
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd

# -------------------- Reproducibility -------------------- #
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SentenceTokens = List[str]
SentenceTags = List[str]

# -------------------- I/O helpers -------------------- #
def read_conll_sentences(path: str) -> List[Tuple[SentenceTokens, SentenceTags]]:
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

# -------------------- Label utilities -------------------- #
def collect_tags_from_pairs(pairs: List[Tuple[SentenceTokens, SentenceTags]]) -> List[str]:
    tags = set(t for _, ts in pairs for t in ts if t and t != "O")
    return ["O"] + sorted(tags)

# -------------------- Tokenizer alignment -------------------- #
def align_labels(word_labels: List[str], word_ids: List[Optional[int]], label2id: Dict[str, int], ignore_all: bool) -> List[int]:
    if ignore_all:
        return [-100 if wi is not None else -100 for wi in word_ids]
    return [label2id.get(word_labels[wi], label2id.get("O", 0)) if wi is not None else -100 for wi in word_ids]

# -------------------- Datasets -------------------- #
class MultiTaskNERDataset(Dataset):
    """Mix of disease and gene CoNLL sentences.
    Each item returns tokenized tensors and which task it belongs to.
    """
    def __init__(self, dis_pairs, gene_pairs, tokenizer, label2id_dis, label2id_gene, max_length=256):
        self.samples = [(t, g, "disease") for t, g in dis_pairs] + [(t, g, "gene") for t, g in gene_pairs]
        random.shuffle(self.samples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id_dis = label2id_dis
        self.label2id_gene = label2id_gene

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        words, word_tags, task = self.samples[idx]
        enc = self.tokenizer(words, is_split_into_words=True, truncation=True, max_length=self.max_length, padding=False, return_offsets_mapping=False)
        word_ids = enc.word_ids()
        if task == "disease":
            labels_disease = align_labels(word_tags, word_ids, self.label2id_dis, False)
            labels_gene    = align_labels(word_tags, word_ids, self.label2id_gene, True)
        else:
            labels_disease = align_labels(word_tags, word_ids, self.label2id_dis, True)
            labels_gene    = align_labels(word_tags, word_ids, self.label2id_gene, False)
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels_disease": torch.tensor(labels_disease, dtype=torch.long),
            "labels_gene": torch.tensor(labels_gene, dtype=torch.long),
            "task": task,
            "words": words,
            "word_tags": word_tags,
        }

def ner_collate_fn(batch, pad_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    def pad(seq, pad_val):
        return seq + [pad_val] * (max_len - len(seq))
    return {
        "input_ids": torch.tensor([pad(x["input_ids"].tolist(), pad_id) for x in batch], dtype=torch.long),
        "attention_mask": torch.tensor([pad(x["attention_mask"].tolist(), 0) for x in batch], dtype=torch.long),
        "labels_disease": torch.tensor([pad(x["labels_disease"].tolist(), -100) for x in batch], dtype=torch.long),
        "labels_gene": torch.tensor([pad(x["labels_gene"].tolist(), -100) for x in batch], dtype=torch.long),
        "task": [x["task"] for x in batch],
        "words": [x["words"] for x in batch],
        "word_tags": [x["word_tags"] for x in batch],
    }

class IntentDataset(Dataset):
    """Sentence-level intention dataset from CSV.
    Expects columns: text_en, score (0..10) in each CSV.
    Produces targets as normalized scores in [0,1] for (gene, disease).
    """
    def __init__(self, gene_csv: str, disease_csv: str, tokenizer, max_length: int = 128):
        gdf = pd.read_csv(gene_csv)
        ddf = pd.read_csv(disease_csv)
        for df in (gdf, ddf):
            if "text_en" not in df.columns or "score" not in df.columns:
                raise ValueError("CSV must contain 'text_en' and 'score' columns")
        gdf = gdf[["text_en", "score"]].rename(columns={"score": "gene_score"})
        ddf = ddf[["text_en", "score"]].rename(columns={"score": "disease_score"})
        m = pd.merge(gdf, ddf, on="text_en", how="outer")
        m["gene_score"].fillna(0, inplace=True)
        m["disease_score"].fillna(0, inplace=True)
        m["gene_score"] = m["gene_score"].clip(0,10)/10.0
        m["disease_score"] = m["disease_score"].clip(0,10)/10.0
        self.df = m.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text_en"]) if isinstance(row["text_en"], str) else ""
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, padding=False)
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "target": torch.tensor([row["gene_score"], row["disease_score"]], dtype=torch.float),
        }

def intent_collate_fn(batch, pad_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    def pad(seq, pad_val):
        return seq + [pad_val] * (max_len - len(seq))
    return {
        "input_ids": torch.tensor([pad(x["input_ids"].tolist(), pad_id) for x in batch], dtype=torch.long),
        "attention_mask": torch.tensor([pad(x["attention_mask"].tolist(), 0) for x in batch], dtype=torch.long),
        "target": torch.stack([x["target"] for x in batch], dim=0),
    }

# -------------------- Model -------------------- #
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
            nn.Linear(hidden_router, 2),  # gene, disease
        )
        self.classifier_dis  = nn.Linear(hid, num_dis)
        self.classifier_gene = nn.Linear(hid, num_gene)

    def forward(self, input_ids, attention_mask,
                labels_disease: Optional[torch.Tensor] = None,
                labels_gene: Optional[torch.Tensor] = None,
                intent_targets: Optional[torch.Tensor] = None):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        H = self.dropout(enc.last_hidden_state)
        CLS = H[:, 0]
        router_logits = self.router(CLS)
        router_probs  = torch.sigmoid(router_logits)  # multi-label probs
        logits_dis  = self.classifier_dis(H)
        logits_gene = self.classifier_gene(H)

        loss = None
        loss_dis = None
        loss_gene = None
        loss_int = None

        if labels_disease is not None and labels_gene is not None:
            ce = nn.CrossEntropyLoss(ignore_index=-100)
            loss_dis  = ce(logits_dis.view(-1, logits_dis.size(-1)), labels_disease.view(-1))
            loss_gene = ce(logits_gene.view(-1, logits_gene.size(-1)), labels_gene.view(-1))
        if intent_targets is not None:
            loss_int = nn.functional.mse_loss(router_probs, intent_targets)
        if (loss_dis is not None) or (loss_gene is not None) or (loss_int is not None):
            loss = (loss_dis or 0) + (loss_gene or 0) + (loss_int or 0)

        return {
            "loss": loss,
            "loss_dis": loss_dis,
            "loss_gene": loss_gene,
            "loss_int": loss_int,
            "router_probs": router_probs,
            "logits_dis": logits_dis,
            "logits_gene": logits_gene,
        }

# -------------------- Span helpers -------------------- #
_punct_tbl = str.maketrans({p: " " for p in string.punctuation})

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = s.translate(_punct_tbl)
    s = " ".join(s.split())
    return s

def bio_spans_from_rows(rows: List[Tuple[str, str]]) -> List[str]:
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

@torch.no_grad()
def decode_words_with_logits(tokenizer, words: List[str], logits, input_ids, attention_mask, id2label) -> List[Tuple[str, str]]:
    # Re-encode to obtain word_ids mapping
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=input_ids.size(1), padding=False)
    word_ids = enc.word_ids()
    pred_tok = logits.argmax(-1).squeeze(0).tolist()
    attn = attention_mask.squeeze(0).tolist()
    out_rows = []
    used = set()
    for ti, wid in enumerate(word_ids):
        if wid is None or attn[ti] == 0 or wid in used:
            continue
        used.add(wid)
        w = words[wid]
        tg = id2label.get(pred_tok[ti], "O")
        out_rows.append((w, tg))
    return out_rows

# Evaluate span F1 for one task (pairs: (tokens, tags))
@torch.no_grad()
def eval_span_prf(model, tokenizer, pairs: List[Tuple[List[str], List[str]]], which: str, id2label: Dict[int,str], max_length: int = 256):
    assert which in {"gene", "disease"}
    TP = FP = FN = 0
    for words, gold_tags in pairs:
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length, padding=False, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"], None, None, None)
        logits = out["logits_gene"] if which == "gene" else out["logits_dis"]
        rows_pred = decode_words_with_logits(tokenizer, words, logits, enc["input_ids"], enc["attention_mask"], id2label)
        pred_spans = bio_spans_from_rows(rows_pred)
        gold_spans = bio_spans_from_rows(list(zip(words, gold_tags)))
        matched_gold = 0
        gold_set = set(_norm(s) for s in gold_spans)
        pred_set = [_norm(s) for s in pred_spans]
        for g in gold_set:
            if g in pred_set:
                TP += 1
                matched_gold += 1
            else:
                FN += 1
        FP += max(0, len(pred_set) - matched_gold)
    P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = 0.0 if (P + R) == 0 else 2*P*R/(P+R)
    return P, R, F1, TP, FP, FN

# -------------------- Main train -------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', type=str, required=True)
    ap.add_argument('--save_dir', type=str, required=True)
    ap.add_argument('--huner_train', type=str, required=True)
    ap.add_argument('--huner_test', type=str, required=True)
    ap.add_argument('--ncbi_train', type=str, required=True)
    ap.add_argument('--ncbi_dev', type=str, required=True)
    ap.add_argument('--ncbi_test', type=str, required=True)
    ap.add_argument('--gene_csv', type=str, required=True)
    ap.add_argument('--disease_csv', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--intent_batch_size', type=int, default=32)
    ap.add_argument('--max_len', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--warmup_ratio', type=float, default=0.1)
    ap.add_argument('--lambda_intent', type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(args.save_dir, 'runs')
    os.makedirs(log_dir, exist_ok=True)

    # Load corpora
    print('Loading corpora...')
    ncbi_train = read_conll_sentences(args.ncbi_train)
    ncbi_dev   = read_conll_sentences(args.ncbi_dev)
    ncbi_test  = read_conll_sentences(args.ncbi_test)
    huner_train= read_conll_sentences(args.huner_train)
    huner_test = read_conll_sentences(args.huner_test)

    # Label spaces
    labels_dis  = collect_tags_from_pairs(ncbi_train + ncbi_dev + ncbi_test)
    labels_gene = collect_tags_from_pairs(huner_train + huner_test)
    label2id_dis  = {t: i for i, t in enumerate(labels_dis)}
    label2id_gene = {t: i for i, t in enumerate(labels_gene)}
    id2label_dis  = {i: t for t, i in label2id_dis.items()}
    id2label_gene = {i: t for t, i in label2id_gene.items()}

    with open(os.path.join(args.save_dir, 'labels_dis.json'), 'w', encoding='utf-8') as f:
        json.dump(labels_dis, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.save_dir, 'labels_gene.json'), 'w', encoding='utf-8') as f:
        json.dump(labels_gene, f, ensure_ascii=False, indent=2)

    # Tokenizer
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)

    # Datasets & loaders
    train_ds = MultiTaskNERDataset(ncbi_train, huner_train, tokenizer, label2id_dis, label2id_gene, max_length=args.max_len)
    dev_ds   = MultiTaskNERDataset(ncbi_dev,   huner_test, tokenizer, label2id_dis, label2id_gene, max_length=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: ner_collate_fn(b, tokenizer.pad_token_id))
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b: ner_collate_fn(b, tokenizer.pad_token_id))

    intent_ds = IntentDataset(args.gene_csv, args.disease_csv, tokenizer, max_length=128)
    intent_loader = DataLoader(intent_ds, batch_size=args.intent_batch_size, shuffle=True,
                               collate_fn=lambda b: intent_collate_fn(b, tokenizer.pad_token_id))

    # Model
    model = RouterFirstMultiHead(args.backbone, len(labels_dis), len(labels_gene)).to(device)

    # Optimizer/scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = args.epochs * len(train_loader) * 2  # NER + Intent steps per batch
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * num_training_steps), num_training_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Logger
    writer = SummaryWriter(log_dir)
    writer.add_text('labels/disease', '\n'.join(labels_dis))
    writer.add_text('labels/gene', '\n'.join(labels_gene))

    # Intent iterator
    intent_iter = iter(intent_loader)
    def next_intent_batch():
        nonlocal intent_iter
        try:
            return next(intent_iter)
        except StopIteration:
            intent_iter = iter(intent_loader)
            return next(intent_iter)

    best_dev_loss = float('inf')

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_ner_loss = 0.0
        tot_int_loss = 0.0
        tot_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar, start=1):
            # ---- NER step ----
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_disease = batch['labels_disease'].to(device)
            labels_gene = batch['labels_gene'].to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(input_ids, attention_mask, labels_disease, labels_gene, intent_targets=None)
                ner_loss = (out['loss_dis'] or 0) + (out['loss_gene'] or 0)

            scaler.scale(ner_loss).backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tot_ner_loss += float(ner_loss.item())
            tot_steps += 1
            global_step += 1

            writer.add_scalar('train/ner_loss', float(ner_loss.item()), global_step)
            if out['loss_dis'] is not None:
                writer.add_scalar('train/loss_dis', float(out['loss_dis'].item()), global_step)
            if out['loss_gene'] is not None:
                writer.add_scalar('train/loss_gene', float(out['loss_gene'].item()), global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
            writer.add_scalar('train/grad_norm', float(grad_norm), global_step)

            # ---- Intent step ----
            intent_batch = next_intent_batch()
            optimizer.zero_grad(set_to_none=True)
            i_input_ids = intent_batch['input_ids'].to(device)
            i_attention = intent_batch['attention_mask'].to(device)
            targets = intent_batch['target'].to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out_i = model(i_input_ids, i_attention, None, None, intent_targets=targets)
                int_loss = (out_i['loss_int'] or 0) * args.lambda_intent

            scaler.scale(int_loss).backward()
            grad_norm_i = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tot_int_loss += float(int_loss.item())
            global_step += 1
            writer.add_scalar('train/intent_loss', float(int_loss.item()), global_step)
            writer.add_scalar('train/grad_norm_int', float(grad_norm_i), global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)

            pbar.set_postfix({
                'ner_loss': f"{(tot_ner_loss/max(1,tot_steps)):.4f}",
                'intent_loss': f"{(tot_int_loss/max(1,tot_steps)):.4f}",
            })

        # ---------------- Dev evaluation ---------------- #
        model.eval()
        dev_loss_dis = 0.0
        dev_loss_gene = 0.0
        dev_acc_dis_list, dev_acc_gene_list = [], []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc='Validate', leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_disease = batch['labels_disease'].to(device)
                labels_gene = batch['labels_gene'].to(device)
                out = model(input_ids, attention_mask, labels_disease, labels_gene, None)
                if out['loss_dis'] is not None:
                    dev_loss_dis += float(out['loss_dis'].item())
                if out['loss_gene'] is not None:
                    dev_loss_gene += float(out['loss_gene'].item())
                # masked token accuracy
                def masked_acc(logits, labels):
                    pred = logits.argmax(-1)
                    mask = labels.ne(-100)
                    tot = int(mask.sum().item())
                    if tot == 0:
                        return 0.0
                    correct = int((pred[mask] == labels[mask]).sum().item())
                    return correct / tot
                dev_acc_dis_list.append(masked_acc(out['logits_dis'], labels_disease))
                dev_acc_gene_list.append(masked_acc(out['logits_gene'], labels_gene))

        n_dev_batches = max(1, len(dev_loader))
        dev_loss_dis /= n_dev_batches
        dev_loss_gene /= n_dev_batches
        dev_ner_loss = dev_loss_dis + dev_loss_gene
        dev_acc_dis = float(sum(dev_acc_dis_list)/max(1,len(dev_acc_dis_list)))
        dev_acc_gene= float(sum(dev_acc_gene_list)/max(1,len(dev_acc_gene_list)))

        # Span-level PRF on original pairs (disease on ncbi_dev, gene on huner_test)
        P_dis, R_dis, F1_dis, TPd, FPd, FNd = eval_span_prf(model, tokenizer, ncbi_dev, which='disease', id2label=id2label_dis, max_length=args.max_len)
        P_gene,R_gene,F1_gene,TPg,FPg,FNg = eval_span_prf(model, tokenizer, huner_test, which='gene',    id2label=id2label_gene, max_length=args.max_len)

        # Log dev scalars
        writer.add_scalar('dev/ner_loss', dev_ner_loss, epoch)
        writer.add_scalar('dev/loss_dis', dev_loss_dis, epoch)
        writer.add_scalar('dev/loss_gene', dev_loss_gene, epoch)
        writer.add_scalar('dev/acc_dis_masked', dev_acc_dis, epoch)
        writer.add_scalar('dev/acc_gene_masked', dev_acc_gene, epoch)
        writer.add_scalar('dev_span/P_dis', P_dis, epoch)
        writer.add_scalar('dev_span/R_dis', R_dis, epoch)
        writer.add_scalar('dev_span/F1_dis', F1_dis, epoch)
        writer.add_scalar('dev_span/P_gene', P_gene, epoch)
        writer.add_scalar('dev_span/R_gene', R_gene, epoch)
        writer.add_scalar('dev_span/F1_gene', F1_gene, epoch)

        print(f"Epoch {epoch}: dev_ner_loss={dev_ner_loss:.4f} | dis_acc(masked)={dev_acc_dis:.4f} | gene_acc(masked)={dev_acc_gene:.4f}")
        print(f"Span DEV (Disease): P={P_dis:.4f} R={R_dis:.4f} F1={F1_dis:.4f} | TP={TPd} FP={FPd} FN={FNd}")
        print(f"Span DEV (Gene):    P={P_gene:.4f} R={R_gene:.4f} F1={F1_gene:.4f} | TP={TPg} FP={FPg} FN={FNg}")

        # Save checkpoints
        ckpt = {
            'model_state': model.state_dict(),
            'labels_dis': labels_dis,
            'labels_gene': labels_gene,
            'backbone': args.backbone,
            'tokenizer': args.backbone,
            'epoch': epoch,
            'dev_ner_loss': dev_ner_loss,
            'dev_acc_dis_masked': dev_acc_dis,
            'dev_acc_gene_masked': dev_acc_gene,
            'dev_span_F1_dis': F1_dis,
            'dev_span_F1_gene': F1_gene,
        }
        torch.save(ckpt, os.path.join(args.save_dir, 'last_model.pt'))
        if dev_ner_loss < best_dev_loss:
            best_dev_loss = dev_ner_loss
            torch.save(ckpt, os.path.join(args.save_dir, 'best_model.pt'))
            print('✓ Saved best model')

    writer.flush()
    writer.close()

# -------------------- Inference helpers -------------------- #
@torch.no_grad()
def predict_intent_text(model, tokenizer, text: str, max_length: int = 128):
    model.eval()
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(enc['input_ids'], enc['attention_mask'], None, None, None)
    probs = out['router_probs'][0].detach().cpu().tolist()
    return {'gene_prob': probs[0], 'disease_prob': probs[1]}

@torch.no_grad()
def predict_ner_tokens(model, tokenizer, text: str, id2label_dis: Dict[int,str], id2label_gene: Dict[int,str], max_length: int = 256):
    model.eval()
    words = text.split()
    enc = tokenizer(words, is_split_into_words=True, return_tensors='pt', truncation=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(enc['input_ids'], enc['attention_mask'], None, None, None)
    def decode_logits(logits, mask, id2label):
        ids = logits.argmax(-1)[0].cpu().tolist()
        mask_l = mask[0].cpu().tolist()
        return [id2label[i] if m==1 else 'PAD' for i,m in zip(ids, mask_l)]
    return {
        'gene_tags': decode_logits(out['logits_gene'], enc['attention_mask'], id2label_gene),
        'disease_tags': decode_logits(out['logits_dis'], enc['attention_mask'], id2label_dis),
        'tokens': tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
    }

if __name__ == '__main__':
    main()
