#!/usr/bin/env python3
"""
Multi-task NER + Intention (Router) training on shared BioLinkBERT encoder.

What's new vs previous script
- Adds a lightweight **Intention Classifier Head** (router) BEFORE the NER decoder heads.
- Router is trained from CSVs with weak labels (scores 0-10) for disease/gene intent.
- Shared encoder feeds: router → (optionally) NER heads. (Architecturally, router is computed first.)
- Joint loss: L = λ_ner * (NER losses) + λ_intent * (MSE on normalized scores).

Inputs
- NER datasets (CoNLL):
  * HUNER (gene) train/test
  * NCBI (disease) train/dev/test
- Intention datasets (CSV):
  * /cluster/home/gw/Backend_project/NER/dataset/data/merged_data1.translated.disease.csv
  * /cluster/home/gw/Backend_project/NER/dataset/data/merged_data1.translated.gene.csv
  Required columns: text_en, score (0-10)

Outputs
- TensorBoard logs
- Checkpoints with model_state and labels_* jsons

Run (example)
python train_router_first_multitask.py \
  --backbone /cluster/home/gw/Backend_project/models/BioLinkBERT-base \
  --save_dir /cluster/home/gw/Backend_project/NER/tuned \
  --disease_csv /cluster/home/gw/Backend_project/NER/dataset/data/merged_data1.translated.disease.csv \
  --gene_csv /cluster/home/gw/Backend_project/NER/dataset/data/merged_data1.translated.gene.csv
"""

import os
import io
import json
import math
import random
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
import time
# =====================
# CLI-like config (edit here or swap to argparse)
# =====================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
HF_BACKBONE = "/cluster/home/gw/Backend_project/models/BioLinkBERT-base"
SAVE_DIR = f"/cluster/home/gw/Backend_project/NER/tuned/intention/two_head{time.time()}"
LOG_DIR = os.path.join(SAVE_DIR, "runs")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

BASE = "/cluster/home/gw/Backend_project/NER/dataset/data"
HUNER_DIR = os.path.join(BASE, "huner_gene_nlm_gene")
NCBI_DIR = os.path.join(BASE, "ncbi_disease")

HUNER_TRAIN = os.path.join(
    HUNER_DIR,
    "SciSpacySentenceSplitter_core_sci_sm_0.5.1_SciSpacyTokenizer_core_sci_sm_0.5.1_train.conll",
)
HUNER_TEST = os.path.join(
    HUNER_DIR,
    "SciSpacySentenceSplitter_core_sci_sm_0.5.1_SciSpacyTokenizer_core_sci_sm_0.5.1_test.conll",
)
NCBI_TRAIN = os.path.join(
    NCBI_DIR,
    "SciSpacySentenceSplitter_core_sci_sm_0.5.1_SciSpacyTokenizer_core_sci_sm_0.5.1_train.conll",
)
NCBI_DEV = os.path.join(
    NCBI_DIR,
    "SciSpacySentenceSplitter_core_sci_sm_0.5.1_SciSpacyTokenizer_core_sci_sm_0.5.1_dev.conll",
)
NCBI_TEST = os.path.join(
    NCBI_DIR,
    "SciSpacySentenceSplitter_core_sci_sm_0.5.1_SciSpacyTokenizer_core_sci_sm_0.5.1_test.conll",
)

DISEASE_CSV = "/cluster/home/gw/Backend_project/NER/dataset/data/merged_data1.translated.disease.csv"
GENE_CSV    = "/cluster/home/gw/Backend_project/NER/dataset/data/merged_data1.translated.gene.csv"

# =====================
# Reproducibility
# =====================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SentenceTokens = List[str]
SentenceTags = List[str]

# =====================
# Data: CoNLL readers
# =====================

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

print("Loading local CoNLL corpora…")
ncbi_train = read_conll_sentences(NCBI_TRAIN)
ncbi_dev   = read_conll_sentences(NCBI_DEV)
ncbi_test  = read_conll_sentences(NCBI_TEST)
huner_train= read_conll_sentences(HUNER_TRAIN)
huner_test = read_conll_sentences(HUNER_TEST)


def collect_tags_from_pairs(pairs: List[Tuple[SentenceTokens, SentenceTags]]) -> List[str]:
    tags = set(t for _, ts in pairs for t in ts if t and t != "O")
    return ["O"] + sorted(tags)

labels_dis  = collect_tags_from_pairs(ncbi_train + ncbi_dev + ncbi_test)
labels_gene = collect_tags_from_pairs(huner_train + huner_test)
label2id_dis  = {t: i for i, t in enumerate(labels_dis)}
label2id_gene = {t: i for i, t in enumerate(labels_gene)}
id2label_dis  = {i: t for t, i in label2id_dis.items()}
id2label_gene = {i: t for t, i in label2id_gene.items()}

with open(os.path.join(SAVE_DIR, "labels_dis.json"), "w") as f:
    json.dump(labels_dis, f, ensure_ascii=False, indent=2)
with open(os.path.join(SAVE_DIR, "labels_gene.json"), "w") as f:
    json.dump(labels_gene, f, ensure_ascii=False, indent=2)

# =====================
# Tokenizer
# =====================
print("Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(HF_BACKBONE, use_fast=True)

# =====================
# Datasets
# =====================

def align_labels(word_labels: List[str], word_ids: List[Optional[int]], label2id: Dict[str, int], ignore_all: bool) -> List[int]:
    if ignore_all:
        return [-100 if wi is not None else -100 for wi in word_ids]
    return [label2id.get(word_labels[wi], label2id.get("O", 0)) if wi is not None else -100 for wi in word_ids]


class MultiTaskNERDataset(Dataset):
    """Mix of disease and gene CoNLL sentences.
    Each item is (input_ids, attention_mask, labels_disease, labels_gene).
    """
    def __init__(self, dis_pairs, gene_pairs, tokenizer, max_length=256):
        self.samples = [(t, g, "disease") for t, g in dis_pairs] + [(t, g, "gene") for t, g in gene_pairs]
        random.shuffle(self.samples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        words, word_tags, task = self.samples[idx]
        enc = self.tokenizer(words, is_split_into_words=True, truncation=True, max_length=self.max_length, padding=False, return_offsets_mapping=False)
        word_ids = enc.word_ids()
        if task == "disease":
            labels_disease = align_labels(word_tags, word_ids, label2id_dis, False)
            labels_gene    = align_labels(word_tags, word_ids, label2id_gene, True)
        else:
            labels_disease = align_labels(word_tags, word_ids, label2id_dis, True)
            labels_gene    = align_labels(word_tags, word_ids, label2id_gene, False)
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels_disease": torch.tensor(labels_disease, dtype=torch.long),
            "labels_gene": torch.tensor(labels_gene, dtype=torch.long),
        }


def ner_collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    def pad(seq, pad_val):
        return seq + [pad_val] * (max_len - len(seq))
    return {
        "input_ids": torch.tensor([pad(x["input_ids"].tolist(), tokenizer.pad_token_id) for x in batch], dtype=torch.long),
        "attention_mask": torch.tensor([pad(x["attention_mask"].tolist(), 0) for x in batch], dtype=torch.long),
        "labels_disease": torch.tensor([pad(x["labels_disease"].tolist(), -100) for x in batch], dtype=torch.long),
        "labels_gene": torch.tensor([pad(x["labels_gene"].tolist(), -100) for x in batch], dtype=torch.long),
    }

class IntentDataset(Dataset):
    """Sentence-level intention dataset from CSV.
    Produces targets as normalized scores in [0,1] for (gene, disease).
    Rows from the gene CSV set gene target; disease target is 0 unless present in disease CSV (and vice versa).
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
        # normalize to [0,1]
        m["gene_score"] = m["gene_score"].clip(0,10) / 10.0
        m["disease_score"] = m["disease_score"].clip(0,10) / 10.0
        self.df = m.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text_en"]) if isinstance(row["text_en"], str) else ""
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, padding=False, return_tensors=None)
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "target": torch.tensor([row["gene_score"], row["disease_score"]], dtype=torch.float),
        }

def intent_collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    def pad(seq, pad_val):
        return seq + [pad_val] * (max_len - len(seq))
    return {
        "input_ids": torch.tensor([pad(x["input_ids"].tolist(), tokenizer.pad_token_id) for x in batch], dtype=torch.long),
        "attention_mask": torch.tensor([pad(x["attention_mask"].tolist(), 0) for x in batch], dtype=torch.long),
        "target": torch.stack([x["target"] for x in batch], dim=0),  # (B,2)
    }

# =====================
# Dataloaders
# =====================
train_ds = MultiTaskNERDataset(ncbi_train, huner_train, tokenizer)
dev_ds   = MultiTaskNERDataset(ncbi_dev,   huner_test, tokenizer)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=ner_collate_fn)
dev_loader   = DataLoader(dev_ds,   batch_size=16, shuffle=False, collate_fn=ner_collate_fn)

intent_ds = IntentDataset(GENE_CSV, DISEASE_CSV, tokenizer, max_length=128)
intent_loader = DataLoader(intent_ds, batch_size=32, shuffle=True, collate_fn=intent_collate_fn)

# =====================
# Model with Router + NER heads
# =====================
class RouterFirstMultiHead(nn.Module):
    def __init__(self, backbone_name: str, num_dis: int, num_gene: int, hidden_router: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hid = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # Intention (router) head — small MLP → 2 outputs in [0,1]
        self.router = nn.Sequential(
            nn.Linear(hid, hidden_router),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_router, 2),
        )
        # Token classifiers
        self.classifier_dis = nn.Linear(hid, num_dis)
        self.classifier_gene = nn.Linear(hid, num_gene)

    def forward(self, input_ids, attention_mask, labels_disease: Optional[torch.Tensor] = None, labels_gene: Optional[torch.Tensor] = None, intent_targets: Optional[torch.Tensor] = None):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        H = self.dropout(enc.last_hidden_state)       # (B,T,h)
        CLS = H[:, 0]                                  # [CLS] pooled rep (BioLinkBERT uses CLS at pos 0)

        # ---- Router first ----
        router_logits = self.router(CLS)              # (B,2)
        router_probs  = torch.sigmoid(router_logits)  # multi-label probs in [0,1]

        # ---- NER decoders ----
        logits_dis  = self.classifier_dis(H)          # (B,T,Cd)
        logits_gene = self.classifier_gene(H)         # (B,T,Cg)

        loss, loss_dis, loss_gene, loss_int = None, None, None, None
        if labels_disease is not None and labels_gene is not None:
            ce = nn.CrossEntropyLoss(ignore_index=-100)
            loss_dis  = ce(logits_dis.view(-1, logits_dis.size(-1)),   labels_disease.view(-1))
            loss_gene = ce(logits_gene.view(-1, logits_gene.size(-1)), labels_gene.view(-1))
        if intent_targets is not None:
            # MSE on normalized scores (targets in [0,1])
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

# =====================
# Training setup
# =====================
EPOCHS = 6
LR = 2e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1
LAMBDA_INTENT = 1.0  # weight to scale intent loss contribution if doing joint steps

model = RouterFirstMultiHead(HF_BACKBONE, len(labels_dis), len(labels_gene)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
num_training_steps = EPOCHS * len(train_loader)
scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * num_training_steps), num_training_steps)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
writer = SummaryWriter(LOG_DIR)

writer.add_text("labels/disease", "".join(labels_dis))
writer.add_text("labels/gene", "".join(labels_gene))

best_dev_loss = float("inf")

# Prepare an iterator for the intent loader so we can alternate
intent_iter = iter(intent_loader)

def next_intent_batch():
    global intent_iter
    try:
        return next(intent_iter)
    except StopIteration:
        intent_iter = iter(intent_loader)
        return next(intent_iter)

# =====================
# Train epochs (alternating NER and Intent steps)
# =====================
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_ner_loss = 0.0
    total_int_loss = 0.0
    total_steps = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for step, ner_batch in enumerate(pbar, start=1):
        # ----------------- NER step -----------------
        optimizer.zero_grad(set_to_none=True)
        input_ids = ner_batch["input_ids"].to(device)
        attention_mask = ner_batch["attention_mask"].to(device)
        labels_disease = ner_batch["labels_disease"].to(device)
        labels_gene = ner_batch["labels_gene"].to(device)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out = model(input_ids, attention_mask, labels_disease, labels_gene, intent_targets=None)
            ner_loss = (out["loss_dis"] or 0) + (out["loss_gene"] or 0)
        scaler.scale(ner_loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_ner_loss += float(ner_loss.item())
        total_steps += 1
        writer.add_scalar("train/ner_loss", float(ner_loss.item()), (epoch-1)*len(train_loader)+step)

        # ----------------- Intent (router) step -----------------
        intent_batch = next_intent_batch()
        optimizer.zero_grad(set_to_none=True)
        i_input_ids = intent_batch["input_ids"].to(device)
        i_attention = intent_batch["attention_mask"].to(device)
        targets = intent_batch["target"].to(device)  # (B,2) in [0,1]
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out_i = model(i_input_ids, i_attention, labels_disease=None, labels_gene=None, intent_targets=targets)
            int_loss = out_i["loss_int"] * LAMBDA_INTENT
        scaler.scale(int_loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_int_loss += float(int_loss.item())
        writer.add_scalar("train/intent_loss", float(int_loss.item()), (epoch-1)*len(train_loader)+step)

        pbar.set_postfix({"ner_loss": f"{(total_ner_loss/max(1,total_steps)):.4f}", "int_loss": f"{(total_int_loss/max(1,total_steps)):.4f}"})

    # =====================
    # Validation on NER only (dev set)
    # =====================
    model.eval()
    dev_loss = 0.0
    dev_acc_dis_list, dev_acc_gene_list = [], []
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Validate", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_disease = batch["labels_disease"].to(device)
            labels_gene = batch["labels_gene"].to(device)
            out = model(input_ids, attention_mask, labels_disease, labels_gene, intent_targets=None)
            loss = (out["loss_dis"] or 0) + (out["loss_gene"] or 0)
            dev_loss += float(loss.item())
            # masked token accuracy
            def masked_acc(logits, labels):
                pred = logits.argmax(-1)
                mask = labels.ne(-100)
                tot = int(mask.sum().item())
                if tot == 0:
                    return 0.0
                correct = int((pred[mask] == labels[mask]).sum().item())
                return correct / tot
            dev_acc_dis_list.append(masked_acc(out["logits_dis"], labels_disease))
            dev_acc_gene_list.append(masked_acc(out["logits_gene"], labels_gene))

    dev_loss /= max(1, len(dev_loader))
    dev_acc_dis = float(sum(dev_acc_dis_list)/max(1,len(dev_acc_dis_list)))
    dev_acc_gene= float(sum(dev_acc_gene_list)/max(1,len(dev_acc_gene_list)))

    writer.add_scalar("dev/ner_loss", dev_loss, epoch)
    writer.add_scalar("dev/acc_dis", dev_acc_dis, epoch)
    writer.add_scalar("dev/acc_gene", dev_acc_gene, epoch)

    print(f"Epoch {epoch}: dev_ner_loss={dev_loss:.4f} | dis_acc={dev_acc_dis:.4f} | gene_acc={dev_acc_gene:.4f}")

    # Save checkpoints
    is_best = dev_loss < best_dev_loss
    ckpt = {
        "model_state": model.state_dict(),
        "labels_dis": labels_dis,
        "labels_gene": labels_gene,
        "backbone": HF_BACKBONE,
        "tokenizer": HF_BACKBONE,
        "epoch": epoch,
        "dev_ner_loss": dev_loss,
        "dev_acc_dis": dev_acc_dis,
        "dev_acc_gene": dev_acc_gene,
    }
    torch.save(ckpt, os.path.join(SAVE_DIR, "last_model.pt"))
    if is_best:
        best_dev_loss = dev_loss
        torch.save(ckpt, os.path.join(SAVE_DIR, "best_model.pt"))
        print("✓ Saved best model")

writer.flush()
writer.close()

# =====================
# Inference helpers
# =====================
@torch.no_grad()
def predict_intent(text: str, max_length: int = 128):
    model.eval()
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    out = model(enc["input_ids"], enc["attention_mask"], None, None, None)
    probs = out["router_probs"][0].detach().cpu().tolist()
    return {"gene_prob": probs[0], "disease_prob": probs[1]}

@torch.no_grad()
def predict_ner(text: str, max_length: int = 256):
    model.eval()
    words = text.split()
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(enc["input_ids"], enc["attention_mask"], None, None, None)
    def decode_logits(logits, mask, id2label):
        ids = logits.argmax(-1)[0].cpu().tolist()
        mask_l = mask[0].cpu().tolist()
        tags = [id2label[i] if m==1 else "PAD" for i,m in zip(ids, mask_l)]
        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        return list(zip(toks, tags))
    return {
        "gene": decode_logits(out["logits_gene"], enc["attention_mask"], id2label_gene),
        "disease": decode_logits(out["logits_dis"], enc["attention_mask"], id2label_dis),
    }
