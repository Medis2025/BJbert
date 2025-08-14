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

# =====================
# Config
# =====================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # default to CUDA:1 if not set
HF_BACKBONE = "/cluster/home/gw/Backend_project/models/BioLinkBERT-base"
SAVE_DIR = "/cluster/home/gw/Backend_project/NER/tuned"
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

# =====================
# Reproducibility
# =====================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SentenceTokens = List[str]
SentenceTags = List[str]


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
ncbi_dev = read_conll_sentences(NCBI_DEV)
ncbi_test = read_conll_sentences(NCBI_TEST)
huner_train = read_conll_sentences(HUNER_TRAIN)
huner_test = read_conll_sentences(HUNER_TEST)


def collect_tags_from_pairs(
    pairs: List[Tuple[SentenceTokens, SentenceTags]]
) -> List[str]:
    tags = set(t for _, ts in pairs for t in ts if t and t != "O")
    return ["O"] + sorted(tags)


labels_dis = collect_tags_from_pairs(ncbi_train + ncbi_dev + ncbi_test)
labels_gene = collect_tags_from_pairs(huner_train + huner_test)

label2id_dis = {t: i for i, t in enumerate(labels_dis)}
label2id_gene = {t: i for i, t in enumerate(labels_gene)}
id2label_dis = {i: t for t, i in label2id_dis.items()}
id2label_gene = {i: t for t, i in label2id_gene.items()}

print(f"Disease labels: {len(labels_dis)} classes")
print(f"Gene labels:    {len(labels_gene)} classes")

# Persist label maps for later inference
with open(os.path.join(SAVE_DIR, "labels_dis.json"), "w") as f:
    json.dump(labels_dis, f, ensure_ascii=False, indent=2)
with open(os.path.join(SAVE_DIR, "labels_gene.json"), "w") as f:
    json.dump(labels_gene, f, ensure_ascii=False, indent=2)

# =====================
# Tokenizer
# =====================
print("Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(HF_BACKBONE, use_fast=True)


def align_labels(
    word_labels: List[str],
    word_ids: List[Optional[int]],
    label2id: Dict[str, int],
    ignore_all: bool,
) -> List[int]:
    if ignore_all:
        return [-100 if wi is not None else -100 for wi in word_ids]
    return [
        label2id.get(word_labels[wi], label2id.get("O", 0)) if wi is not None else -100
        for wi in word_ids
    ]


class MultiTaskNERItem(Dict[str, torch.Tensor]):
    pass


class MultiTaskNERDataset(Dataset):
    def __init__(self, dis_pairs, gene_pairs, tokenizer, max_length=256):
        # Create task-typed samples
        self.samples = [(t, g, "disease") for t, g in dis_pairs] + [
            (t, g, "gene") for t, g in gene_pairs
        ]
        random.shuffle(self.samples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        words, word_tags, task = self.samples[idx]
        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_offsets_mapping=False,
        )
        word_ids = enc.word_ids()
        if task == "disease":
            labels_disease = align_labels(word_tags, word_ids, label2id_dis, False)
            labels_gene = align_labels(word_tags, word_ids, label2id_gene, True)
        else:
            labels_disease = align_labels(word_tags, word_ids, label2id_dis, True)
            labels_gene = align_labels(word_tags, word_ids, label2id_gene, False)

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels_disease": torch.tensor(labels_disease, dtype=torch.long),
            "labels_gene": torch.tensor(labels_gene, dtype=torch.long),
        }


def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)

    def pad(seq, pad_val):
        return seq + [pad_val] * (max_len - len(seq))

    return {
        "input_ids": torch.tensor(
            [pad(x["input_ids"].tolist(), tokenizer.pad_token_id) for x in batch],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [pad(x["attention_mask"].tolist(), 0) for x in batch], dtype=torch.long
        ),
        "labels_disease": torch.tensor(
            [pad(x["labels_disease"].tolist(), -100) for x in batch], dtype=torch.long
        ),
        "labels_gene": torch.tensor(
            [pad(x["labels_gene"].tolist(), -100) for x in batch], dtype=torch.long
        ),
    }


# =====================
# Dataloaders
# =====================
train_ds = MultiTaskNERDataset(ncbi_train, huner_train, tokenizer)
dev_ds = MultiTaskNERDataset(ncbi_dev, huner_test, tokenizer)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)


# =====================
# Model
# =====================
class MultiHeadTokenClassifier(nn.Module):
    def __init__(self, backbone_name, num_dis, num_gene, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier_dis = nn.Linear(hidden, num_dis)
        self.classifier_gene = nn.Linear(hidden, num_gene)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels_disease: Optional[torch.Tensor] = None,
        labels_gene: Optional[torch.Tensor] = None,
    ):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(out.last_hidden_state)
        logits_dis = self.classifier_dis(seq)
        logits_gene = self.classifier_gene(seq)

        loss = None
        loss_dis = None
        loss_gene = None
        if labels_disease is not None and labels_gene is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss_dis = loss_fct(
                logits_dis.view(-1, logits_dis.size(-1)), labels_disease.view(-1)
            )
            loss_gene = loss_fct(
                logits_gene.view(-1, logits_gene.size(-1)), labels_gene.view(-1)
            )
            loss = loss_dis + loss_gene
        return {
            "loss": loss,
            "loss_dis": loss_dis,
            "loss_gene": loss_gene,
            "logits_dis": logits_dis,
            "logits_gene": logits_gene,
        }


# =====================
# Metrics
# =====================
@torch.no_grad()
def masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute token accuracy ignoring -100."""
    pred = logits.argmax(-1)
    mask = labels.ne(-100)
    if mask.sum().item() == 0:
        return 0.0
    correct = (pred[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / max(1, total)


# =====================
# Train
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiHeadTokenClassifier(HF_BACKBONE, len(labels_dis), len(labels_gene)).to(device)

EPOCHS = 6
LR = 2e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
num_training_steps = EPOCHS * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optimizer, int(WARMUP_RATIO * num_training_steps), num_training_steps
)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
writer = SummaryWriter(LOG_DIR)

# Log label spaces
writer.add_text("labels/disease", "\n".join(labels_dis))
writer.add_text("labels/gene", "\n".join(labels_gene))

best_dev_loss = float("inf")

def log_graph_once():
    # Build a tiny example batch to log the graph (do it once)
    model.eval()
    with torch.no_grad():
        ex = next(iter(dev_loader))
        ex_ids = ex["input_ids"][:1].to(device)
        ex_mask = ex["attention_mask"][:1].to(device)
        ex_ld = ex["labels_disease"][:1].to(device)
        ex_lg = ex["labels_gene"][:1].to(device)
        try:
            writer.add_graph(model, (ex_ids, ex_mask, ex_ld, ex_lg))
        except Exception as e:
            print(f"[WARN] add_graph failed (skipping): {e}")
    model.train()

logged_graph = False

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total_loss_dis = 0.0
    total_loss_gene = 0.0
    total_steps = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for step, batch in enumerate(pbar, start=1):
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_disease = batch["labels_disease"].to(device)
        labels_gene = batch["labels_gene"].to(device)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out = model(input_ids, attention_mask, labels_disease, labels_gene)
            loss = out["loss"]

        scaler.scale(loss).backward()
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        if out["loss_dis"] is not None:
            total_loss_dis += out["loss_dis"].item()
        if out["loss_gene"] is not None:
            total_loss_gene += out["loss_gene"].item()
        total_steps += 1

        # Log per-step scalars
        global_step = (epoch - 1) * len(train_loader) + step
        writer.add_scalar("train/loss", loss.item(), global_step)
        if out["loss_dis"] is not None:
            writer.add_scalar("train/loss_dis", out["loss_dis"].item(), global_step)
        if out["loss_gene"] is not None:
            writer.add_scalar("train/loss_gene", out["loss_gene"].item(), global_step)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
        writer.add_scalar("train/grad_norm", float(grad_norm), global_step)

        # Optionally log tokens/sec (rough proxy)
        tokens_this_batch = int(attention_mask.sum().item())
        writer.add_scalar("train/tokens_per_step", tokens_this_batch, global_step)

        if not logged_graph:
            log_graph_once()
            logged_graph = True

        # Update tqdm
        avg_loss = total_loss / max(1, total_steps)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": scheduler.get_last_lr()[0]})

    # =====================
    # Validation
    # =====================
    model.eval()
    dev_loss = 0.0
    dev_loss_dis = 0.0
    dev_loss_gene = 0.0
    acc_dis_meter = []
    acc_gene_meter = []

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Validate", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_disease = batch["labels_disease"].to(device)
            labels_gene = batch["labels_gene"].to(device)
            out = model(input_ids, attention_mask, labels_disease, labels_gene)

            dev_loss += out["loss"].item()
            dev_loss_dis += out["loss_dis"].item() if out["loss_dis"] is not None else 0.0
            dev_loss_gene += out["loss_gene"].item() if out["loss_gene"] is not None else 0.0

            # Accuracies (masked)
            acc_dis = masked_token_accuracy(out["logits_dis"], labels_disease)
            acc_gene = masked_token_accuracy(out["logits_gene"], labels_gene)
            acc_dis_meter.append(acc_dis)
            acc_gene_meter.append(acc_gene)

    dev_steps = max(1, len(dev_loader))
    dev_loss /= dev_steps
    dev_loss_dis /= dev_steps
    dev_loss_gene /= dev_steps
    dev_acc_dis = float(sum(acc_dis_meter) / max(1, len(acc_dis_meter)))
    dev_acc_gene = float(sum(acc_gene_meter) / max(1, len(acc_gene_meter)))

    # Log epoch-level metrics
    writer.add_scalar("dev/loss", dev_loss, epoch)
    writer.add_scalar("dev/loss_dis", dev_loss_dis, epoch)
    writer.add_scalar("dev/loss_gene", dev_loss_gene, epoch)
    writer.add_scalar("dev/acc_dis", dev_acc_dis, epoch)
    writer.add_scalar("dev/acc_gene", dev_acc_gene, epoch)

    # Add text summary for quick glance
    writer.add_text(
        "epoch_summary",
        f"Epoch {epoch}: dev_loss={dev_loss:.4f}, dev_acc_dis={dev_acc_dis:.4f}, dev_acc_gene={dev_acc_gene:.4f}",
        epoch,
    )

    print(
        f"Epoch {epoch}: dev_loss={dev_loss:.4f} | dis_acc={dev_acc_dis:.4f} | gene_acc={dev_acc_gene:.4f}"
    )

    # Save checkpoints
    is_best = dev_loss < best_dev_loss
    ckpt = {
        "model_state": model.state_dict(),
        "labels_dis": labels_dis,
        "labels_gene": labels_gene,
        "backbone": HF_BACKBONE,
        "tokenizer": HF_BACKBONE,
        "epoch": epoch,
        "dev_loss": dev_loss,
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

print("Training done.")


# =====================
# Inference helpers
# =====================
@torch.no_grad()
def decode_logits(logits, mask, id2label):
    ids = logits.argmax(-1).cpu().tolist()
    mask_l = mask.cpu().tolist()
    return [id2label[i] if m == 1 else "PAD" for i, m in zip(ids, mask_l)]


@torch.no_grad()
def predict_sentence(text: str, max_length: int = 256):
    model.eval()
    words = text.split()
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(enc["input_ids"], enc["attention_mask"])  # no labels for inference
    gene_tags = decode_logits(out["logits_gene"][0], enc["attention_mask"][0], id2label_gene)
    dis_tags = decode_logits(out["logits_dis"][0], enc["attention_mask"][0], id2label_dis)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return list(zip(tokens, gene_tags, dis_tags))


# =====================
# How to run TensorBoard (example):
#   tensorboard --logdir /cluster/home/gw/Backend_project/NER/tuned/runs --host 0.0.0.0 --port 6006
# =====================
