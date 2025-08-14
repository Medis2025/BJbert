from flair.datasets import ColumnCorpus

data_folder = "/cluster/home/gw/Backend_project/NER/dataset/huner_gene_nlm_gene"
# Flair default format: token in col 0, NER tag in col 1
columns = {0: "text", 1: "ner"}

corpus_gene = ColumnCorpus(
    data_folder=data_folder,
    column_format=columns,
    train_file="train",  # or "train.txt" if that’s the exact filename
    dev_file="dev",
    test_file="test",
)

print(corpus_gene)