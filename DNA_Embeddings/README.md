# DNA embedding extraction

## Usage

To generate DNA embeddings on the INSECT dataset, run a command similar to one of the below:

```
# BIOSCAN BERT model
python -m DNA_Embeddings.bert_extract_dna_feature --model bioscanbert --checkpoint ../data/bioscanbert/model_44.pth --output ../data/INSECT/dna_embedding_insect_bioscanbert.csv

# DNABERT
python -m DNA_Embeddings.dnabert_extract_dna_feature --model dnabert --checkpoint ../data/dnabert_pretrained --output ../data/INSECT/dna_embedding_insect_dnabert.csv

# DNABERT-2
python -m DNA_Embeddings.bert_extract_dna_feature --model dnabert2 --output ../data/INSECT/dna_embedding_insect_dnabert2.csv
```

### Model weights

- BIOSCAN BERT: model saved in BIOSCAN google drive folder
- DNABERT: downloadable from [DNABERT repository](https://github.com/jerryji1993/DNABERT)
- DNABERT-2: provided in Huggingface