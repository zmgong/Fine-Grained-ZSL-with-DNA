# DNA embedding extraction

## Usage

### Data Download

The INSECT data used for these scripts can be found at /project/3dlg-hcvc/bioscan/BZSL_INSECT_data.zip.

### Embedding extraction

To generate DNA embeddings on the INSECT dataset, run a command similar to one of the below:

```
# BIOSCAN BERT model
python -m DNA_Embeddings.bert_extract_dna_feature --model bioscanbert --checkpoint ../data/bioscanbert/model_44.pth --output ../data/INSECT/dna_embedding_insect_bioscanbert.csv

# DNABERT
python -m DNA_Embeddings.bert_extract_dna_feature --model dnabert --checkpoint ../data/dnabert_pretrained --output ../data/INSECT/dna_embedding_insect_dnabert.csv

# DNABERT-2
python -m DNA_Embeddings.bert_extract_dna_feature --model dnabert2 --output ../data/INSECT/dna_embedding_insect_dnabert2.csv
```

Note that for DNABERT-2, I ran into some issues with `trans_b` no longer being a supported parameter for `tl.dot`, and
in the end just disabled flash_attention in the repo by setting `flash_attn_qkvpacked_func` to `None` in bert_layers.py.
However, the code which needed to be modified is from huggingface and not part of this repository, so you may need to make
that change manually yourself. The alternative is downgrading our python and pytorch versions until the particular version
of triton we need is supported.

#### Model weights

- BIOSCAN BERT: model saved in BIOSCAN google drive folder
- DNABERT: downloadable from [DNABERT repository](https://github.com/jerryji1993/DNABERT)
- DNABERT-2: provided in Huggingface

### Fine-tuning

To fine tune the model, run the following:
```

```