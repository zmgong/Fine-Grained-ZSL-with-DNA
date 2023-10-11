# DNA embedding extraction

## Usage

### Data Download

The INSECT data used for these scripts can be found at /project/3dlg-hcvc/bioscan/BZSL_INSECT_data.zip.

### Embedding extraction

To generate DNA embeddings on the INSECT dataset, run a command similar to one of the below:

```
# BarcodeBERT model
python bert_extract_dna_feature.py --input_path ../../data/INSECT/res101.mat --model barcodebert --checkpoint ../../data/bioscanbert/latest_model_5mer.pth --output ../../data/INSECT/embeddings/dna_embedding_insect_barcodebert.csv -k 5

# DNABERT
python bert_extract_dna_feature.py --input_path ../../data/INSECT/res101.mat --model dnabert --checkpoint ../../data/dnabert_pretrained --output ../../data/INSECT/embeddings/dna_embedding_insect_dnabert.csv -k 6

# DNABERT-2
python bert_extract_dna_feature.py --input_path ../../data/INSECT/res101.mat --model dnabert2 --output ../../data/INSECT/embeddings/dna_embedding_insect_dnabert2.csv
```

Note that for DNABERT-2, I ran into some issues with `trans_b` no longer being a supported parameter for `tl.dot`, and
in the end just disabled flash_attention in the repo by setting `flash_attn_qkvpacked_func` to `None` in bert_layers.py.
However, the code which needed to be modified is from huggingface and not part of this repository, so you may need to make
that change manually yourself. The alternative is downgrading our python and pytorch versions until the particular version
of triton we need is supported.

### Fine-tuning

To fine tune the model (BarcodeBERT or DNABERT), run the following:
```
# BarcodeBERT
python supervised_learning.py --input_path ../../data/INSECT/res101.mat --model barcodebert --output_dir path/to/output/ --n_epoch 12

# DNABERT
python supervised_learning.py --input_path ../../data/INSECT/res101.mat --model dnabert --output_dir path/to/output/ --n_epoch 12
```

For DNABERT-2, you will need to use the [DNABERT-2 repository](https://github.com/Zhihan1996/DNABERT_2) and apply 
fine-tuning with the data files (`train.csv` and `dev.csv`) created at 
`/project/3dlg-hcvc/bioscan/bzsl/dnabert2_fine_tuning/`.
