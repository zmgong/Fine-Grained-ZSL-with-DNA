# DNA embedding extraction

## DNABERT Model

To extract DNA feature embeddings using the [pretrained DNABERT model](https://github.com/jerryji1993/DNABERT), download
the pretrained weights from the link in the DNABERT Github repository and run
```
python -m DNA_Embeddings.dnabert_extract_dna_feature --checkpoint path/to/pretrained/dir
```