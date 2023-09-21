from itertools import product
from typing import Optional

import torch
from torchtext.vocab import build_vocab_from_iterator
from transformers import AutoModel, AutoTokenizer, BertConfig, BertForMaskedLM

from dnabert.tokenization_dna import DNATokenizer
from pablo_bert_with_prediction_head import Bert_With_Prediction_Head

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


class kmer_tokenizer(object):
    def __init__(self, k, stride=1):
        self.k = k
        self.stride = stride

    def __call__(self, dna_sequence):
        tokens = []
        for i in range(0, len(dna_sequence) - self.k + 1, self.stride):
            k_mer = dna_sequence[i : i + self.k]
            tokens.append(k_mer)
        return tokens


class PadSequence(object):
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, dna_sequence):
        if len(dna_sequence) > self.max_len:
            return dna_sequence[: self.max_len]
        else:
            return dna_sequence + "N" * (self.max_len - len(dna_sequence))

        # return new_sequence


def remove_extra_pre_fix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]  # 去除 'module.' 前缀
        new_state_dict[key] = value
    return new_state_dict


def get_dnabert_tokenizer(tokenizer, pad_token, pad_token_segment_id=0, max_len=512):
    def tokenize(inp):
        """Design adapted from https://github.com/jerryji1993/DNABERT."""
        inputs = tokenizer.encode_plus(inp, max_length=max_len, add_special_tokens=True)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1 for _ in range(len(input_ids))]

        # apply padding
        padding_length = max_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + [0 for _ in range(padding_length)]
        token_type_ids = token_type_ids + [pad_token_segment_id for _ in range(padding_length)]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

    return tokenize


def load_model(args, *, k: int = 6, classification_head: bool = False, num_classes: Optional[int] = None):
    kmer_iter = (["".join(kmer)] for kmer in product("ACGT", repeat=k))
    vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>", "<CLS>", "<UNK>"])
    vocab.set_default_index(vocab["<UNK>"])
    vocab_size = len(vocab)
    max_len = 660
    pad = PadSequence(max_len)

    print("Initializing the model . . .")

    if args.model == "bioscanbert":  # FIXME: not sure what to call this ':D
        tokenizer = kmer_tokenizer(k, stride=k)
        sequence_pipeline = lambda x: [0, *vocab(tokenizer(pad(x)))]

        configuration = BertConfig(vocab_size=vocab_size, output_hidden_states=True)

        model = BertForMaskedLM(configuration)
        state_dict = torch.load(args.checkpoint)
        state_dict = remove_extra_pre_fix(state_dict)
        model.load_state_dict(state_dict)

    elif args.model == "dnabert":
        max_len = 512
        configuration = BertConfig.from_pretrained(
            pretrained_model_name_or_path=args.checkpoint, output_hidden_states=True
        )
        # tokenizer = kmer_tokenizer(k, stride=k)
        # sequence_pipeline = lambda x: vocab(tokenizer(pad(x)))
        tokenizer = DNATokenizer.from_pretrained(args.checkpoint, do_lower_case=False)
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        sequence_pipeline = get_dnabert_tokenizer(tokenizer, pad_token, pad_token_segment_id=0, max_len=max_len)

        model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=args.checkpoint, config=configuration)

    elif args.model == "dnabert2":
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        sequence_pipeline = lambda x: tokenizer(x, return_tensors="pt")["input_ids"]
        model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    else:
        raise ValueError(f"Could not parse model name: {args.model}")

    if classification_head:
        model = Bert_With_Prediction_Head(out_feature=num_classes, bert_model=model)

    model.to(device)

    print("The model has been succesfully loaded . . .")
    return model, sequence_pipeline
