"""
Download and pre-tokenize a huggingface dataset.
Based on: pretokenize.py of https://github.com/Guitaricet/relora.git

Usage:
    python download_data.py --save_dir preprocessed_data --tokenizer t5-base --dataset allenai/c4 --dataset_config en --take 15000000 --text_field text --sequence_length 128
    python download_data.py --save_dir preprocessed_data --tokenizer t5-base --dataset allenai/c4 --dataset_config en --take 46000000 --text_field text --sequence_length 512
"""
import os
import time
import json
import argparse
import multiprocessing

from loguru import logger
from datasets import load_dataset, DatasetDict, Dataset, Value
from transformers import AutoTokenizer

from torch.utils.data import IterableDataset

from itertools import chain


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True, help="HuggingFace tokenizer name")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name. E.g., wikitext")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="HuggingFace dataset config name. E.g., wikitext-2-v1")
    parser.add_argument("--text_field", type=str, default="text", help="Name of the text field in the dataset")
    parser.add_argument("--sequence_length", type=int, default=2048, help="Sequence length")
    parser.add_argument("--num_cpu", type=int, default=multiprocessing.cpu_count(), help="Number of CPU cores")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the pre-tokenized dataset")
    parser.add_argument("--from_json", action='store_true',
                        help="Load from *.json.gz files. the files are specified by --dataset with wildcard. e.g. --dataset c4-0*.json.gz")
    parser.add_argument("--compress", action='store_true',
                        help="Compress to .json.gz format.")

    parser.add_argument("--take", type=int, default=None, help="Number of examples to take from the dataset")
    args = parser.parse_args(args)

    return args


def tokenize_and_chunk(
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        text_field: str,
        sequence_length: int,
        num_cpu: int = multiprocessing.cpu_count(),
):
    """
    Build data loaders for training.

    This function performs the following steps:
    1. Load the tokenizer from the pretrained "EleutherAI/gpt-neox-20b" model.
    2. Load the "openwebtext" dataset.
    3. Tokenize the dataset, adding the end-of-sentence token to each text.
    4. Process the tokenized dataset into chunks of a specified block size.

    Returns:
        Dataset: The processed dataset ready for training.
    """
    extra_map_kwargs = {"num_proc": num_cpu}  # iterable dataset does not support workers in map
    if isinstance(dataset, IterableDataset):
        extra_map_kwargs = {}

    _len_pre = len(dataset)
    # check that text_field is in dataset
    tokenized_dataset = dataset.map(
        lambda example: tokenizer([t + tokenizer.eos_token for t in example[text_field]]),
        batched=True,
        remove_columns=[text_field],
        **extra_map_kwargs,
    )
    assert "input_ids" in tokenized_dataset["train"].features
    assert len(tokenized_dataset["train"]) > 0
    logger.info(f"Tokenization finished")
    logger.info(f"\n{tokenized_dataset}")
    assert len(tokenized_dataset) == _len_pre

    block_size = sequence_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}

        total_length = len(concatenated_examples["input_ids"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
            if k != "attention_mask"  # we never pad for LM, so it's best to minimize the dataset storage
        }
        return result

    remove_columns = ["attention_mask"]
    train_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        remove_columns=remove_columns,
        **extra_map_kwargs,
    )
    logger.info(f"Chunking finished")
    logger.info(f"\n{train_dataset}")

    return train_dataset


def main(args):
    print("In main")
    logger.info("*" * 40)
    logger.info(f"Starting script with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    # --- save path -----------------------------------
    _tokenizer_name_for_save = args.tokenizer.replace("/", "_")
    cache_dir = os.path.join(args.save_dir, f"{args.dataset}_{_tokenizer_name_for_save}")
    save_path = cache_dir + f"_{args.sequence_length}"

    if args.dataset_config is not None:
        save_path = os.path.join(args.save_dir,
                                 f"{args.dataset}_{args.dataset_config}_{_tokenizer_name_for_save}_{args.sequence_length}")

    if os.path.exists(save_path):
        raise ValueError(f"Path {save_path} already exists")

    # --- tokenizer -----------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir="tokenizer" + _tokenizer_name_for_save)
    logger.info(f"Loaidng the dataset in streaming mode: {args.take is not None}")

    # --- download dataset ----------------------------
    import datasets
    if args.from_json:
        save_path = args.save_dir
        import glob
        files_path = glob.glob(args.dataset)
        dataset = datasets.load_dataset("json", data_files=files_path, cache_dir='/data/huggingface')
        # dataset = dataset.shuffle()
        # dataset = {"train": dataset}
        def cast_time_to_string(example):
            example['timestamp'] = example['timestamp'].isoformat()
            return example

        dataset_dict = {k: v.cast_column('timestamp', Value('string')) for k, v in dataset.items()}
        dataset = DatasetDict(dataset_dict)

    else:
        download_config = datasets.DownloadConfig(resume_download=True, max_retries=100,
                                                  cache_dir=cache_dir + "_download")
        dataset = load_dataset(args.dataset, args.dataset_config, streaming=args.take is not None,
                               download_config=download_config)
        if args.take is not None:
            logger.info(f"Taking {args.take} examples from the dataset")

            def take(ds, n):
                return Dataset.from_generator(lambda: (yield from ds.take(n)), cache_dir=cache_dir + "_take")

            dataset_dict = {k: take(v, args.take) for k, v in dataset.items()}
            dataset = DatasetDict(dataset_dict)

    # --- set dataset format --------------------------
    logger.info("Tokenizing and chunking the dataset")
    _time = time.time()
    dataset = tokenize_and_chunk(
        tokenizer=tokenizer,
        dataset=dataset,
        text_field=args.text_field,
        sequence_length=args.sequence_length,
        num_cpu=args.num_cpu,
    )
    _hours = (time.time() - _time) / 3600
    logger.info(f"Tokenization and chunking took {_hours:.2f} hours")

    # --- save dataset --------------------------------
    if args.compress:
        dataset['train'].to_json(save_path+".json.gz", compression="gzip")
        logger.info(f"Saved the dataset to {save_path}.json.gz")
    else:
        dataset.save_to_disk(save_path)
        logger.info(f"Saved the dataset to {save_path}")

        with open(os.path.join(save_path, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    # -------------------------------------------------
    print("Finish downloading dataset")


if __name__ == "__main__":
    print("Starting the script")
    args = parse_args()
    main(args)
