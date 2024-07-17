import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset
from utils import query_chat, compute_metrics
from collections import Counter
from random import shuffle

def evaluate(dataset, verbose=False):
    new_ds = []
    for d in dataset:
        context = d["context"]
        question = d["question"]
        outs = d["out"]
        out = outs[0]
        if "passage does not" in out or "no information" in out or "passage did not" in out or "don't know" in out or "don’t know" in out or "passage doesn't" in out or "article doesn't" in out or "article does not" in out or "article did not" in out:
            d["predicted"] = 0
        else:
            d["predicted"] = 1
        new_ds.append(d)
        if verbose and d["answerable"] != d["predicted"]:
            print(context)
            print(question)
            print(out)
            print("="*100)
    new_ds = Dataset.from_list(new_ds)
    return new_ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="result to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose")
    args = parser.parse_args()

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    new_ds = evaluate(dataset, args.verbose)
    compute_metrics(new_ds['answerable'], new_ds['predicted'])


if __name__ == '__main__':
    main()
