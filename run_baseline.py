import argparse
import re
from tqdm import tqdm
from datasets import load_dataset, Dataset
from utils import query_chat, load_model_and_tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model to use")
    parser.add_argument("--dataset", type=str, required=True, help="dataset to use")
    parser.add_argument("--cot", action="store_true", help="whether to use chain of thought")
    parser.add_argument("--fewshot", action="store_true", help="whether to use few-shot prompting")
    args = parser.parse_args()

    model, tok = load_model_and_tokenizer(args.model)

    dataset = load_dataset("wentingzhao/couldask", args.dataset, split="test")
    fewshot = load_dataset('json', data_files="fewshot_examples/baseline.json", split='train')
    new_ds = []

    for d in tqdm(dataset):
        context = d["context"]
        question = d["question"]
        outs = []
        if args.cot and args.fewshot:
            prompt = []
            for one in fewshot:
                prompt.append({"role": "user", "content": f"{one['context']}\n\nAnswer the following question from the above article alone, and if you can't determine the answer based on the passage, say that you don’t know the answer. Let's think step by step.\n\n{one['question']}"})
                prompt.append({"role": "assistant", "content": one['cot']})
            prompt.append({"role": "user", "content": f"{context}\n\nAnswer the following question from the above article alone, and if you can't determine the answer based on the passage, say that you don’t know the answer. Let's think step by step.\n\n{question}"})
        elif args.fewshot:
            prompt = []
            for one in fewshot:
                prompt.append({"role": "user", "content": f"{one['context']}\n\nAnswer the following question from the above article alone, and if you can't determine the answer based on the passage, say that you don’t know the answer.\n\n{one['question']}"})
                prompt.append({"role": "assistant", "content": one['answer']})
            prompt.append({"role": "user", "content": f"{context}\n\nAnswer the following question from the above article alone, and if you can't determine the answer based on the passage, say that you don’t know the answer.\n\n{question}"})
        elif args.cot:
            prompt = [{"role": "user", "content": f"{context}\n\nAnswer the following question from the above article alone, and if you can't determine the answer based on the passage, say that you don’t know the answer. Let's think step by step.\n\n{question}"}]
        else:
            prompt = [{"role": "user", "content": f"{context}\n\nAnswer the following question from the above article alone, and if you can't determine the answer based on the passage, say that you don’t know the answer.\n\n{question}"}]
        out = query_chat(prompt, model=model, tokenizer=tok, temperature=1)
        outs.append(out)
        d["out"] = outs
        if "passage does not" in out or "no information" in out or "passage did not" in out or "don't know" in out or "don’t know" in out or "passage doesn't" in out or "article doesn't" in out or "article does not" in out or "article did not" in out:
            if args.cot and args.fewshot:
                prompt = []
                for one in fewshot:
                    if one['correction'] == "": continue
                    prompt.append({"role": "user", "content": f"{one['context']}\n\nAnswer the following question from the above article alone, and if you can't determine the answer based on the passage, say that you don’t know the answer. Let's think step by step.\n\n{one['question']}"})
                    prompt.append({"role": "assistant", "content": one['cot']})
                    prompt.append({"role": "user", "content": f"Can you think step by step to reason about the minimum edits you can make to reformulate the question so that the question answerable? Lay out your work."})
                    prompt.append({"role": "assistant", "content": one['cot_q']})
                prompt.append({"role": "user", "content": f"{context}\n\nAnswer the following question from the above article alone, and if you can't determine the answer based on the passage, say that you don’t know the answer. Let's think step by step.\n\n{question}"})
                prompt.append({"role": "assistant", "content": out})
                prompt.append({"role": "user", "content": f"Can you think step by step to reason about the minimum edits you can make to reformulate the question so that the question answerable? Lay out your work."})
            elif args.fewshot:
                prompt = []
                for one in fewshot:
                    if one['correction'] == "": continue
                    prompt.append({"role": "user", "content": f"{one['context']}\n\nAnswer the following question from the above article alone, and if you can't determine the answer based on the passage, say that you don’t know the answer.\n\n{one['question']}"})
                    prompt.append({"role": "assistant", "content": one["answer"]})
                    prompt.append({"role": "user", "content": "Can you minimally edit the question so that it becomes answerable?"})
                    prompt.append({"role": "assistant", "content": one['correction']})
                prompt.append({"role": "user", "content": f"{d['context']}\n\nAnswer the following question from the above article alone, and if you can't determine the answer based on the passage, say that you don’t know the answer.\n\n{d['question']}"})
                prompt.append({"role": "assistant", "content": out})
                prompt.append({"role": "user", "content": "Can you minimally edit the question so that it becomes answerable?"})
            elif args.cot:
                prompt = [{"role": "user", "content": f"{context}\n\nAnswer the following question from the above article alone, and if you can't determine the answer based on the passage, say that you don’t know the answer. Let's think step by step.\n\n{question}"}, {"role": "assistant", "content": out}, {"role": "user", "content": "Can you think step by step to reason about the minimum edits you can make to reformulate the question so that the question answerable? Lay out your work."}]
            else:
                prompt = [{"role": "user", "content": f"{context}\n\nAnswer the following question from the above article alone, and if you can't determine the answer based on the passage, say that you don’t know the answer.\n\n{question}"}, {"role": "assistant", "content": out}, {"role": "user", "content": "Can you minimally edit the question so that it becomes answerable?"}]
            out = query_chat(prompt, model=model, tokenizer=tok, temperature=1)
            d["correction"] = out
        else:
            d["correction"] = ""
        new_ds.append(d)

    new_ds = Dataset.from_list(new_ds)
    args.model = args.model.split('/')[-1]
    name = f"out/{args.dataset.split('/')[-1].replace('.json', '')}_{args.model}_baseline.json"
    if args.fewshot:
        name = name.replace('.json', f'_fewshot.json')
    if args.cot:
        name = name.replace('.json', f'_cot.json')
    new_ds.to_json(name)

if __name__ == '__main__':
    main()
