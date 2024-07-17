import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset
from utils import query_chat, initialize_question_evaluator, evaluate_question
from utils import get_embedding
import pylev
from sklearn.metrics.pairwise import cosine_similarity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="dataset to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose")
    args = parser.parse_args()

    dataset = load_dataset('json', data_files=args.dataset, split='train')

    if "corrected" not in dataset.column_names:
        model, tokenizer = initialize_question_evaluator()
        new_ds = []
        for d in tqdm(dataset):
            if d["answerable"] or d["correction"] == "":
                d["corrected"] = 0
                d["correction_entities"] = []
                d["entity_overlap"] = 0
                d["cosine_similarity"] = 0
                d["distance"] = 0
                new_ds.append(d)
                continue
            context = d["context"]
            question = d["correction"]
            if ": " in question or ":\n" in question:
                question = question.split(": ")[-1]
                question = question.split(":\n")[-1]
                question = question.replace("\"", "").strip()
            if "\n" in question:
                question = question.split("\n")[0]
            if question.count("\"") == 4:
                temp = question[:question.rfind("\"")]
                question = temp[temp.rfind("\"")+1:]
            a = d["question"].split(" ")
            b = question.split(" ")
            d["distance"] = pylev.levenshtein(a,b)
            if '?' not in question:
                d["corrected"] = 0
                d["entity_overlap"] = 0
                d["cosine_similarity"] = 0
                d["correction_entities"] = []
            else:
                question = question.split('?')[0].strip()+'?'
                d["corrected"] = evaluate_question(context, question, model, tokenizer)
                messages = [{"role": "system", "content": "You are an entity tagger. Your task is to list all the entities mentioned in the question provided by the user. Each entity should be on its own line."}, {"role": "user", "content": question}]
                out = query_chat(messages, "gpt-4", None)
                out = [one.strip() for one in out.split('\n')]
                d["correction_entities"] = out
                d["entities"] = [x.lower() for x in d["entities"]]
                d["correction_entities"] = [x.lower() for x in d["correction_entities"]]
                d["entity_overlap"] = len([x for x in d["correction_entities"] if x in d["entities"]]) / len(d["entities"])
                a_emb = get_embedding(d["question"])
                b_emb = get_embedding(question)
                d["cosine_similarity"] = cosine_similarity([a_emb], [b_emb])[0][0]
            if args.verbose:
                print(question)
                print("old entities:", d["entities"])
                print("new entities:", d["correction_entities"])
                print("answerable:", d["corrected"])
                print("entity overlap:", d["entity_overlap"])
                print("cosine similarity:", d["cosine_similarity"])
                print("edit distance:", d["distance"])
                print('-'*10)
            new_ds.append(d)

        new_ds = Dataset.from_list(new_ds)
        name = f"answerable_check_{args.dataset}"
        new_ds.to_json(name)
    else:
        new_ds = dataset
    tot = len(new_ds) - sum(new_ds["answerable"])
    refined = len([x for x in new_ds if x["correction"] != "" and x['answerable'] == 0])
    final = [1 if (x and y >= 0.5) else 0 for x, y in zip(new_ds['corrected'], new_ds['entity_overlap'])]
    print(tot, refined)
    print("corrected:", sum(new_ds["corrected"])/tot)
    print("refined corrected:", sum(new_ds["corrected"])/refined)
    print("entity overlap:", sum(new_ds["entity_overlap"])/tot)
    print("refined entity overlap:", sum(new_ds["entity_overlap"])/refined)
    print("cosine similarity:", sum(new_ds["cosine_similarity"])/tot)
    print("refined entity overlap:", sum(new_ds["cosine_similarity"])/refined)
    print("edit distance", sum(new_ds["distance"])/tot)
    print("refined edit distance:", sum(new_ds["distance"])/refined)
    print("final:", sum(final)/tot)
    print("final refined:", sum(final)/refined)

if __name__ == '__main__':
    main()
