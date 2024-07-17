import sys
import time
import numpy as np
import torch
import os
import openai
import anthropic
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
anthropic_client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

def initialize_question_evaluator():
    model = AutoModelForSequenceClassification.from_pretrained("wentingzhao/question-evaluator")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("wentingzhao/question-evaluator")
    return model, tokenizer

def evaluate_question(context, question, model, tokenizer):
    inputs = tokenizer(f"{context}\n{question}", return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return predicted_class_id


def query_chat(messages, model, tokenizer, temperature=1, max_tokens=512):
    if isinstance(model, str):
        error = True
        while error:
            try:
                if 'gpt' in model:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    output = response.choices[0].message.content.strip()
                elif 'claude' in model:
                    message = anthropic_client.messages.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    output = message.content[0].text.strip()
                else:
                    raise NotImplementedError()
                error = False
            except openai._exceptions.OpenAIError as e:
                if 'context_length_exceeded' in str(e):
                    if len(messages) > 1:
                        messages = messages[-1:]
                    else:
                        messages[-1]['content'] = messages[-1]['content'][:int(0.9*len(messages[-1]['content']))]
                time.sleep(5)
                print(type(e), e)
    else:
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        attn_mask = torch.ones_like(tokenized_chat, device=device)
        with torch.no_grad():
            outputs = model.generate(tokenized_chat, attention_mask=attn_mask, max_new_tokens=max_tokens, temperature=temperature, do_sample=True)
        output = tokenizer.decode(outputs[0][len(tokenized_chat[0]):], clean_up_tokenization_spaces=True, skip_special_tokens=True).strip()
    return output

def compute_metrics(label, predicted):
    res = accuracy_score(label, predicted)
    print("Accuracy:", res*100)
    res = f1_score(label, predicted, pos_label=1)
    print("Answerable F1:", res*100)
    res = precision_score(label, predicted, pos_label=1)
    print("Answerable Precision:", res*100)
    res = recall_score(label, predicted, pos_label=1)
    print("Answerable Recall:", res*100)
    res = f1_score(label, predicted, pos_label=0)
    print("Unanswerable F1:", res*100)
    res = precision_score(label, predicted, pos_label=0)
    print("Unanswerable Precision:", res*100)
    res = recall_score(label, predicted, pos_label=0)
    print("Unanswerable Recall:", res*100)

def get_question_category(ds):
    types = ['what', 'who', 'how', 'when', 'which', 'where', 'why']
    qtypes = []
    for d in ds:
        q = d["question"].lower()
        for t in types:
            if t in q:
                qtypes.append(t)
                break
        else:
            qtypes.append("other")
            print(f"invalid question: {q}")
    return ds.add_column(name="qtype", column=qtypes)

def load_model_and_tokenizer(name):
    if "gpt" not in name and "claude" not in name:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model = name
        tokenizer = ""
    return model, tokenizer

def get_embedding(text, model="text-embedding-3-large"):
    error = True
    while error:
        try:
            out = client.embeddings.create(input = [text], model=model).data[0].embedding
            error = False
        except openai._exceptions.OpenAIError as e:
            print(type(e), e)
            time.sleep(1)
    return out
