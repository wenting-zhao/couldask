# couldask
Code to reproduce experiments in paper "I Could've Asked That: Reformulating Unanswerable Questions"

To use models from OpenAI or Anthropic:
```
export OPENAI_API_KEY=[your openai api key]
export ANTHROPIC_API_KEY=[your anthropic api key]
```

The following script will reproduce all experiments in the paper:
```
for f in bandit_qa bbc qa2 reddit squad_v2 yelp
    do
        for model in gpt-3.5-turbo-0125 gpt-4-0613 meta-llama/Llama-2-7b-chat-hf mistralai/Mistral-7B-Instruct-v0.2 HuggingFaceH4/zephyr-7b-beta
            do
		python run_baseline.py --model $model --dataset $f
		python run_baseline.py --model $model --dataset $f --fewshot
		python run_baseline.py --model $model --dataset $f --cot
		python run_baseline.py --model $model --dataset $f --cot --fewshot
            done
    done
```

To evaluate the results produced from the experiments:
```
for f in out/*
    do
        echo $f
        python parse_baseline.py --dataset $f # compute detection scores
        python evaluate_new_question.py --dataset $f # compute question reformulation scores
    done
```
