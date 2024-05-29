from argparse import Namespace
import torch
import pandas as pd  
from transformers import AutoModel, AutoTokenizer, pipeline
from experiment import Experiment
from gsm_utils import *

torch.manual_seed(0)

hf_model = "meta-llama/Meta-Llama-3-8B-Instruct"
hf_token = "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"
max_new_tokens = 200
model = AutoModel.from_pretrained(hf_model, device_map="auto", load_in_4bit=True, token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(hf_model, token=hf_token)
pipe = pipeline(
    "text-generation",
    model=hf_model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

hyp = {
    "gsm_dataset_file": "test.jsonl",
    "database_file": "gsm_experiment_data.csv",
    "Nquestions": 50,
    "max_tokens": 150
}
hyp = Namespace(**hyp)

agents_list = [3,4,5,6]
rounds_list = [1,2,3,4,5,6]

for agents in agents_list:
    for rounds in rounds_list:
        exp = Experiment(agents, rounds, hyp)
        exp.gen_gsm(pipe)
        exp.eval_gsm()