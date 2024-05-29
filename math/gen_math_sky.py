import pdb
import json
import random
import numpy as np
import pandas as pd
import itertools
import time
import pickle
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline
import torch

torch.manual_seed(0)
hf_model = "meta-llama/Meta-Llama-3-8B-Instruct"
hf_token = "{your-token}"
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

def generate_answer(inquiry):
    '''
    We can parellize this for multiple queries at once: ref https://huggingface.co/docs/transformers/en/llm_tutorial
    '''
    response = pipe(inquiry,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=10,
                return_full_text=False)

    return response

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets

def construct_message(agents, question, idx):

    # Use introspection in the case in which there are no other agents.
    if len(agents) == 0:
        return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

    prefix_string = "These are the recent/updated opinions from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response.".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion[0]['generated_text']
    return {"role": "assistant", "content": content}

def parse_answer(sentence):
    parts = sentence.replace(".", " ").replace("assistant"," ").split(" ")
    for part in parts[::-1]:
        try:
            answer = float(part)
            return answer
        except:
            continue


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num


if __name__ == "__main__":


    agents_list = [1,3,5,6,7,8,9]
    rounds_list = [1,2,3,4,5]
    evaluation_round = 30

    experiment_data_file = "experiment_data.csv"
    data = pd.read_csv(experiment_data_file)

    for agents in agents_list:
        for rounds in rounds_list:

            generated_description = {}
            scores = []

            for round in tqdm(range(evaluation_round)):

                a, b, c, d, e, f = np.random.randint(0, 30, size=6)

                # Generate random ints, randomize order of operators, and create equation
                a, b, c, d, e, f = np.random.randint(0, 20, 6)
                operators = ['+', '-', '*']*2
                random.shuffle(operators)
                equation = "{}{}{}{}{}{}{}{}{}".format(a, operators[0], b, operators[1], c, operators[2],
                                                       d, operators[3], e, operators[4], f)

                answer = eval(equation)

                agent_contexts = [[{"role": "user", "content": """What is the result of {}? Please do not respond with any questions.  Make sure to state your answer at the end of the response.""".format(equation)}] for agent in range(agents)]

                content = agent_contexts[0][0]['content']
                question_prompt = "We seek to find the result of {}?".format(equation)

                for round in range(rounds):
                    for i, agent_context in enumerate(agent_contexts):

                        if round != 0:
                            agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                            message = construct_message(agent_contexts_other, question_prompt, 2*round - 1)
                            agent_context.append(message)

                        completion = generate_answer(agent_context)

                        assistant_message = construct_assistant_message(completion)
                        agent_context.append(assistant_message)

                text_answers = []

                for agent_context in agent_contexts:
                    text_answer = string =  agent_context[-1]['content']
                    text_answer = text_answer.replace(",", ".")
                    text_answer = parse_answer(text_answer)

                    if text_answer is None:
                        continue

                    text_answers.append(text_answer)

                generated_description[(a, b, c, d, e, f)] = (agent_contexts, answer)

                try:
                    text_answer = most_frequent(text_answers)
                    if text_answer == answer:
                        scores.append(1)
                    else:
                        scores.append(0)
                except:
                    continue

                eval_mean, eval_std = np.mean(scores), np.std(scores) / (len(scores) ** 0.5)
                print("performance:", eval_mean, eval_std)

            pickle.dump(generated_description, open("math_agents{}_rounds{}.p".format(agents, rounds), "wb"))
            data = pd.concat([pd.DataFrame([[agents, rounds, evaluation_round, eval_mean, eval_std]], columns=data.columns), data], ignore_index=True)
            data.to_csv("experiment_data.csv", index=False)

    import pdb
    pdb.set_trace()
    print(answer)
    print(agent_context)
