import json
import os
import random
import pandas as pd  
import numpy as np
from gsm_utils import *

class Experiment(): 
    """Given a number of agents, rounds, performs an experiment. Saves out results
    """
    def __init__(self, agents, rounds, hyp):
        """Initialize experiment
        Args:
            agents      - (int)        - Number of agents used in the debate 
            rounds      - (int)        - Number of debate rounds 
            hyp         - (Namespace)  - Dict-like containing hyperparams

        Attributes: 
            gsm_dataset   - (str) - Path to the gsm dataset list of questions
            database_file - (str) - .csv-like file containing all the experiment results
            Nquestions    - (int) - Number of questions to be used for evaluation of a 
                Nagents, Nrounds combinations
            max_tokens    - (int) - Generated tokens from LLM 
        """
        self.agents = int(agents)
        self.rounds = int(rounds) 
        self.gsm_dataset = hyp.gsm_dataset_file
        self.database_file = hyp.database_file
        self.Nquestions  = hyp.Nquestions
        self.max_tokens = hyp.max_tokens

        if not os.path.exists(self.database_file): 
            df = pd.DataFrame(columns=["agents", "rounds", "Nquestions", "mean_accuracy", "std"])
            df.to_csv(self.database_file)

    def gen_gsm(self, pipe):
        """Given a model or pipe, goes through N questions and generates a debate
        Args: 
            pipe - (HuggingFace.pipe or HuggingFace.Model) - model interface to input context
                .. and generates response
        """
        agents, rounds = self.agents, self.rounds
        generated_description = {}
        questions = read_jsonl(self.gsm_dataset)
        random.shuffle(questions)

        for data in tqdm(questions[:self.Nquestions]):
            question = data['question']
            answer = data['answer']
            agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} 
                                Explain your reasoning. Your final answer should be a single numerical number, 
                                in the form \\boxed{{answer}}, at the end of your response. 
                                """.format(question)}] for agent in range(agents)]
            
            for round in range(rounds):
                for i, agent_context in enumerate(agent_contexts):
                    if round != 0:
                        agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                        message = construct_message(agent_contexts_other, question, 2*round - 1)
                        agent_context.append(message)

                    completion = generate_answer(agent_context, pipe, self.max_tokens)
                    assistant_message = construct_assistant_message(completion)
                    agent_context.append(assistant_message)

            generated_description[question] = (agent_contexts, answer)

        json.dump(generated_description, open("gsm_{}_{}.json".format(agents, rounds), "w"))
    

    def eval_gsm(self): 

        agents, rounds = self.agents, self.rounds

        response_dict = json.load(open(f"gsm_{agents}_{rounds}.json", "r"))
        questions = list(response_dict.keys())
        accuracies = []

        for question in questions:
            responses, gt = response_dict[question]
            pred_solutions = []

            for response in responses:
                pred_solution = response[-1]['content']
                pred_solutions.append(pred_solution)

            accurate = compute_accuracy(gt, pred_solutions)
            if accurate is not None:
                accuracies.append(float(accurate))
            else:
                import pdb
                pdb.set_trace()
                print(gt)

        eval_mean, eval_std = np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5)

        # Update database
        data = pd.read_csv(self.database_file)
        data = pd.concat([pd.DataFrame([[agents, rounds, self.Nquestions, eval_mean, eval_std]], 
                                       columns=data.columns), data], ignore_index=True)
        data.to_csv(self.database_file, index=False)
