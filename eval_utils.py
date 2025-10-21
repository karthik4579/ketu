from ast import Dict, List
import re
import json_repair
import json
from llama_cpp import Llama
from torchmetrics.classification import BinaryAccuracy
import torch
from pathlib import Path
from dotenv import dotenv_values
import requests
import time
import matplotlib.pyplot as plt
import numpy as np

config_values = dotenv_values(f"{Path.cwd()}/config.env")
main_model_name = config_values["MAIN_AI_MODEL_NAME"]
llm_judge_api_url = config_values["LLM_JUDGE_API_URL"] + "/v1/chat/completions"
api_url = config_values["AI_API_URL"]
ai_api_key = config_values["AI_API_KEY"] 

with open(f"{Path.cwd()}/prompts/llm_judge_eval.txt") as llm_judge_eval:
    llm_judge_eval_prompt = llm_judge_eval.read()


def make_sequential_request(endpoint, header, instance, retry_count=8):
    for attempt in range(retry_count):
        try:
            response = requests.post(
                endpoint, headers=header, json=instance, timeout=17000
            )
            return response.json()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retry_count - 1:
                continue
            raise


def generate_responses(model_path: str, input_queries: Dict, system_prompt: str):
    # Initialize the Llama model using llama-cpp-python.
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        n_threads=8,
        temperature=0.7,
        top_k=40,
        top_p=0.95,
        repeat_penalty=1.2,
    )
    results = []
    for input_query, _ in input_queries.items():
        raw_output = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": input_query},
            ],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {"output": {"type": "string"}},
                    "required": ["output"],
                },
            },
        )
        result = raw_output["choices"][0]["message"]["content"]
        json_output = json_repair.loads(result)["output"]
        results.append(json_output)
    return results


def calculate_accuracy(predictions: list, labels: list):
    preds = torch.tensor([predictions])
    target = torch.tensor([labels])
    binary_accuracy = BinaryAccuracy()
    accuracy = binary_accuracy(preds, target)
    accuracy_percentage = accuracy.item() * 100
    return accuracy_percentage


def generate_comparison_metrics(finetuned_accuracy, unfinetuned_accuracy,metrics_save_path):
    models = ["Unfinetuned", "Finetuned"]
    accuracies = [unfinetuned_accuracy, finetuned_accuracy]
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(15, 10))
    bars = ax.bar(
        models,
        accuracies,
        color=["lightskyblue", "seagreen"],
        edgecolor="black",
        linewidth=1.5,
    )

    # Annotate bars with accuracy values
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 1,
            f"{yval}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test Set Accuracy (%)", fontsize=14)
    ax.set_title("Performance Benchmark: Fine-tuned vs Unfine-tuned", fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"{metrics_save_path}/accuracy_comparison.png", dpi=600, bbox_inches="tight")


def evaluate_model(
    finetuned_model_path: str,
    unfinetuned_model_path: str,
    test_data_path: str,
    dataset_goal: str,
):
    
    with open(test_data_path + "/test.json", "r") as f:
        test_data = json.load(f)

    with open(f"{Path.cwd()}/prompts/system_prompt_gen.txt") as system_prompt_gen_file:
        system_prompt_gen = system_prompt_gen_file.read()

    system_prompt_response = make_sequential_request(
        endpoint= f"{api_url}/v1/chat/completions",
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ai_api_key}",
        },
        instance={
            "model": main_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": system_prompt_gen.format(dataset_goal=dataset_goal),
                }
            ],
            "temperature": 0.6,
            "top_k": 40,
            "top_p": 0.95,
            "stream": False,
            "max_tokens": 4192
        },
    )
    raw_system_prompt = json_repair.loads(
        system_prompt_response["choices"][0]["message"]["content"]
    )
    system_prompt = raw_system_prompt["system_prompt"]
    with open(f"{finetuned_model_path}/system_prompt.txt", "w") as f:
        f.write(system_prompt)
        
    default_reference_scores = [1 for _ in range(len(test_data))]
    final_finetuned_scores, final_unfinetuned_scores = [], []
    finetuned_responses, unfinetuned_responses, reference_responses = [], [], []

    unfinetuned_responses.extend(
        generate_responses(
            unfinetuned_model_path,
            test_data,
            system_prompt,
        )
    )
    finetuned_responses.extend(
        generate_responses(
            finetuned_model_path + "/model_Q4_K_M.gguf",
            test_data,
            system_prompt,
        )
    )
    reference_responses.extend(list(test_data.values()))

    for reference, finetuned, unfinetuned, prompt in zip(
        reference_responses,
        finetuned_responses,
        unfinetuned_responses,
        list(test_data.keys()),
    ):
        # Evaluate the finetuned model response with the base model.
        judge_eval_prompt = llm_judge_eval_prompt.format(
            dataset_goal=dataset_goal,
            finetuned_response=finetuned,
            unfinetuned_response=unfinetuned,
            reference_response=reference,
            input_prompt=prompt,
        )
        eval_instance = {
            "model": "",
            "messages": [
                {"role": "user", "content": judge_eval_prompt},
            ],
            "temperature": 0.1,
            "stream": False,
            "max_tokens": 8192,
            "response_format": {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "finetuned_result": {"type": "integer"},
                        "base_result": {"type": "integer"},
                    },
                    "required": ["finetuned_result", "base_result"],
                },
            },
        }
        headers = {
            "Content-Type": "application/json"
        }
        raw_eval_response = make_sequential_request(
            endpoint=llm_judge_api_url, header=headers, instance=eval_instance
        )
        eval = json_repair.loads(raw_eval_response["choices"][0]["message"]["content"])
        final_finetuned_scores.append(eval["finetuned_result"])
        final_unfinetuned_scores.append(eval["base_result"])

    finetuned_accuracy = calculate_accuracy(
        final_finetuned_scores, default_reference_scores
    )
    unfinetuned_accuracy = calculate_accuracy(
        final_unfinetuned_scores, default_reference_scores
    )
    
    generate_comparison_metrics(
        finetuned_accuracy=finetuned_accuracy,
        unfinetuned_accuracy=unfinetuned_accuracy,
        metrics_save_path=finetuned_model_path,
    )
    return finetuned_model_path
