import os
import uuid
import json
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import shutil

def finetune_model(model_name, dataset_path, dataset_type):
    if dataset_type == "text_only":
        supported_models = {
            "Qwen2.5-3B": "unsloth/Qwen2.5-3B-Instruct",
            "Llama-3.2B-3B": "unsloth/Llama-3.2-3B-Instruct"
        }

        if model_name not in supported_models:
            raise ValueError(
                f"Model {model_name} not supported. Choose from {list(supported_models.keys())}"
            )

        with open(os.path.join(dataset_path, "train.json"), "r") as f:
            data = json.load(f)

        formatted_data = []
        for prompt, response in data.items():
            formatted_data.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ]
                }
            )

        model, tokenizer = FastLanguageModel.from_pretrained(
            supported_models[model_name],
            max_seq_length=2048,
            load_in_4bit=True,
            device_map="auto",
        )

        chat_template = "chatml" if "Qwen" in model_name else "llama-3.1"
        tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

        dataset = Dataset.from_list(formatted_data)

        def formatting_func(examples):
            formatted = []
            eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else "<|endofresponse|>"
            for msg in examples["messages"]:
                messages = [{"role": m["role"], "content": m["content"]} for m in msg]
                if messages[-1]["role"] == "assistant":
                    if not messages[-1]["content"].strip().endswith(eos_token):
                        messages[-1]["content"] = messages[-1]["content"].strip() + " " + eos_token
                formatted.append(messages)
            texts = [tokenizer.apply_chat_template(conv, tokenize=False) for conv in formatted]
            return {"text": texts}

        dataset = dataset.map(
            formatting_func, batched=True, remove_columns=dataset.column_names
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        training_args = TrainingArguments(
            output_dir="outputs",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-5,
            warmup_steps=20,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_strategy="steps",
            save_steps=50,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            packing=False,
        )

        if "Qwen" in model_name:
            instruction_part = "<|im_start|>user\n"
            response_part = "<|im_start|>assistant\n"
        else:
            instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
            response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"

        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )

        trainer.train()

        model_dir_name = f"{model_name}_{str(uuid.uuid4())[:8]}"
        os.makedirs(f"finetuned_models/{model_dir_name}", exist_ok=True)
        model_path = os.path.join("finetuned_models", model_dir_name)
        model.save_pretrained_gguf(
            "finetuned_models/temp",
            tokenizer,
            quantization_method="q4_k_m",
            maximum_memory_usage=0.45
        )
        shutil.move(
            "finetuned_models/temp/unsloth.Q4_K_M.gguf",
            model_path + "/model_Q4_K_M.gguf",
        )
        shutil.rmtree("./finetuned_models/temp")
        shutil.rmtree("outputs")
        return model_path