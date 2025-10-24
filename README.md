# Ketu

Ketu is a AI chatbot that allows you to finetune smaller models like `Qwen/Qwen2.5-3B` and `meta-llama/Llama-3.2-3B-Instruct` for any custom text generation usecase by just chatting with it. It works by using larger models like `z-ai/glm-4.5-air` to distill knowledge from them to then create smaller focused synthetic datasets using seed data from precurated types of datasets available on huggingface using their datasets api to search and fetch seed data on the fly. It also benchmarks the finetuned model on a test split from the synthetically generated data which is then scored by a llm-as-a-judge like `R-I-S-E/RISE-Judge-Qwen2.5-7B`.

## Requirements

- `Python 3.12+`
- `uv` (optional but recommended)  [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
- `Openrouter account (recommended)` [signup here](https://openrouter.ai/)
- `groq account`[signup here](https://console.groq.com/login)
- `Huggingface account` [signup here](https://huggingface.co/join)
- `Ngrok account` [signup here](https://dashboard.ngrok.com/signup)
  
## Setup

1. Install dependencies:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt  OR  pip install -r requirements.txt
   ```
2. Copy `config.env.example` → `config.env` and update values: 
   
   - `AI_API_URL` – Target AI API endpoint
   - `AI_API_API_KEY` - Openrouter / any OpenAI compatible endpoint for the generator model
   - `AI_API_KEY_CHAT` - API key for chatbot part (uses groq for maximum response speed)
   - `MAIN_AI_MODEL` - Synthetic data genenrator model
   - `HF_API_KEY` - Huggingface api key
   - `LLM_JUDGE_API_URL` - OpenAI compatible LLM judge API URL

3. Run the llm judge script (run this on a seperate machine with a gpu atleast >= a nvidia T4):
   ```bash
   bash ./start start_llm_judge_server.sh YOUR_NGROK_AUTH_TOKEN YOUR_NGROK_FREE_DOMAIN
   ```
   This will start a llama.cpp server running the llm judge model.

## Run the app:
   ```bash
   streamlit run ui.py
   ```