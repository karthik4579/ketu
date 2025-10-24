apt update && apt upgrade -y && apt install sudo zip unzip -y
sudo apt update && sudo apt upgrade -y
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates gnupg software-properties-common wget git build-essential -y
wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb	
sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2_amd64.deb
wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo apt-key add -
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
sudo apt update
sudo apt-get install cmake -y
sudo apt-get install libcurl4-openssl-dev -y
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
export CUDACXX=/usr/local/cuda/bin/nvcc
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
cd build/bin
pip install huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download mradermacher/RISE-Judge-Qwen2.5-7B-GGUF --include RISE-Judge-Qwen2.5-7B.Q4_K_M.gguf --local-dir ./model
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
	| sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
	&& echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
	| sudo tee /etc/apt/sources.list.d/ngrok.list \
	&& sudo apt update \
	&& sudo apt install ngrok -y
ngrok config add-authtoken $0
ngrok http --url=$1 http://localhost:8080 >> /dev/null &
./llama-server -m ./model/RISE-Judge-Qwen2.5-7B.Q4_K_M.gguf -c 8192 -ngl 9999 -t 8