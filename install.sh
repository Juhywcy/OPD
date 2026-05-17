pip install "sglang[all]==0.5.2" --no-cache-dir && pip install torch-memory-saver --no-cache-dir

pip install --no-cache-dir "vllm==0.11.0"

pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard 

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

mkdir third_party

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Juhywcy/verl-whl --local-dir third_party

pip install --no-cache-dir third_party/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install --no-cache-dir flashinfer-python==0.3.1

pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

pip install -e verl
pip install swanlab matplotlib gpustat

pip install latex2sympy2_extended math_verify

export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir /home/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

huggingface-cli download --resume-download hbx/JustRL-DeepSeek-1.5B --local-dir /home/models/hbx/JustRL-DeepSeek-1.5B