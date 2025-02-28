conda create --name hugface9 python=3.12 -y

conda activate hugface9

conda install nvidia::cuda-toolkit=11.8 nvidia::pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 -y

pip install transformers accelerate

pip install git+https://github.com/huggingface/diffusers

pip install datasets peft wandb

git clone https://github.com/facebookresearch/xformers.git --branch v0.0.29 --single-branch

cd xformers

git submodule update --init --recursive

pip install -e .

cd ..