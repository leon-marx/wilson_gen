git clone https://github.com/CompVis/latent-diffusion.git

git clone https://github.com/Stability-AI/stablediffusion.git

conda create --name combined_env python=3.8.5 -y

conda activate combined_env

conda install --file conda_requirements.txt -y -v

pip install -r pip_requirements.txt -v

pip install -v -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers

pip install -v -e git+https://github.com/openai/CLIP.git@main#egg=clip

pip install -v -e ./latent-diffusion

pip install -v -e ./stablediffusion

rm -rf latent-diffusion

rm -rf stablediffusion

conda install gxx_linux-64 -y

conda install cmake -y

pip install inplace-abn --no-cache-dir

git clone https://github.com/facebookresearch/xformers.git --branch v0.0.16 --single-branch

cd xformers

git submodule update --init --recursive

pip install -e .