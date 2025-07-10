# Statement

This is an example code for paper [Navigate the Unknown: Enhancing LLM Reasoning with Intrinsic Motivation Guided Exploration](https://arxiv.org/pdf/2505.17621)

The project of iMENTOR is conducted on the [TinyZero](https://github.com/Jiayi-Pan/TinyZero) environment with reinforcement learning tool [verl](https://github.com/volcengine/verl).

# iMENTOR codes

## Installation

```
conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
```

## An example for running

```
# First, please log in to your wandb with "wandb login"

# Then download the base model into ./models/
conda activate zero

# We usually store our dataset at ./data/
python ./examples/data_preprocess/countdown-4.py --local_dir {path_to_your_dataset}

# An example for iMENTOR with grpo
bash scripts/iMENTOR_countdown-4_3b.sh
```

## Note
Due to the update of our code, we moved the reward network that was originally trained on CPU in the paper to the GPU. 

However, due to the author's limited understanding of ray and distributed computing. At present, we are unable to incorporate the training of the reward network into the existing GPU resourse pool.

Therefore, this project will additionally use 1 GPU to train the reward network separately (if the GPU number written in the scripts is 4, then actually 4 + 1 = 5 GPUs are used). This action will significantly enhance the overall training efficiency. 

## Citation
```
@article{gao2025navigate,
  title={Navigate the unknown: Enhancing llm reasoning with intrinsic motivation guided exploration},
  author={Gao, Jingtong and Pan, Ling and Wang, Yejing and Zhong, Rui and Lu, Chi and Cai, Qingpeng and Jiang, Peng and Zhao, Xiangyu},
  journal={arXiv preprint arXiv:2505.17621},
  year={2025}
}
```