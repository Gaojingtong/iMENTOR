# Statement

The project of iMENTOR is conducted on the [TinyZero](https://github.com/Jiayi-Pan/TinyZero) environment with reinforcement learning tool [verl](https://github.com/volcengine/verl).

Therefore, we retained the copyright of the TinyZero in the code.  

We guarantee that the code complies with the double-blind standard, and the institutions mentioned in the copyright have nothing to do with this paper.

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
conda activate zero
python ./examples/data_preprocess/countdown-4.py --local_dir {path_to_your_dataset}
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_iMENTOR.sh
```
