# DiTAS: Quantizing Diffusion Transformers via Enhanced Activation Smoothing (WACV 2025)
### [Paper] https://arxiv.org/abs/2409.07756
![DiTAS samples](sample.png)

## Setup

First, download and set up the repo:

```bash
git clone https://github.com/DZY122/DiTAS.git
cd DiTAS
```

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate DiTAS
```
Then, choose and set the bit-width setting in [`config.py`](config.py) both for activation and weight quantization.

## Quantizing Diffusion Transformers

DiTAS provides a quantization script for DiT in [`QuantDiT.py`](QuantDiT.py). This script can be used to quantize DiT models through advanced methods of DiTAS. The output is the quantized weight checkpoints and the optimized parameters for activation quantization:

```bash
python QuantDiT.py --image-size 256 --seed 1 --model DiT-XL/2 --act-bit 8 --weight-bit 4 --num-sampling-steps 50
```


## Evaluation (FID, Inception Score, etc.)

DiTAS includes a [`sample_merge_TAS.py`](sample_merge_TAS.py) script which samples a large number of images from a DiTAS model in parallel. For the QKV and FC1 layers in DiT blocks, we can merge the smoothing factor of activation into the side MLP, and merge the smoothing factor of Projection layer’s activation into V’s weight. Finally, we operate on-the-fly activation smoothing for FC2 layer. 

This script generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. For example, to sample 10K images from our quantized DiT-XL/2 model over `N` GPUs, run:

```bash
torchrun --nnodes=1 --nproc_per_node=1 sample_merge_TAS.py --image-size 256 --model DiT-XL/2 --num-fid-samples 10000 --act-bit 8 --weight-bit 4 --path /path/DiTAS_Model
```



## BibTeX

```bibtex
@article{dong2024ditas,
  title={DiTAS: Quantizing Diffusion Transformers via Enhanced Activation Smoothing},
  author={Dong, Zhenyuan and Zhang, Sai Qian},
  journal={arXiv preprint arXiv:2409.07756},
  year={2024}
}
```
