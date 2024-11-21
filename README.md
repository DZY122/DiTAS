# DiTAS: Quantizing Diffusion Transformers via Enhanced Activation Smoothing (WACV 2025)
### [Paper] https://arxiv.org/abs/2409.07756
![DiTAS samples](sample.png)


## Evaluation (FID, Inception Score, etc.)

We include a [`sample_merge_TAS.py`](sample_merge_TAS.py) script which samples a large number of images from a DiT model in parallel. For the QKV and FC1 layers in DiT blocks, we can merge the smooth
ing factor of activation into the side MLP, and merge the smoothing factor of Projection layer’s activation into V’s weight. Finally, we operate on-the-fly activation smoothing for FC2 layer. And this script 
generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. For example, to sample 10K images from our pre-trained DiT-XL/2 model over `N` GPUs, run:

```bash
torchrun --nnodes=1 --nproc_per_node=1 sample_merge_TAS.py --image-size 256 --model DiT-XL/2 --num-fid-samples 10000 --act-bit 8 --weight-bit 4 --path /path/DiTAS_Model
```



## Differences from JAX

Our models were originally trained in JAX on TPUs. The weights in this repo are ported directly from the JAX models. 
There may be minor differences in results stemming from sampling with different floating point precisions. We re-evaluated 
our ported PyTorch weights at FP32, and they actually perform marginally better than sampling in JAX (2.21 FID 
versus 2.27 in the paper).


## BibTeX

```bibtex
@article{Peebles2022DiT,
  title={Scalable Diffusion Models with Transformers},
  author={William Peebles and Saining Xie},
  year={2022},
  journal={arXiv preprint arXiv:2212.09748},
}
```
