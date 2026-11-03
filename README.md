# ReGDiff
Code base for ReGDiff: Guided Diffusion in Regulated Latent Space for Exploring Metamaterial Voxel Geometry (NeurIPS 2026). [[arXiv]](https://arxiv.org/abs/2609.34231)

## Dataset
The ReGDiff MetaTruss 10k dataset is hosted on Hugging Face: [wzhan24/ReGDiff_MetaTruss_10k](https://huggingface.co/datasets/wzhan24/ReGDiff_MetaTruss_10k). It contains 10,000 truss-lattice metamaterial unit cells, each voxelized as a binary 48×48×48 grid, stored bit-packed and compressed in a single 17.8 MB file.

Download and load it with `load_packed` from `convert.py`:
```python
from huggingface_hub import hf_hub_download
from convert import load_packed

path = hf_hub_download("wzhan24/ReGDiff_MetaTruss_10k", "ReGDiff_MetaTruss_10k.npz", repo_type="dataset")
ini_data = load_packed(path)  # uint8, shape (10000, 48, 48, 48), values in {0, 1}
```
`main.ipynb` already loads the data this way. It converts the cubes to float64 with `.astype(np.float64)`, because the negative-sample augmentation writes Gaussian noise into them. `huggingface_hub` can be installed with `pip install huggingface_hub`.

## Implementation Details
### Model Architecture
The encoder and decoder we use are two transformers of the same structure, which have 4 layers and whose model dimension is by default 128. The latent space is also set to be 128-dimensional. The voxel data are decomposed into patches with the size of 8^3, and flattened as the input to the encoder. After the encoder E and before the decoder D, there is each an 2-layer multilayer-perceptron (MLP) to resize the data to and from 128-dimensional. The latent diffusion model we use has a backbone of MLP which has 16 layers and a model dimension of 512, with residual links connecting adjacent layers.

### Train Scheme
The autoencoder is trained with RAS regulation. To enable this operation, we have to construct a negative dataset and combine it with the initial positive dataset. The negative data are created by noising each positive sample. We randomly select an eighth of the voxels and substitute them to void or Gaussian noise or an eighth of another positive sample, or simply add Gaussian to the initial values. The diffusion model is trained following the DDPM paradigm, and the SRR mechanism only functions in inference stage.

### Other Details
The clustering process in SRR uses the vanilla k-means clustering. The number of clusters is set to 40 by default. This hyperparameter can also be tuned, e.g., from 30 to 50.
The directly generated voxel results are of continuous value. We use a default threshold of 0.5 to binarize the direct results. This hyperparameter can also be tuned, e.g., from 0.4 to 0.6.
When synthesizing the negative samples, we corrupt an eighth (randomly sampled) of the clean sample with noise, or structures from other samples, or void.
When computing novelty, we set a quality threshold (default 0.5). Only when a sample's quality scores are all above the threshold, we calculate the novelty score of this sample. Else its novelty score is set to be 0, considered as a failure sample. 


## Implementation Details of Baselines Used in ReGDiff Paper
* DiT-3D has a point-cloud output space, and the point-cloud sample are devoxelized from voxel space, which means the generation process itself still targets for voxel generation. We therefore use the generated voxel sample as the output, keeping the generation process intact.
* Y. Yang et al (2024) is targeted for voxel-based metamaterial generation, and therefore can be directly applied to our problem setting.
* XCube aims for voxelized 3D generation, and therefore can be readily applied to our problem setting.
* The structure generation part of Trellis which targets for voxel generation can be directly applied to our problem setting directly.
* 3D-CDM is also designed for voxel-based metamaterial generation, and therefore can be directly applied to our problem setting.

## Citation
ReGDiff has been accepted to NeurIPS 2026. Until the proceedings version is available, please cite the arXiv version:
```bibtex
@article{zhan2026regdiff,
  title   = {ReGDiff: Guided Diffusion in Regulated Latent Space for Exploring Metamaterial Voxel Geometry},
  author  = {Zhan, W. and Chen, J. and Fu, D. and Zhou, D.},
  journal = {arXiv preprint arXiv:2609.34231},
  year    = {2026}
}
```
