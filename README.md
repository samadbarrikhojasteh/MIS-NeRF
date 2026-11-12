
<h1 align="center">MIS-NeRF</h1>
<h2 align="center"> Neural Radiance Fields in Minimally-Invasive Surgery </h2>

<p align="center">
  <img src="assets/merged2.gif" width="100%"/>
</p>


## 1. 🚀 Installation and Environment Setup

### 📦 Dependencies

MIS-NeRF is built upon the [Nerfstudio framework](https://docs.nerf.studio/) and its [Nerfacto method](https://docs.nerf.studio/nerfology/methods/nerfacto.html). Therefore, the dependencies and system requirements of Nerfstudio also apply to MIS-NeRF.

### ⚙️ Installation

To simplify the setup process, an `install.sh` script is provided in the repository. It automates the installation of the required dependencies and the MIS-NeRF environment.

Clone the repository:

```bash
git clone https://github.com/samadbarrikhojasteh/MIS-NeRF.git 
```
navigate to its root directory:
```bash
cd MIS-NeRF 
```
and run:
```bash
chmod +x install.sh
./install.sh
```
## 2. 📂 Synthetic Liver Dataset

This repository includes a synthetic rigid liver experiment dataset consisting of RGB images, depth maps, and corresponding masks.

The RGB images and depth maps were generated using Blender, while the masks were obtained using the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything).

You can access the dataset in the [`Dataset/Synth_rigid`](Dataset/Synth_rigid) directory.

In addition, the SfM outputs and camera poses are available in the `output_sfm` folder.

If you wish to use a different dataset, you can process it using the following command:

```bash
ns-process-data images --sfm-tool colmap --data data/YOUR/DATA/PATH/images --output-dir data/YOUR/OUTPUT/PATH
```

**Note:** If your custom dataset includes a `masks` folder, it must be placed at the same directory level as the `images` folder before processing. Otherwise, SfM assumes that the images contain foreground and background regions without the provided masks.

## 3. 🏋️‍♂️ Training

To train the MIS-NeRF model, use the following command:

```bash
ns-train nerfacto --pipeline.model.predict-normals True --pipeline.model.camera-optimizer.mode off --data data/YOUR/DATA/PATH --output-dir data/YOUR/OUTPUT/PATH nerfstudio-data
```
## 4. 📊 Evaluation
To evaluate your trained model on the test dataset and compute the photometric metrics, you can run the following command:
```bash
ns-render dataset --load-config YOUR/PATH/config.yml
```
The metrics CSV file can be found in the `renders` folder.

## 📖 Citation

If you find MIS-NeRF useful, please cite our paper:
```bibtex
@article{Khojasteh2025,
  author    = {Samad Barri Khojasteh and David Fuentes-Jimenez and Daniel Pizarro and Yamid Espinel and Adrien Bartoli},
  title     = {MIS-NeRF: neural radiance fields in minimally-invasive surgery},
  journal   = {International Journal of Computer Assisted Radiology and Surgery},
  year      = {2025},
  doi       = {10.1007/s11548-025-03429-7},
  url       = {https://doi.org/10.1007/s11548-025-03429-7}
}
