
<h1 align="center">MIS-NeRF</h1>
<h2 align="center"> Neural Radiance Fields in Minimally-Invasive Surgery </h2>

<p align="center">
  <img src="assets/merged2.gif" width="100%"/>
</p>


## 1. 🚀 Installation and Environment Setup

### 📦 Dependencies
<div align="justify">

MIS-NeRF is built upon the [Nerfstudio framework](https://docs.nerf.studio/) and its [Nerfacto method](https://docs.nerf.studio/nerfology/methods/nerfacto.html). Therefore, the dependencies and system requirements of Nerfstudio also apply to MIS-NeRF.

</div>

### ⚙️ Installation
<div align="justify">

To simplify the setup process, an `install.sh` script is provided in the repository. It automates the installation of the required dependencies and the MIS-NeRF environment.

</div>
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
<div align="justify">

This repository includes a synthetic rigid liver experiment dataset consisting of RGB images, depth maps, and corresponding masks.
The RGB images and depth maps were generated using Blender, while the masks were obtained using the [SAM](https://github.com/facebookresearch/segment-anything).
You can access the dataset in the [`Synth_rigid`](Dataset/Synth_rigid) directory.
In addition, the SfM outputs and camera poses are available in the `output_sfm` folder.
If you wish to use a different dataset, you can process it using the following command:

</div>

```bash
ns-process-data images --sfm-tool colmap --data /YOUR/DATA/PATH/images --output-dir /YOUR/OUTPUT/PATH
```
<div align="justify">

**Note:** If your custom dataset includes a `mask` folder, it must be placed at the same directory level as the `images` folder before processing. Otherwise, SfM assumes that the images contain foreground and background regions without the provided masks.

</div>

## 3. 🏋️‍♂️ Training
To train the MIS-NeRF model, run the following command:

```bash
ns-train nerfacto \
    --pipeline.model.background-color random \
    --pipeline.model.predict-normals True \
    --pipeline.model.camera-optimizer.mode off \
    --data /YOUR/DATA/PATH \
    --output-dir /YOUR/OUTPUT/PATH \
    nerfstudio-data
```
Replace `/YOUR/DATA/PATH` with the directory containing your processed dataset and `/YOUR/OUTPUT/PATH` with the desired directory for saving the training outputs.
## 4. 📊 Evaluation
To evaluate your trained model on the test dataset and compute the photometric metrics, you can run the following command:
```bash
ns-render dataset --load-config /YOUR/PATH/config.yml
```
The metrics CSV file can be found in the `renders` folder.
## 5. 🔄 ICP Registration

<div align="justify">

The registration process uses a small number of corresponding landmarks (typically 4–5) between the preoperative 3D model and an intraoperative RGB image.
Before running the registration, select 4–5 landmarks on your preoperative 3D model (preferably using [MeshLab](https://www.meshlab.net/)) and save their 3D coordinates in a `.txt` file. 
An example is provided in [`preop_landmarks.txt`](Dataset/Synth_rigid/preop_landmarks.txt).

To run the ICP registration, use the following command:

</div>
```bash
python ICP/MIS_ICP.py \
    --dir /YOUR/OUTPUT/SfM/PATH \
    --source_path /YOUR/OUTPUT/RECONSTRUCTED/PATH/poisson_mesh.ply \
    --target_path /YOUR/PREOPERATIVE/PATH/liver.obj \
    --preop_landmarks /YOUR/PREOPERATIVE/LANDMARKS/PATH/preop_landmarks.txt \
    --intraop_landmarks /YOUR/INTRAOPERATIVE/LANDMARKS/PATH/intraop_landmarks.txt
```
<div align="justify">

The `--intraop_landmarks` argument is optional. If it is not provided, the script allows you to interactively select the corresponding landmarks on the first intraoperative RGB image. The selected 2D coordinates are then used by the script to obtain their corresponding 3D points from the reconstructed model and save under 
`intraop_landmarks.txt`.

</div>
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
