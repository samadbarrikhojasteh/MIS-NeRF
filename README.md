<p align="center"> Neural Radiance Fields in Minimally-Invasive Surgery </p>
<h1 align="center">MIS-NeRF</h1>

<p align="center">
  <img src="assets/rgb.gif" width="33%"/><img src="assets/Intraoperative.gif" width="33%"/><img src="assets/Preoperative.gif" width="33%"/>
</p>

<p align="center">
  <table>
    <tr>
      <td><img src="assets/rgb.gif"></td>
      <td><img src="assets/Intraoperative.gif" ></td>
      <td><img src="assets/Preoperative.gif"></td>
    </tr>
  </table>
</p>




# About


Minimally-Invasive Surgery (MIS) reduces the trauma compared to open surgery but is challenging for endophytic lesion localisation. Augmented
Reality (AR) is a promising assistance, which superimposes a preoperative 3D lesion model onto the MIS images. It requires solving the difficult problem of
3D model to MIS image registration. We propose MIS-NeRF, a Neural Radiance Field (NeRF) which provides high-fidelity intraoperative 3D reconstruction, used
to bootstrap Iterative Closest Point (ICP) registration.



## 1. Installation and Setup the environment

### Dependencies

Since MIS-NeRF is built upon the Nerfstudio framework, all requirements and dependencies of Nerfstudio remain applicable to MIS-NeRF. 
We have developed and modified version v1.0.0 of Nerfstudio, which can be accessed [here](https://github.com/nerfstudio-project/nerfstudio/releases/tag/v1.0.0).
## 2. Using the synthetic liver dataset
This repository includes a synthetic liver experiment dataset consisting of RGB images, depth maps, and corresponding masks. 
The RGB images and depth maps were generated using Blender, while the masks were produced using the Segment Anything Model ([SAM](https://github.com/facebookresearch/segment-anything)). 
method. You can access the dataset [here](Dataset).
In addition, we also put camera location ```transforms.json``` file that was provided by _SfM_ method.
If you wish to use a different dataset, you can process it using the following command:
```bash
ns-process-data images --sfm-tool colmap --data data/YOUR/DATA/PATH --output-dir data/YOUR/OUTPUT/PATH
```
Please note that if your custom dataset has _mask_ folder, it must be placed at the same directory level as the images folder.
The SfM method will utilize the mask folder if it is available. Otherwise, SfM will assume background and foreground.
## 3. Training 
To effectively train the _MIS-NeRF_ model, we recommend using the following code snippet:
```bash
# Train model
ns-train nerfacto --pipeline.model.background-color random --pipeline.model.predict-normals True --pipeline.model.camera-optimizer.mode off --data data/YOUR/DATA/PATH --output-dir data/YOUR/OUTPUT/PATH nerfstudio-data  
```




# Citation
If you find MIS-NeRF useful, please cite our paper:

```

```

# Contributors


