PointCLM: A Contrastive Learning-based Framework for Multi-instance Point Cloud Registration (ECCV2022)
===
This repository represents the official implementation of the paper:
[PointCLM: A Contrastive Learning-based Framework for Multi-instance Point Cloud Registration](https://arxiv.org/abs/2209.00219)

### Instructions
This code has been tested on 
- Python 3.8, PyTorch 1.7.1, CUDA 10.2, GeForce GTX 1080Ti

#### Requirements
To create a virtual environment and install the required dependences please run:
```shell
git clone https://github.com/phdymz/PointCLM.git
conda create --name PointCLM python=3.8
conda activate PointCLM
pip install -r requirements.txt
```

### Make dataset 
#### ModelNet40
When calling code data/modelnet40.py, the dataset ModelNet40 will be automatically downloaded to the path 'DATA_DIR'. No need for extra complex processing. 


#### Scan2CAD

You need to pre-download dataset [ScanNet](https://github.com/ScanNet/ScanNet), [ShapeNet](https://www.shapenet.org/)  and [Scan2CAD](https://github.com/skanti/Scan2CAD). 
1. Offline computing benchmark. 
```shell
python make_dataset/make_scan2cad.py --scan2cad <root_scan2cad>/full_annotations.json --output <root_output> --scannet <root_scannet> --shapenet <root_shapenet>
```

2. Extracting features using fun-tuned FCGF. The details of fun-tuning are illustrated in our paper.
```shell
python make_dataset/extract_feature.py --output <above_raw_output> --weight <fun-tuned parameter> --save_root <output_data_contains_feature>
```


3. Computing correspondences using extracted features. 
```shell
python make_dataset/make_correspondence.py --save_root <above_calculated_feature>
```



### Train on ModelNet40
After creating the virtual environment and downloading the datasets, PointCLM can be trained using:
```shell
python train_modelnet40.py
```

### Train on Scan2CAD
After creating the virtual environment and processing the datasets, PointCLM can be trained using:
```shell
python train_scan2cad.py
```


### Inference
The trained model can be evaluated by:

#### ModelNet40
We provide a pre-trained weight on ModelNet40 for PointCLM in [BaiDuyun](https://pan.baidu.com/s/1QBacb5pTfbcWMdPqPtZ1uA?pwd=eccv), Password: eccv.
```shell
python eval_modelnet40.py --checkpoint_root <weight_root>
```

#### Scan2CAD
We also provide a pre-trained weight on Scan2CAD for PointCLM in [BaiDuyun](https://pan.baidu.com/s/10WplLftBL2Pfpk02mko18g?pwd=eccv), Password: eccv.
```shell
python eval_scan2cad.py --checkpoint_root <weight_root>
```


### Citation
If you find this code useful for your work or use it in your project, please consider citing:

```shell
@article{yuan2022pointclm,
  title={PointCLM: A Contrastive Learning-based Framework for Multi-instance Point Cloud Registration},
  author={Yuan, Mingzhi and Li, Zhihao and Jin, Qiuye and Chen, Xinrong and Wang, Manning},
  journal={arXiv preprint arXiv:2209.00219},
  year={2022}
}
```


### Acknowledgments
In this project we use (parts of) the official implementations of the followin works:

- [FCGF](https://github.com/chrischoy/FCGF) (Feature extraction)
- [PointDSC](https://github.com/XuyangBai/PointDSC) (SCNonlocal Module)
- [DCP](https://github.com/WangYueFt/dcp) (ModelNet40 download)
- [Scan2CAD](https://github.com/skanti/Scan2CAD) (Scan2CAD benchmark)
- [ScanNet](https://github.com/ScanNet/ScanNet) (Make dataset)
- [ShapeNet](https://www.shapenet.org/) (Make dataset)

 We thank the respective authors for open sourcing their methods. 