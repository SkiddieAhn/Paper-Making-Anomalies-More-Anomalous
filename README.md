# Making Anomalies More Anomalous: Video Anomaly Detection Using a Novel Generator and Destroyer (IEEE Access)

[![Paper](https://img.shields.io/badge/Paper-PDF-DO9874)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10462109)
[![Project](https://img.shields.io/badge/Project-Website(ENG)-87CEEB)](https://shacoding.com/2024/03/15/making-anomalies-more-anomalous-video-anomaly-detection-using-a-novel-generator-and-destroyer-ieee-access/)
[![slides](https://img.shields.io/badge/Presentation-Slides(KOR)-B762C1)](https://shacoding.com/wp-content/uploads/2024/03/%EB%B0%9C%ED%91%9C-%EC%9E%90%EB%A3%8C-F2LM-%EA%B8%B0%EB%B0%98%EC%9D%98-%EB%B9%84%EB%94%94%EC%98%A4-%EC%9D%B4%EC%83%81-%ED%83%90%EC%A7%80-%EB%B0%A9%EB%B2%95-%EB%B0%8F-%EC%9E%A5%EC%B9%98.pdf)

This repository is the ```official open-source``` of [Making Anomalies More Anomalous: Video Anomaly Detection Using a Novel Generator and Destroyer](https://ieeexplore.ieee.org/document/10462109/)
by Seungkyun Hong*, Sunghyun Ahn*, Youngwan Jo and Sanghyun Park. ```(*equally contributed)```

## 📣 News
* **[2024/02/26]** **MAMA** codes & weights are released!
* **[2024/02/26]** Our **MAMA** paper has been published in IEEE Access!

## Architecture overview of F2LM Generator
This new Generator excels at predicting normal frame but struggles with abnormal one. It includes a module to **transform frame feature to label and motion feature** in bottleneck areas, reducing its ability to generate abnormal frame.
If a frame with abnormal objects or behavior is inputted, the transformation from frame features to other features becomes challenging, incurring a penalty in predicting future frames.

<img width="936" alt="fig-generator" src="https://github.com/SkiddieAhn/Paper-Making-Anomalies-More-Anomalous/assets/52392658/ca3bd86b-e2e5-40c7-8b2a-ee084e606360">

## Architecture overview of Destroyer
It's a Destroyer that takes generated future frame as input, **detects low-quality regions, and destroyes them**. This enhances the abnormality in the output. We trained the Destroyer using self-supervised learning because the training data doesn't include abnormal frames.

<img width="936" alt="fig-destroyer" src="https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/ede2a4d7-fff6-4b7d-a4f5-9caa4aec44a6">

## Model Training Process
It's a **two-stage video anomaly detection method** based on <ins>unsupervised learning</ins> for the F2LM Generator and <ins>self-supervised learning</ins> for the Destroyer. Both models are individually optimized.

<img width="936" alt="fig-traing" src="https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/638c8b69-7c55-45b6-9116-fa342717ddc7">

## Results
**AUC and EER comparison** with the state-of-the-art methods on ```UCSD Ped2```, ```CUHK Avenue```, and ```Shanghai Tech.```
Best results are **bolded**. Best seconds are <ins>underlined</ins>. We compared our model with prominent papers published from 2018 to 2023.

<img width="936" alt="results" src="https://github.com/SkiddieAhn/Paper-Making-Anomalies-More-Anomalous/assets/52392658/0254fbf8-b965-45b5-bc7e-8eeacf0c05f6">

## Qualitative Comparison
The Destroyer model enhances abnormality by destroying abnormal areas, resulting in a **larger gap in Anomaly Scores** between normal and abnormal data and an increased AUC.

<img width="936" alt="mama_visualization" src="https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/101d5544-99e9-49b6-8d83-85f8c4e1eded">


## Dependencies
- python >= 3.8.  
- torch = 1.11.0+cu113
- torchvision = 0.12.0+cu113
- scikit-learn = 1.0.2.
- tensorboardX
- opencv-python  
- matplotlib
- einops  
- timm
- scipy
- Other common packages.

## Datasets
- You can specify the dataset's path by editing ```'data_root'``` in ```config.py```.
  
|     UCSD Ped2    | CUHK Avenue    |Shnaghai Tech.    |
|:------------------------:|:-----------:|:-----------:|
|[Google Drive](https://drive.google.com/file/d/1UZwAFTbFec7-tcNY0M4AIEix3vO86Eda/view?usp=sharing)|[Github Page](https://github.com/SkiddieAhn/Paper-VideoPatchCore/blob/main/DATA_README.md)|[Github Page](https://github.com/SkiddieAhn/Paper-VideoPatchCore/blob/main/DATA_README.md)|
  
## Training
- Navigate to the ```F2LM_Generator``` directory and enter the following command.
- When training starts, the ```tensorboard_log```, ```weights```, and ```results``` folders are automatically created.
- All saved models are located in the ```weights``` folder.
- You can input ```dataset_name``` as one of the following choices: **ped2**, **avenue**, **shanghai**.
  
```Shell
# default option for generator training.
python train.py --dataset={dataset_name} 
# change 'seed'.
python train.py --dataset={dataset_name} --manualseed=50
# change 'max iteration'.
python train.py --dataset={dataset_name} --iters=60000
# change 'model save interval'.
python train.py --dataset={dataset_name} --save_interval=10000
# change 'validation interval'.
python train.py --dataset={dataset_name} --val_interval=1000
# Continue training with latest model
python train.py --dataset={dataset_name} --resume=latest_{dataset_name}
```
- Navigate to the ```Destroyer``` directory and enter the following command.
- Before training, save the pretrained Generator weights in the ```Destroyer/weights``` directory.
- When training the Destroyer, we set ```iters``` to **15,000** and ```val_interval``` to **100**.

```Shell
# destroyer training with pre-trained generator model.
python train.py --dataset={dataset_name} --resume=g_best_auc_{dataset_name} --iters=15000 --val_interval=100
```
- Tensorboard visualization

```Shell
# check losses and psnr while training.
tensorboard --logdir=tensorboard_log/{dataset_name}_bs{batch_size}
```

## Evaluation
- ```g_best_auc_{dataset_name}.pth``` contains only generator weights.
```Shell
# recommended code for generator evaluation.
python eval.py --dataset={dataset_name} --trained_model=g_best_auc_{dataset_name} --show_status=True
```
- ```a_best_auc_{dataset_name}.pth``` contains both generator weights and destroyer weights.
```Shell
# recommended code for destroyer evaluation.
python eval.py --dataset={dataset_name} --trained_model=a_best_auc_{dataset_name} --show_status=True
```

## Pre-trained models
- Refer to the PyTorch tutorial and pre-download the ```deeplabv3_resnet101``` model to your environment.
- Download the FlowNetv2 weight and put it under the ```F2LM_Generator/pretrained_flownet``` and ```Destroyer/pretrained_flownet``` folders.
- Create a ```weights``` folder and put the pre-trained model weights in that folder.
- If the specified conditions in the ```Dependencies``` are different, the **AUC** may **different**.
  
|   DeepLabv3     | FlowNetv2    |Ours    |
|:--------------:|:-----------:|:-----------:|
|[PyTorch Tutorial](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)|[Google Drive](https://drive.google.com/file/d/1QYlo-ebSQ58KmVFhJIkXDNILCNF7Gd6B/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1XievE92zPEUGNZX6LGlwzndyejGB6uKA/view?usp=sharing)|

## Additional material
We observed a **1.6% performance improvement** on the ```UCSD Ped2``` dataset by applying a <ins>Gaussian 1D filter</ins> to the anomaly score in our model. However, we chose not to conduct performance comparisons in the paper to maintain fairness.

<img width="750" alt="auc-gaussian" src="https://github.com/SkiddieAhn/Paper-Making-Anomalies-More-Anomalous/assets/52392658/1e364b5c-a69a-46a7-bd60-cbf775532a99">

- You can evaluate performance by using the following command.
```Shell
# recommended code for destroyer evaluation with 1-D gaussian filter.
python eval.py --dataset={dataset_name} --trained_model=a_best_auc_{dataset_name} --gaussian=True --show_status=True
```

## Citation
If you use our work, please consider citing:  
```Shell
@article{Hong2024MakingAM,
  title={Making Anomalies More Anomalous: Video Anomaly Detection Using a Novel Generator and Destroyer}, 
  author={Seungkyun Hong and Sunghyun Ahn and Youngwan Jo and Sanghyun Park},
  journal={IEEE Access},
  year={2024},
  volume={12},
  pages={36712-36726},
}

```

## Contact
Should you have any question, please create an issue on this repository or contact me at skd@yonsei.ac.kr.
