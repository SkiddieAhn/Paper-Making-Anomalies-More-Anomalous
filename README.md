# Making Anomalies More Anomalous: Video Anomaly Detection using New Generator and Destroyer

This repository is the ```official open-source``` of [Making Anomalies More Anomalous: Video Anomaly Detection using New Generator and Destroyer](http://)
by Seungkyun Hong, Sunghyun Ahn, Youngwan Jo and Sanghyun Park. 

Seungkyun Hong and <ins>Sunghyun Ahn</ins> are equal contributors to this work and designated as co-first authors.

## Architecture overview of F2LM Generator
This new Generator excels at predicting normal frame but struggles with abnormal one. It includes a module to **transform normal feature** in bottleneck areas, reducing its ability to generate abnormal frame.

<img width="936" alt="fig-generator" src="https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/5c4f4ab5-b12e-4e31-8acd-3899e285eed2">

## Architecture overview of Destroyer
It's a Destroyer that takes generated future frame as input, **detects low-quality regions, and destroyes them**. This enhances the abnormality in the output. We trained the Destroyer using self-supervised learning because the training data doesn't include abnormal frames.

<img width="936" alt="fig-destroyer" src="https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/ede2a4d7-fff6-4b7d-a4f5-9caa4aec44a6">

## Model Training Process
It's a **two-stage video anomaly detection method** based on <ins>unsupervised learning</ins> for the F2LM Generator and <ins>self-supervised learning</ins> for the Destroyer. Both models are individually optimized.

<img width="936" alt="fig-traing" src="https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/638c8b69-7c55-45b6-9116-fa342717ddc7">

## Results
AUC comparison with the state of the art methods. Best results are **bolded**. Best seconds are <ins>underlined</ins>.  
We compared our model with prominent papers published from 2016 to 2023.

<img width="936" alt="mama_sota" src="https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/9b214fe4-88c7-495d-aebc-903ffe73e755">

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
- Other common packages.

## Datasets
- You can specify the dataset's path by editing ```'data_root'``` in ```config.py```.
  
|     UCSD Ped2    | CUHK Avenue    |Shnaghai Tech.    |
|:------------------------:|:-----------:|:-----------:|
|[Official Site](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)|[Official Site](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)|[Official Site](https://svip-lab.github.io/dataset/campus_dataset.html)|
  
## Training
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
python eval.py --dataset={dataset_name} --trained_model=g_best_auc_{dataset_name}
```
- ```a_best_auc_{dataset_name}.pth``` contains both generator weights and destroyer weights.
```Shell
# recommended code for destroyer evaluation.
python eval.py --dataset={dataset_name} --trained_model=a_best_auc_{dataset_name}
```

## Pre-trained models
- Refer to the PyTorch tutorial and pre-download the ```deeplabv3_resnet101``` model to your environment.
- Download the FlowNetv2 weight and put it under the ```pretrained_flownet``` folder.
- Create a ```weights``` folder and put the pre-trained model weights in that folder.
- If the specified conditions in the ```Dependencies``` are different, the **AUC** may **different**.
  
|   DeepLabv3     | FlowNetv2    |Ours    |
|:--------------:|:-----------:|:-----------:|
|[PyTorch Tutorial](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)|[Google Drive](https://drive.google.com/file/d/1G3p84hzYRTCboNnJTb3iLwIPiHeNg-D_/view?usp=drive_link)|[Google Drive](https://drive.google.com/file/d/1oMopZL4qAI1dIzrABu-J-wJpE0q1ge5F/view?usp=sharing)|

## Citation
If you use our work, please consider citing:  
  
**TBD**
