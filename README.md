# WTST
Pedestrians trajectory prediction-Tensorflow  Implementation
## Contents

- [Background](#background)
- [Preliminary](#preliminary)
	- [Dataset](#dataset)
	- [Weight](#weight)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Background

As a perception algorithm, trajectory prediction algorithm plays an important role in autonomous
driving and monitoring systems. Accurately predicting pedestrian movement information in
the future can effectively ensure pedestrian traffic safety and reduce the risk of infrastructure.
We propose a novel pedestrian trajectory prediction model based on Transformer framework
called WTST aiming to solve the challenges of multi-agent, dynamic, partially observable and
stochastic when it comes to pedestrian trajectory prediction. For the partially observable, a world
module including frame padding and trajectory padding, the data input to the network is all
pedestrians in the scene, not partial pedestrians whose trajectories only meet the length requirement.
For the dynamic, the temporal multi-head attention(TMAE&TMAD) and the temporal
multi-head interactive attention(TMIAD) are used to model motion characteristics of pedestrians
in the temporal dimension. For the multi-agent, social multi-head attention(SMAE&SMAD)
combined with dynamic graphs is adopted to capture the social behavior of all pedestrians in
the scene at each moment. For the stochastic, WTST employs bi-variate Gaussian distribution
to sample the potential paths of pedestrians. The WTST consists of embedding layer, encoder
layers, decoder layers and output layer. The embedding layer performs attention encoding for
SE-ResNet in both temporal and social dimensions. The encoder and decoder layers include
attention layers and feed forward layers, and masking techniques are introduced to eliminate
the effect of world padding. The output layer outputs the parameter information required by
the bi-variate Gaussian distribution through dimension transformation.Finally, the WTST and
baseline model are extensively experimented on the benchmark, and the overall performance of
WTST is improved by 13.16% and 13.70%. Ablation experiments find that the world module
improves performance slightly, while temporal modeling and social modeling have a significant
impact on prediction performance. Quantitative analysis shows that the masking technique can
weaken the motion tendency of stopping pedestrians to a certain extent. In addition, WTST can
give a plausible result for predicting pedestrian interaction behavior. Benefiting from multi-head
attention mechanism, WTST gives different patterns of explanation in the temporal and social
dimensions.

## Preliminary
Before entering this project, you may need to configure the environment based on `Tensorflow2.x-gpu`.
```
pip install networkx
```

### Dataset

If you want to run this project, please download the datasets and weight file from  the [Google](https://drive.google.com/drive/folders/1Dcsf1Y9MIQzZ6Th9abmNb4F9mlpZ2GpV?usp=sharingy). Then put the `ckpt_pems08` and `ckpt_taxi` into the project and crate a new file folder named `data` and put `NYC_taxi` and `PEMS08` into it. You must change the folder named `NYC_taxi->NYC`. After some tossing, you can run [data_fac.py](data_fac.py) to generate data files in `pkl format` for your training and testing, which may be a long wait. The `pkl flie` consists of 5 parts->`traind data`, `validation data`, `test data`, `multi graph`, `node2vec results`, and  `inverse_transform scalar `

### Weight
If you just want to inference and not train your own datasets, you can modify any dataset and name it `ckpt`, for example `ckpt_pems08->ckpt`


## Training
The backbone STGMT
![image](pc/framework.png)

The [layer.py](layer.py) and [framework.py](framework.py) are the most important componets in this project. Moerover, You can come up with some innovative and great ideas and you can also can change the hyperparmetes in the [Hyperparameters.py](Hyperparameters.py) if you like .Before train your own datasets, you can just change the [train.py](train.py), `line 24` you can change your datasets path from [Hyperparameters.py](Hyperparameters.py), `line 53`, l2 loss is used and `line 55`, l1 loss is used if the datasets are senstive.
So you can finally train the model by running the following command:
```
python train.py
```
You will get a new file of your own trained weights saved in `ckpt` folders.Don't worry about getting an error, even if there are weight files in the folder, they will be overwritten during training.


## Testing 
If you only want to inferrence on our dataset, it doesn't matter. Take the dataset in New York as an example, PEMS08 performs the same operation
The [test.py](test.py) is the kernel, before testing, the operation as follows
```
change the data path-> line 16
change the graph name -> line 24 
change the test epoch is up to you -> line 32
change the data you want to save -> line 66, line 72
python test.py
```
We provide three metrics: `MAE`, `RMSE`, and `SMAPE`

In the end, the terminate will show the results of `3,6,9,12` steps errors and average errors of each steps. Three tables will saved into your project `multi_error_our.csv', 'pred.csv', and  `gt.csv`


## Results
The result of the NYC prediction:



![image](pc/visual.png)

More details please see the paper!

## Contributing


At last, thank you very much for the contribution of the co-author in the article, and also thank my girlfriend for giving me the courage to pursue for a Ph.d.

## License

[MIT](LICENSE) © YanjieWen

## Preliminary
Before entering this project, you may need to configure the environment based on `Tensorflow2.x-gpu`.
```
pip install node2vec
```

### Dataset

If you want to run this project, please download the datasets and weight file from  the [Google](https://drive.google.com/drive/folders/1Dcsf1Y9MIQzZ6Th9abmNb4F9mlpZ2GpV?usp=sharingy). Then put the `ckpt_pems08` and `ckpt_taxi` into the project and crate a new file folder named `data` and put `NYC_taxi` and `PEMS08` into it. You must change the folder named `NYC_taxi->NYC`. After some tossing, you can run [data_fac.py](data_fac.py) to generate data files in `pkl format` for your training and testing, which may be a long wait. The `pkl flie` consists of 5 parts->`traind data`, `validation data`, `test data`, `multi graph`, `node2vec results`, and  `inverse_transform scalar `

### Weight
If you just want to inference and not train your own datasets, you can modify any dataset and name it `ckpt`, for example `ckpt_pems08->ckpt`


## Training
The backbone STGMT
![image](pc/framework.png)

The [layer.py](layer.py) and [framework.py](framework.py) are the most important componets in this project. Moerover, You can come up with some innovative and great ideas and you can also can change the hyperparmetes in the [Hyperparameters.py](Hyperparameters.py) if you like .Before train your own datasets, you can just change the [train.py](train.py), `line 24` you can change your datasets path from [Hyperparameters.py](Hyperparameters.py), `line 53`, l2 loss is used and `line 55`, l1 loss is used if the datasets are senstive.
So you can finally train the model by running the following command:
```
python train.py
```
You will get a new file of your own trained weights saved in `ckpt` folders.Don't worry about getting an error, even if there are weight files in the folder, they will be overwritten during training.


## Testing 
If you only want to inferrence on our dataset, it doesn't matter. Take the dataset in New York as an example, PEMS08 performs the same operation
The [test.py](test.py) is the kernel, before testing, the operation as follows
```
change the data path-> line 16
change the graph name -> line 24 
change the test epoch is up to you -> line 32
change the data you want to save -> line 66, line 72
python test.py
```
We provide three metrics: `MAE`, `RMSE`, and `SMAPE`

In the end, the terminate will show the results of `3,6,9,12` steps errors and average errors of each steps. Three tables will saved into your project `multi_error_our.csv', 'pred.csv', and  `gt.csv`


## Results
The result of the NYC prediction:



![image](pc/visual.png)

More details please see the paper!

## Contributing


At last, thank you very much for the contribution of the co-author in the article, and also thank my girlfriend for giving me the courage to pursue for a Ph.d.

## License

[MIT](LICENSE) © YanjieWen
