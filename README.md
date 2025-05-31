<!--HEADER-->
<h1 align="center"> Multinomial Logistic Regression | 
 <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://cdn.simpleicons.org/42/white">
  <img alt="42" width=40 align="center" src="https://cdn.simpleicons.org/42/Black">
 </picture>
 Advanced 
  <!-- <img alt="Complete" src="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/dark-theme/complete.svg"> -->
</h1>
<!--FINISH HEADER-->

<!--MINI DESCRIPTION-->
> 

![]()
<!--
> [!IMPORTANT]  
> When refering in the project about theta0($\theta_0$) and theta1($\theta_1$) in the project:
> * theta0($\theta_0$) is the intercept, and can be used interchangeably with the term "intercept." It represents the value of 𝑦 when 𝑥=0.
> * theta1($\theta_1$) is the slope, and can be used interchangeably with the term "slope." It represents how much 𝑦 changes for each unit increase in 𝑥.
-->
### Install Dependencies
```bash
. ./install.sh
```
### Multinomial Logistic Regression Train Program
Computes a Linear Regression using Gradient Descend Algorithm with the dataset specified.
* use ```--dataset``` or ```-d``` to specify a dataset
* use ```--target``` or ```-t``` to open a graph window with the result
* use ```--output``` or ```-o``` to save the model result with a specific filename (default: model.json) 

#### Features
* use ```--features FEATURE1 FEATURE2 FEATURE3``` or  ```-f FEATURE1 FEATURE2 FEATURE3``` to specify the features names corresponding to the column on the dataset to use for training
* use ```--features_file FEATURES_FILE``` or ```-fl FEATURES_FILE``` to specify the path to a file containing a JSON list with the features names  

#### Gradient Descend Types
* use ```--batch``` or ```-b``` for Batch Gradient Descent (default).  
* use ```--stochastic``` or ```-st``` for Stochastic Gradient Descent.
* use ```--mini_batch [BATCH_NUMBER]``` or ```-mb [BATCH_NUMBER]``` (default batch size:32) for Mini-Batch Gradient Descent and specify a batch size for it. (default: 32)
##### for more information about options use:
```bash
python3 logreg_train.py -h
```
Try:
```bash
python3 logreg_train.py -d ../datasets/dataset_train.csv -t 'Hogwarts House' -fl example_features_list.json
```

> [!NOTE]
> The train program outputs a file by default called ```model.json``` 
   
<!--
### Multinomial Predictor Program
This program calculates the predicted value of Y based on a given X value using a simple linear equation \( Y = $theta_0$ + $theta_1$ · X \)

* use --theta0 or -t0 to specify the theta0 or intercept
* use --theta1 or -t1 to specify the theta1 or slope
* use --json or -j for input a json with theta0 and theta1 result from the previous program
##### for more information about options use:
```bash
python3 linear_predictor.py -h
```
Try:
```bash
python3 linear_predictor.py --theta0 8474.34137591075 --theta1 -0.021199045602042395
```-->
