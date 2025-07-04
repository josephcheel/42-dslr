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


## Table of Contents
- [Introduction](#introduction)
- [Install Dependencies](#install-dependencies)
- [Usage](#usage)
  - [Train Program](#multinomial-logistic-regression-train-program)
  - [Predictor Program](#multinomial-predictor-program)
<!--
- [Contributing](#contributing)
- [License](#license)
-->
![]()
<!--
> [!IMPORTANT]  
> When refering in the project about theta0($\theta_0$) and theta1($\theta_1$) in the project:
> * theta0($\theta_0$) is the intercept, and can be used interchangeably with the term "intercept." It represents the value of 𝑦 when 𝑥=0.
> * theta1($\theta_1$) is the slope, and can be used interchangeably with the term "slope." It represents how much 𝑦 changes for each unit increase in 𝑥.
-->
## Introduction
## Install Dependencies
```bash
. ./install.sh
```
## Usage
### Multinomial Logistic Regression Train Program
Computes a Multinomonal Regression using Gradient Descend Algorithm with the dataset specified.

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
#### for more information about options use:
```bash
python3 logreg_train.py -h
```
Try:
```bash
python3 logreg_train.py -d ../datasets/dataset_train.csv -t 'Hogwarts House' -fl example_features_list.json
```

###### dataset_train.csv
```bash
Index  Hogwarts House  First Name  Last Name   Birthday    Best Hand  Arithmancy  Astronomy            Herbology            Defense Against the Dark Arts  Divination           Muggle Studies       Ancient Runes        History of Magic     Transfiguration     Potions               Care of Magical Creatures  Charms               Flying
0      Ravenclaw       Tamara      Hsu         2000-03-30  Left       58384.0     -487.88608595139016  5.727180298550763    4.8788608595139005             4.7219999999999995   272.0358314131986    532.4842261151226    5.231058287281048    1039.7882807428462  3.7903690663529614    0.7159391270136213         -232.79405           -26.89
1      Slytherin       Erich       Paredes     1999-10-14  Right      67239.0     -552.0605073421984   -5.987445780050746   5.520605073421985              -5.612               -487.3405572673422   367.7603030171392    4.107170286816076    1058.9445920642218  7.248741976146588     0.091674183916857          -252.18425           -113.45
2      Ravenclaw       Stephany    Braun       1999-11-03  Left       23702.0     -366.0761168823237   7.7250166064392305   3.6607611688232367             6.14                 664.8935212343011    602.5852838484592    3.5555789956034967   1088.0883479121803  8.728530920939827     -0.5153268462809037        -227.34265           30.42
3      Gryffindor      Vesta       Mcmichael   2000-08-19  Left       32667.0     697.742808842469     -6.4972144445985505  -6.9774280884246895            4.026                -537.0011283872882   523.9821331934736    -4.8096366069645935  920.3914493107919   0.8219105005879808    -0.014040417239052931      -256.84675           200.64
4      Gryffindor      Gaston      Gibbs       1998-09-27  Left       60158.0     436.7752035539525    -7.820623052454388   2.2359999999999998             -444.2625366004496   599.3245143172293    -3.4443765754165385  937.4347240534976    4.311065821291761   -0.2640700765443832   -256.3873                  157.98
5      Slytherin       Corrine     Hammond     1999-04-04  Right      21209.0     -613.6871603822729   -4.289196726941419   6.136871603822727              -6.5920000000000005  -440.99770426820817  396.20180391410247   5.3802859494804585   1052.8451637299704  11.751212035101073    1.049894068203692          -247.94548999999998  -34.69
6      Gryffindor      Tom         Guido       2000-09-30  Left       49167.0     628.0460512248516    -4.861976240490781   -6.280460512248515             -926.8925116349667   583.7424423327342    -7.322486416427907   923.5395732944658    1.6466661386700716  0.1530218296077356    -257.83447                 261.55
7      Hufflepuff      Alicia      Hayward     1997-07-08  Right      33010.0     411.4127268406701    5.931831618301035    -4.114127268406701             2.7689999999999997   -502.0213360777252   439.3514157413572    1041.091935399735    6.58179131885481    -0.17170445234101908  -244.03492000000003        72.25
8      Gryffindor      Bella       Leatherman  1998-12-07  Left       20278.0     496.3949449852823    -5.215891145868072   -4.963949449852823             5.855                -626.552041128547    567.8424015938325    -6.198661229053442   925.2555003872844   1.0865178213133744    1.1470315267700957         -252.27561           244.11
```

###### example_features_list.json
```json
[
    "Astronomy",
    "Herbology",
    "Defense Against the Dark Arts",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Charms",
    "Flying"
]
```
> [!NOTE]
> The train program outputs a file by default called ```model.json```. It is important to predict with the next program below ```logreg_predict.py```

###### model.json
```json
{
    "classes": [
        "Gryffindor",
        "Hufflepuff",
        "Ravenclaw",
        "Slytherin"
    ],
    "weights": [
        [
            0.00025455692386502346,
            0.0009674232875324689,
            -0.0006456307377580096,
            -0.000534244818350825
        ],
        [
            -0.05805910531886797,
            0.08211465008508015,
            0.04462064855710283,
            -0.06884237879566252
        ],
        ...
    ],
    "biases": [
        3.2192712316678818,
        -5.2052742237778755,
        10.103497312451184,
        -6.806663360293857
    ],
    "column_names": [
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Divination",
        "Muggle Studies",
        "Ancient Runes",
        "History of Magic",
        "Transfiguration",
        "Potions",
        "Charms",
        "Flying"
    ]
}
```
### Multinomial Predictor Program
This program performs prediction for a categorical target variable using a multinomial logistic regression model. It computes the predicted probabilities of each class using the softmax function

* use ```--dataset DATASET``` or ```-d DATASET``` to specify the test dataset path to predict with the same values as the trained ones (Required)
* use ```--input MODEL_PATH``` or ```-i MODEL_PATH``` to specify the path to the input JSON file containing the model parameters, generated by the train model(Required)
* use ```--output FILENAME``` or ```-o FILENAME``` to change the name of the output CSV with the prediction results (default: result.csv)

##### for more information about options use:
```bash
python3 logreg_predict.py -h
```
Try:
```bash
python3 logreg_predict.py -d ../datasets/dataset_test.csv -i model.json
```

###### dataset_test.csv
```
Index   Hogwarts House  First Name  Last Name    Birthday    Best Hand  Arithmancy          Astronomy            Herbology            Defense Against the Dark Arts  Divination           Muggle Studies      Ancient Runes       History of Magic    Transfiguration     Potions              Care of Magical Creatures  Charms      Flying
0                       Rico        Sargent      2001-10-06  Right      41642.0             696.0960714808824    3.0201723778093963   -6.960960714808824             7.996                -365.1518504531068  393.13818539298967  4.207690767250213   1046.7427360602487  3.6689832316813447   0.3738525472517433         -244.48172  -13.62
1                       Tamara      Shackelford  1998-01-08  Left       45352.0             -370.8446553065122   2.9652261693057698   3.708446553065121              6.349                522.5804860261724   602.8530505526203   6.460017331082373   1048.0538781192083  8.51462223474569     0.5774322733924575         -231.292    -26.26
2                       Staci       Crandall     1998-09-15  Left       43502.0             320.3039904202326    -6.1856965870805025  -3.2030399042023268            4.619                -630.0732069461911  588.0717953227543   -5.565818052162058  936.4373579469948   1.8508293300849623   -1.6471500089738849        -252.99343  200.15
3                       Dee         Gavin        2001-05-10  Right      61831.0             407.2029277445324    4.962441602980879    -449.17980580026085            427.6999656113748    1043.3977178876494  4.656572653588079   1.164707896993084   -244.0166           -11.15
4                       Gregory     Gustafson    1999-02-01  Right      288.33774714151883  3.737655971167968    -2.883377471415188   4.886                          -449.7321661455943   385.712782047816    2.8763465676859883  1051.3779355988024  2.750586103354796   0.10210440893148606  -243.99806                 -7.12
5                       Katharine   Ebert        1998-09-25  Left       60888.0             -482.1390963522175   -7.327170322866657   4.821390963522175              -6.6110000000000015  -525.2725025237612  359.14817180301407  5.197858179249288   1070.7121427170061  10.032461125685584   -2.1852199688256664        -253.62839  -126.66
6                       Elizebeth   Clemens      1998-12-16  Left       46583.0             -488.04145290037286  4.276265977488614    4.880414529003729              4.916                408.92547374155805  633.7330020317315   3.5779424817589587  1046.2171724880452  7.922499688836782    -1.1784556822092844        -231.5856   32.67
7                       Santo       Simonson     1999-01-04  Left       42013.0             675.144047616016     4.024691099123406    -6.7514404761601625            5.1080000000000005   -590.514803881303   409.4206426135492   4.672595325050304   1056.701013197971   7.132861059425293    0.5631179851568311         -245.93425  13.45
8                       Rosalyn     Gee          1997-09-01  Left       27434.0             -508.7670008385748   7.146464798490754    5.087670008385747              4.586                512.8501336932145   591.5241863883227   3.12979163174824    1040.7342323213866  5.783287107300602    0.4193343687566451         -229.42604  60.88
```

###### result.csv
```csv
Index  Predicted Class
0      Hufflepuff
1      Ravenclaw
2      Gryffindor
3      Slytherin
4      Ravenclaw
5      Hufflepuff
6      Ravenclaw
7      Hufflepuff
8      Hufflepuff
...
```
