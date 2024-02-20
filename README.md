# Distance-Loss-Experiment
The official repository for optain experimental performance result on `distance loss functions`.

## Distance Loss Function
Distance loss function based on weighted mean square error. It can apply to monotonic classificaiton. There is a two types of Distance Loss, `Distance Mean Square`(DiMS), and `Absolute Distance Mean Square`(ADiMS). Below is a formula of DiMS.



$$
L_{DiMS} = \frac{1}{nl}\sum\limits_{i=1}^{n} \sum\limits_{j=1}^{l} (\lvert A(T_{i}) - j \rvert + 1)^{2} (T_{ij} - Y_{ij})^{2}
$$

$$
L_{ADiMS} = \frac{1}{nl}\sum\limits_{i=1}^{n} \sum\limits_{j=1}^{l} (\lvert A(T_{i}) - j \rvert + 1)(T_{ij} - Y_{ij})^{2}
$$

where

$$
A(L) = \underset{x \in S}{argmax} L_{x} = \lbrace x \in S|L(s) \leq L(x), \forall s \in S \rbrace
$$

$$
S = \lbrace s|1 \leq s \leq l, s \in N\rbrace
$$

## How to Run

### Install
You should install requirement python packages to run scripts without any error by type below.
```bash
pip install -r requirements.txt
```

### Text Classification
For text classification, type below on your terminal.
```bash
python text_classification.py --loss=dims --data=STS --seed=0
```
There is a three arguments, loss, data, and seed.

`data` argument choose dataset for measure performance.
- [Stanford Semantic Treebank](https://nlp.stanford.edu/sentiment/), SST-5 dataset : SST
- [TripAdvisor Hote Review](https://www.cs.cmu.edu/~jiweil/html/hotel-review.html) dataset : THR
- [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/) Office Products 5-core dataset : ARD 

### Numerical Classification
For numerical classification, type below on your terminal.
```bash
python numerical_classification.py --loss=dims --data=MPG --seed=0
```
There is a three arguments, loss, data, and seed.

`data` argument can choose dataset for measure performance.
- [Auto MPG](https://www.kaggle.com/datasets/uciml/autompg-dataset) dataset : THR
- [Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) dataset : BIKE
- [Diamonds](https://www.openml.org/search?type=data&sort=runs&id=42225&status=active) Dataset : DIA

### Classification on Scikit-learn Virtual Dataset
This task is a common classification with [scikit-learn](https://scikit-learn.org/stable/). By using [make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression) function, the virtual dataset can be generated. The virtual data is performed automatically sequential labeling.

For numerical classification, type below on your terminal.
```bash
python scikit_classification.py --loss=dims --label=10 ratio=0.8 --seed=0
```
There is a three arguments, loss, label, ratio, and seed.

`label` argument determine the number of label, and `ratio` argument determine ratio how much data to use as training data.


### Common Arguments
You can fix seed by `seed` argument. If you set seed 0, then all progress include data split, torch would be run ramdomly.

`loss` argument can determine the type of loss.
- Mean Square Error : mse
- Cross Entropy Loss : ce
- Mean Absolute Error : mae
- Distance Square Error : dims
- Absolute Diatance Square Error : adims

## Related Repository
- [distnace_loss_torch](https://github.com/9tailwolf/distance_loss_torch) : The official repository for distance loss function package.
- [Distance-Loss-Experiments_SST-5](https://github.com/9tailwolf/Distance-Loss-Experiment_SST-5) : The official repository for achieve score on SST-5 semantic analysis. 
