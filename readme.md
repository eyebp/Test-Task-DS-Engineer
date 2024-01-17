## Yuriy Kazanskyi Test Task

### 1. Counting islands. Classical Algorithms

Please refer to `1-islands/islands.ipynb` jupyter notebook.


### 2. Regression on the tabular data. General Machine Learning

A script to generate target values for hidden_test.csv.

Target can be generated by mathematical operations, so training a model is not required, and the training script is provided just for reference.

To train:
`python3 train.py train.csv`

To infer:
`python3 predict.py hidden_test.csv`

Run jupyter notebook to browse `explore.ipynb`

### 3. MNIST classifier. OOP

An example structure of a base class and child classes of an image classifier.

IMPORTANT - please download random forest model from here https://drive.google.com/file/d/19b2CMzigVKQXkzBfPB88-S8wlzV851Vm/view?usp=sharing and put it into `3-oop/`
because it was too large to fit on github directly.
or just use zipped version.

To test: `use_oop.ipynb`
To inspect: `oop.py`
Models trained in: `mnist-train.ipynb`
