# Single Layer Perceptron for Iris Binary Classification

This project implements a **Single Layer Perceptron (SLP)** from scratch in Python to classify two Iris flower classes:

- **Iris-setosa** → `0`
- **Iris-versicolor** → `1`

The notebook performs forward propagation, sigmoid activation, prediction, squared-error calculation, gradient computation, weight updates, validation, and visualization across 5 training epochs.

## Files

- `SPL_Farhan_536804.ipynb` — main Jupyter Notebook containing the dataset, SLP implementation, training, validation, and charts.

## Dataset

The notebook uses a manually embedded subset of the Iris dataset.

### Training set

The training dataset contains **80 samples**:

- 40 Iris-setosa
- 40 Iris-versicolor

Each sample contains four input features:

| Feature | Description |
|---|---|
| `X1` | Sepal length |
| `X2` | Sepal width |
| `X3` | Petal length |
| `X4` | Petal width |
| `Target` | Class label (`0` or `1`) |

### Validation set

The validation dataset contains **20 samples**:

- 10 Iris-setosa
- 10 Iris-versicolor

The validation data is evaluated using the final bias and theta values obtained after each training epoch. No parameter updates are performed during validation.

## Model

The model uses four weights and one bias:

```text
bias
theta1
theta2
theta3
theta4
```

The weighted input is calculated as:

```text
z = bias + theta1*X1 + theta2*X2 + theta3*X3 + theta4*X4
```

The sigmoid activation function is:

```text
g(z) = 1 / (1 + exp(-z))
```

The prediction rule is:

```text
g(z) >= 0.5  ->  class 1
g(z) < 0.5   ->  class 0
```

The error is calculated as:

```text
error = g(z) - target
```

The squared error is:

```text
SSE = error^2
```

## Training

The model is trained using online / sample-by-sample gradient descent.

### Hyperparameters

| Parameter | Value |
|---|---:|
| Learning rate | `0.1` |
| Epochs | `5` |
| Initial bias | `0.5` |
| Initial theta1 | `0.5` |
| Initial theta2 | `0.5` |
| Initial theta3 | `0.5` |
| Initial theta4 | `0.5` |
| Activation | Sigmoid |
| Classification threshold | `0.5` |
| Loss | Mean squared error based on SSE |

For each training sample, the common gradient term is:

```text
d_common = 2 * error * g(z) * (1 - g(z))
```

The gradients are:

```text
dbias  = d_common
dtheta = d_common * X
```

Parameters are updated using:

```text
bias  = bias  - learning_rate * dbias
theta = theta - learning_rate * dtheta
```

## Training Results

The notebook produces the following training summary:

| Epoch | Average Loss | Accuracy |
|---:|---:|---:|
| 1 | 0.449889 | 52.50% |
| 2 | 0.037452 | 95.00% |
| 3 | 0.024372 | 97.50% |
| 4 | 0.017357 | 97.50% |
| 5 | 0.012740 | 98.75% |

The training loss decreases substantially across the five epochs while training accuracy increases to **98.75%**.

## Validation Results

The notebook evaluates the model after every epoch using the fixed parameters produced by that epoch.

| Epoch | Validation Loss | Validation Accuracy |
|---:|---:|---:|
| 1 | 0.328951 | 50.00% |
| 2 | 0.247289 | 50.00% |
| 3 | 0.175892 | 50.00% |
| 4 | 0.119381 | 85.00% |
| 5 | 0.081581 | 100.00% |

Validation loss decreases in every epoch. Validation accuracy remains at 50% for the first three epochs, improves to 85% in Epoch 4, and reaches **100% in Epoch 5**.

## Visualizations

The notebook generates six line charts:

1. **Training Loss per Epoch**
2. **Training Accuracy per Epoch**
3. **Validation Loss per Epoch**
4. **Validation Accuracy per Epoch**
5. **Training vs Validation Loss**
6. **Training vs Validation Accuracy**

These plots make it easier to observe the learning progress and compare model performance between training and validation data.

## Requirements

The notebook uses the following Python libraries:

```text
numpy
pandas
matplotlib
```

Install them with:

```bash
pip install numpy pandas matplotlib
```

## How to Run

1. Open `SPL_Farhan_536804(1).ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab.
2. Run the cells from top to bottom.
3. The notebook will:
   - load the embedded training and validation data,
   - initialize the SLP parameters,
   - train the model for 5 epochs,
   - evaluate validation performance after every epoch,
   - print the training and validation summary tables,
   - generate the loss and accuracy charts.

## Workflow

```text
Dataset
   |
   v
Prepare Training and Validation Data
   |
   v
Initialize Bias and Theta
   |
   v
Forward Propagation
   |
   v
Sigmoid Activation
   |
   v
Prediction
   |
   v
Error and SSE
   |
   v
Gradient Calculation
   |
   v
Update Weights
   |
   v
Repeat for 5 Epochs
   |
   v
Validation
   |
   v
Loss and Accuracy Visualization
```
