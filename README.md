# Adaptive-SMOTE

Adaptive-SMOTE is an improved oversampling technique designed to address class imbalance in machine learning datasets. It adaptively generates synthetic minority samples by distinguishing boundary and inner points, ensuring better data distribution for classification tasks.

## Features

- Adaptive identification of boundary and inner points in the minority class.
- Generates synthetic samples intelligently to balance the dataset.
- Uses k-nearest neighbors (KNN) for identifying data distribution.
- Effective for handling imbalanced classification problems.

## Installation

Clone the repository:

```sh
git clone https://github.com/yourusername/Adaptive-SMOTE.git
```

Navigate to the project directory:

```sh
cd Adaptive-SMOTE
```

Ensure you have the required dependencies installed:

```sh
pip install numpy scikit-learn
```

## Usage

```python
import numpy as np
from adaptive_smote import ASmote, oversampleASmote

# Example dataset (labels in the first column)
data = np.array([
    [0, 2.5, 3.2],
    [0, 3.0, 2.8],
    [1, 1.5, 1.7],
    [1, 1.2, 1.3],
])

# Initialize ASMOTE
asmote = ASmote(data)
X, Xn, tp_more, Xp, tp_less = asmote.dataReset()

# Oversample the minority class
oversampled_data = oversampleASmote(data, Xn, Xp)
print("Oversampled Data:", oversampled_data)
```

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.


## Acknowledgments

- Inspired by SMOTE (Synthetic Minority Over-sampling Technique)
- Uses k-nearest neighbors for synthetic sample generation
