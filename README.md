# Machine Learning Basics from Scratch

Welcome to **Machine Learning Basics from Scratch**! This repository is dedicated to learning and implementing classical machine learning algorithms fundamentally from the ground up using pure Python and NumPy.

The core motivation behind this project is to un-box the mathematical "black-box" of algorithms like Linear Regression, Random Forests, Support Vector Machines, and K-Means Clustering, rather than purely relying on libraries. All implementations are validated side-by-side against `scikit-learn` for numerical parity.

## 📂 Repository Structure

The project is logically divided into primary ML branches along with utility modules:

- **`supervised_ML/`**: Algorithms that learn from labeled training data.
  - *Classification*: Logistic Regression, Support Vector Machines (SVM), Random Forests, Softmax Regression.
  - *Regression*: Simple/Multiple Linear Regression, Ridge Regression, Lasso Regression, Gradient Descent modeling.
- **`unsupervised_ML/`**: Algorithms that infer patterns from unlabeled data.
  - *Clustering*: K-Means Clustering (Iris Dataset example).
- **`utils/`**: Reusable modules for consistent environments across models.
  - `cleaning.py`: Missing value interpolation and text-strip cleaning.
  - `encoding.py`: Word-to-number mappings and One-Hot Encoding functions.
  - `validation.py`: K-Fold cross validation and custom Train/Test split generators.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. Install the necessary dependencies via:
```bash
pip install -r requirements.txt
```

### Running an Example
You can run any of the standalone algorithm demonstrations to see the custom models compute alongside exact `scikit-learn` validation outputs and map to graphical plots.

For example, to run the K-Means clustering algorithm:
```bash
python unsupervised_ML/Clustering/KMeans/example_iris_kmeans.py
```

## 🤝 Acknowledgments & Credits

The curriculum structure, dataset choices, and framing of several graphical exercises in this repository were heavily inspired by the phenomenal [Machine Learning Tutorial Python Playlist](https://www.youtube.com/watch?v=gmvvaobm7eQ&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw) by **codebasics** (Dhaval Patel). 

While the fundamental algorithmic architecture and overarching base code were built from scratch within this repository separately, I highly recommend their channel to anyone beginning their data science journey!
