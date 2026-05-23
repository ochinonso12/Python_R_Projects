# Python & R Data Science Projects
**Chinonso Okoye** | MSc Data Science, University of Lincoln | [chinonsobedeokoye12@gmail.com](mailto:chinonsobedeokoye12@gmail.com)

This repository contains data science projects completed as part of my MSc in Data Science and Applied Analytics at the University of Lincoln. Each project applies machine learning to a real-world classification problem, with full evaluation against rule-based and baseline approaches.

---

## Projects

### 1. Diabetes Patient Classification — Distributed ML Pipeline on Apache Spark
**Files:** `models.py` | `streaming.py`

A distributed binary classification pipeline built on Apache Spark (MLlib + HDFS) to predict diabetes diagnosis from clinical records.

**Dataset:** Pima Indians Diabetes Dataset (NIDDK) — 768 female patients, 8 clinical features, binary outcome.

**Models trained and compared:**
| Model | Accuracy | AUC-ROC | F1-Score |
|---|---|---|---|
| Logistic Regression | 0.7886 | 0.8509 | 0.7822 |
| Random Forest | 0.7724 | **0.8521** | 0.7566 |
| Gradient Boosted Trees | 0.7398 | 0.8309 | 0.7247 |
| Decision Tree | 0.7480 | 0.7501 | 0.7502 |

**Key decisions:**
- Used AUC-ROC as primary metric (not accuracy) due to 65/35 class imbalance — a lazy classifier predicting all non-diabetic would score 65.1% accuracy while missing every diabetic patient
- Applied mean imputation via Spark DataFrame operations for 5 features containing implausible zero values
- Implemented **Spark Structured Streaming** for continuous model retraining on incoming patient data batches, preventing model drift
- 5-fold cross-validation with hyperparameter tuning via `CrossValidator` + `ParamGridBuilder`
- 80:20 train-test split with seed 42

**Tools:** PySpark, Apache Spark MLlib, HDFS, Spark Structured Streaming, Python

---

### 2. Potato Leaf Blight Detection — Rule-Based vs. Machine Learning
**File:** `image_task.ipynb`

Comparison of an HSV colour thresholding classifier (rule-based) against a flattened pixel Multi-Layer Perceptron (machine learning) on a 3-class agricultural image classification task.

**Dataset:** 2,152 potato leaf images — Early Blight (1,000), Late Blight (1,000), Healthy (152). Class imbalance ratio of 6.66x.

**Results on held-out test set (323 images):**
| Method | Accuracy | Macro F1 |
|---|---|---|
| HSV Colour Thresholding (rule-based) | 39.6% | 0.377 |
| Flattened MLP (256, 128) | **89.2%** | **0.841** |

**MLP improvement over rule-based: +49.5% accuracy, +0.464 macro F1**

**Key decisions:**
- Stratified 70/15/15 train/validation/test split to preserve class distribution under severe imbalance
- Macro-averaging used across all 3 classes to avoid inflating scores on majority class
- 5 MLP architectures evaluated on validation set; (256, 128) selected with best validation macro-F1 of 0.8815
- Images resized to 64×64 and normalised to [0,1] before training

**Tools:** Python, scikit-learn, NumPy, OpenCV, Matplotlib

---

### 3. Sarcasm Detection in News Headlines — Lexicon vs. TF-IDF Perceptron
**File:** `text_task.ipynb`

Comparison of a lexicon-based sentiment scorer (rule-based) against a TF-IDF vectoriser with a linear perceptron classifier on 28,620 labelled news headlines.

**Results on held-out test set (4,293 headlines):**
| Method | Accuracy | F1-Score |
|---|---|---|
| Lexicon Sentiment Scorer (rule-based) | 48.7% | 0.634 |
| TF-IDF + Perceptron | **76.5%** | **0.748** |

**TF-IDF Perceptron improvement: +27.9% accuracy, +0.114 F1**

**Key decisions:**
- NLTK Opinion Lexicon + 41 custom hyperbolic/ironic terms for rule-based baseline
- TF-IDF vocabulary of 18,870 terms (unigrams + bigrams); sublinear TF scaling applied
- Max-iter tuned across [50, 100, 200, 500, 1000, 2000] on validation set — all configurations achieved F1 of 0.7356, so min-iter (50) selected for computational efficiency
- WordNetLemmatizer used for TF-IDF preprocessing; PorterStemmer for rule-based

**Tools:** Python, scikit-learn, NLTK, Pandas, Matplotlib

---

## Skills Demonstrated
- Machine learning: classification, ensemble methods, neural networks, hyperparameter tuning
- Big data & distributed computing: Apache Spark, MLlib, HDFS, Structured Streaming
- NLP: TF-IDF vectorisation, lexicon-based sentiment analysis, text preprocessing
- Image processing: HSV colour thresholding, pixel normalisation, bilinear interpolation
- Evaluation: AUC-ROC, F1-score, macro-averaging, handling class imbalance
- Programming: Python, PySpark, scikit-learn, NLTK, NumPy, Pandas

---

## About Me
I am an MSc Data Science student at the University of Lincoln, graduating 2026. I hold a First Class BSc in Computer Science from KNUST, Ghana. I am interested in applying machine learning to health data and real-world decision support systems.
