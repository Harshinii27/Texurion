# Machine Learning — Detailed Study Notes

A complete set of notes covering three modules:

- **Module 1:** Introduction to Machine Learning
- **Module 2:** Model Preparation, Evaluation and Feature Engineering
- **Module 3:** Supervised Learning

Each topic contains a definition, the core idea, key formulas, worked examples where useful, and a comparison/summary where a topic is best understood against its alternatives.

---

# Module 1: Introduction to Machine Learning

## 1.1 Types of Human Learning

Human learning is the process of acquiring knowledge, skills, habits, or values through experience, study, or instruction. In cognitive science and education, several types are recognised. They matter here because machine learning is an attempt to reproduce, on a computer, some of the mechanisms by which humans learn.

| Type of human learning | Meaning | Rough ML analogue |
|---|---|---|
| **Learning by experience** | Improving through repeated practice and feedback | Reinforcement learning; training on data |
| **Learning by examples (inductive learning)** | Generalising a rule from specific instances | Supervised learning |
| **Learning by analogy** | Solving a new problem by relating it to a known similar problem | Case-based reasoning, nearest-neighbour methods |
| **Learning by instruction (deductive learning)** | Applying explicitly given rules/facts | Rule-based / expert systems; symbolic AI |
| **Rote learning (memorisation)** | Storing facts verbatim, recalling without understanding | Lookup tables, overfitting / memorisation |
| **Discovery learning** | Deriving novel concepts with little guidance | Unsupervised learning, clustering |

**Key idea:** Most modern ML corresponds to *inductive learning* — we give the program many examples and it *induces* a general rule (a model) from them.

**Connection to ML:**
- Humans learn from **experience** → machines learn from **data/experience**.
- Humans **generalise** from few examples → machines aim to **generalise** to unseen data, not just memorise.
- Humans improve with **feedback** → machines improve by minimising error on data.

## 1.2 What is Machine Learning?

**Machine Learning (ML)** is a subfield of Artificial Intelligence (AI) concerned with building computer programs that automatically improve their performance at a task through experience (data), without being explicitly programmed for every rule.

### Classical definitions

- **Arthur Samuel (1959):** "Field of study that gives computers the ability to learn without being explicitly programmed." (Famous checkers-playing program.)
- **Tom Mitchell (1997), the formal definition:** "A computer program is said to learn from **experience E** with respect to some class of **tasks T** and **performance measure P**, if its performance at tasks in T, as measured by P, improves with experience E."

### Mitchell's T, E, P — with examples

| Problem | Task (T) | Experience (E) | Performance (P) |
|---|---|---|---|
| Spam filtering | Classify email as spam/not spam | Labelled emails | Accuracy / precision-recall |
| Handwriting recognition | Recognise handwritten digit | Images with digit labels | % correct digits |
| House price prediction | Predict price | Historical house records | Mean squared error |
| Chess | Choose best move | Games played against self/opponent | Win rate |

### ML vs traditional programming

| Aspect | Traditional programming | Machine Learning |
|---|---|---|
| Input | Data + **rules** | Data + **answers (labels)** |
| Output | **Answers** | **Rules (model)** |
| Rules | Hand-coded by humans | Learned automatically |
| Adaptability | Must be re-coded for new cases | Improves with more data |
| Best for | Well-defined, deterministic logic | Complex, changing, pattern-based tasks |

> In traditional programming we write `if/else` logic. In ML we *show* the computer examples and let it discover the logic.

## 1.3 Types of Machine Learning

ML is broadly divided into four paradigms, based on **how much and what kind of supervision** the data provides.

### 1.3.1 Supervised Learning

- **Data:** labelled — every example has an input `X` and a known output `y`.
- **Goal:** learn a mapping `f: X → y` that predicts `y` for unseen `X`.
- **Two sub-types:**
  - **Classification** — output is a *discrete* class (e.g., spam/not-spam, cat/dog).
  - **Regression** — output is a *continuous* number (e.g., price, temperature).
- **Common algorithms:** Linear Regression, Logistic Regression, K-Nearest Neighbours, Support Vector Machines, Decision Trees, Random Forest, Naïve Bayes, Neural Networks.
- **Analogy:** learning with a teacher who provides the correct answer for every practice question.
- **Example:** given labelled photos of tumours (benign/malignant), learn to classify new scans.

### 1.3.2 Unsupervised Learning

- **Data:** unlabelled — only inputs `X`, no output `y`.
- **Goal:** discover hidden structure, patterns, or groupings in the data.
- **Main tasks:**
  - **Clustering** — group similar items (k-Means, Hierarchical, DBSCAN).
  - **Association rule mining** — find co-occurrence rules (Apriori, Eclat; e.g., "bread → butter").
  - **Dimensionality reduction** — compress features while preserving structure (PCA, t-SNE, autoencoders).
- **Analogy:** learning without a teacher; the student groups things by similarity on their own.
- **Example:** customer segmentation from purchase behaviour; anomaly detection.

### 1.3.3 Semi-Supervised Learning

- **Data:** a **small labelled** set + a **large unlabelled** set.
- **Goal:** use the cheap unlabelled data to improve learning when labelling is expensive.
- **Techniques:**
  - **Self-training** — train on labelled data, pseudo-label confident unlabelled examples, retrain.
  - **Co-training** — two views/classifiers label data for each other.
  - **Label propagation / graph-based** methods.
- **Why useful:** labelling is often costly, slow, or requires experts (medical images, legal documents), while unlabelled data is abundant.
- **Example:** classify web pages where only a few are labelled.

### 1.3.4 Reinforcement Learning (RL)

- **Setup:** an **agent** interacts with an **environment**, observes **states**, takes **actions**, and receives **rewards** (or penalties).
- **Goal:** learn a **policy** (a strategy of action selection) that maximises **long-term cumulative reward**.
- **Key concepts:** state, action, reward, policy, value function, discount factor, environment.
- **Central challenge:** **exploration vs exploitation** — try new actions (to learn) vs use known good actions (to score).
- **Characteristics:** trial-and-error, delayed reward, no labelled examples.
- **Algorithms:** Q-Learning, SARSA, Deep Q-Networks (DQN), Policy Gradient, Actor-Critic.
- **Examples:** game playing (AlphaGo, Atari), robotics, self-driving cars, recommendation, resource scheduling.

### Comparison of the four types

| Feature | Supervised | Unsupervised | Semi-supervised | Reinforcement |
|---|---|---|---|---|
| Data labels | Fully labelled | No labels | Few labelled + many unlabelled | No labels; rewards instead |
| Goal | Predict output | Find structure | Improve using unlabelled data | Maximise reward over time |
| Feedback | Correct answer each example | None | Partial | Reward signal |
| Typical tasks | Classification, regression | Clustering, dim. reduction | Classification with limited labels | Control, decision-making |
| Example algorithm | SVM, Regression | k-Means, PCA | Self-training | Q-Learning |
| Example | Spam filter | Customer segments | Web-page classification | Game AI |

## 1.4 Machine Learning Activities

The ML project lifecycle (often called the **ML pipeline**) consists of a sequence of activities. A widely used ordering:

1. **Problem definition** — state the task `T`, the performance measure `P`, and what success means. Decide supervised/unsupervised/RL.
2. **Data collection / gathering** — obtain raw data from databases, sensors, web scraping, APIs, or surveys.
3. **Data preparation (pre-processing)** — clean, integrate, and format the data:
   - handle missing values and outliers,
   - remove duplicates and noise,
   - encode categorical variables, scale numeric ones,
   - split into training/validation/test sets.
4. **Exploratory Data Analysis (EDA)** — visualise and summarise to understand distributions, relationships, and data quality.
5. **Feature engineering** — construct, transform, and select the most informative features.
6. **Model selection** — choose candidate algorithms appropriate to the task and data.
7. **Model training** — fit the model's parameters to the training data by minimising a loss function.
8. **Model evaluation** — measure performance on held-out data using appropriate metrics and cross-validation.
9. **Hyperparameter tuning** — optimise settings that are not learned from data (grid/random search, Bayesian optimisation).
10. **Deployment** — integrate the model into a production system/service.
11. **Monitoring and maintenance** — track performance over time, detect **data drift / concept drift**, and retrain as needed.

**Important note:** this is **iterative**, not linear. Poor evaluation results often send you back to feature engineering, data collection, or model selection.

## 1.5 Applications of Machine Learning

| Domain | Applications |
|---|---|
| **Healthcare** | Disease diagnosis, medical imaging, drug discovery, personalised treatment, epidemic prediction |
| **Finance** | Credit scoring, fraud detection, algorithmic trading, risk assessment, loan approval |
| **Retail / E-commerce** | Recommendation systems, demand forecasting, dynamic pricing, customer segmentation |
| **Natural Language Processing** | Machine translation, sentiment analysis, chatbots, spam filtering, speech recognition |
| **Computer Vision** | Face recognition, object detection, autonomous vehicles, OCR, medical image analysis |
| **Manufacturing** | Predictive maintenance, quality control, defect detection, supply-chain optimisation |
| **Transportation** | Self-driving cars, route optimisation, traffic prediction, ride-share matching |
| **Security** | Intrusion detection, malware detection, biometric authentication |
| **Energy** | Load forecasting, smart-grid optimisation, fault detection |
| **Entertainment** | Content recommendation (Netflix/YouTube), game AI, content generation |

**Everyday examples:** email spam filters, Netflix/YouTube recommendations, voice assistants (Siri/Alexa), Google Translate, credit-card fraud alerts, face unlock on phones, and product recommendations on Amazon.

---

# Module 2: Model Preparation, Evaluation and Feature Engineering

## 2.1 Types of Data in Machine Learning

### By structure

- **Structured data** — organised in rows/columns (tables, spreadsheets, databases). Easy to store and analyse. Examples: age, salary, transaction records.
- **Semi-structured data** — some organisation/tags but not a strict table. Examples: JSON, XML, HTML.
- **Unstructured data** — no predefined model. Examples: images, audio, video, free text.

### By nature of values (levels of measurement)

| Scale | Meaning | Order? | Equal intervals? | True zero? | Examples |
|---|---|---|---|---|---|
| **Nominal** | Categories/labels only | No | No | No | Colour, gender, city |
| **Ordinal** | Ordered categories | Yes | No | No | Ratings (poor/fair/good), education level |
| **Interval** | Ordered, equal intervals, no true zero | Yes | Yes | No | Temperature (°C), year |
| **Ratio** | Ordered, equal intervals, true zero | Yes | Yes | Yes | Height, weight, age, income |

- **Qualitative (categorical):** nominal and ordinal.
- **Quantitative (numerical):**
  - **Discrete** — countable whole values (number of children).
  - **Continuous** — any value in a range (height, temperature).

### Terminology of a dataset

- **Instance / example / sample / record / row** — one data point.
- **Feature / attribute / variable / column / predictor** — a measurable property.
- **Label / target / response / dependent variable** — the value to predict.
- Vector form: instance = `(x1, x2, …, xn)`; dataset = `{ (x_i, y_i) }`.

## 2.2 Exploring Structure of Data

Before modelling, we explore the data (Exploratory Data Analysis, EDA) to understand its structure, quality, and relationships.

### Descriptive statistics

- **Central tendency** — where data centres:
  - Mean (average, sensitive to outliers), Median (middle value, robust), Mode (most frequent).
- **Dispersion / spread** — how spread out data is:
  - Range = max − min, Variance, Standard deviation, Interquartile range (IQR = Q3 − Q1).
- **Shape** — skewness (asymmetry) and kurtosis (peakedness/heaviness of tails).
- **Position** — percentiles, quartiles, z-scores.

Formulas:
- Mean: `x̄ = (1/n) Σ xᵢ`
- Variance (sample): `s² = Σ(xᵢ − x̄)² / (n − 1)`
- Standard deviation: `s = √s²`
- z-score: `z = (x − μ) / σ`

### Types of analysis

- **Univariate** — one variable (histogram, box plot, summary stats).
- **Bivariate** — relationship between two variables (scatter plot, correlation).
- **Multivariate** — three or more variables (pair plots, correlation matrix, heatmaps).

### Common visualisations and what they reveal

| Plot | Reveals |
|---|---|
| Histogram / density plot | Distribution, modality, skew, outliers |
| Box plot | Median, quartiles, IQR, outliers |
| Scatter plot | Relationship, correlation, clusters, trends |
| Pair plot | All pairwise relationships at once |
| Correlation heatmap | Strength/direction of linear relations |
| Bar chart | Frequency of categories |

### Correlation

- **Pearson correlation `r`** — measures **linear** association, range −1 to +1.
- **Spearman** — rank-based, captures monotonic (non-linear) association.
- **Correlation ≠ causation.**
- Multicollinearity (high correlation among predictors) can destabilise linear models.

### Data quality issues to look for

Missing values, noise, outliers, duplicates, inconsistent formats/units, imbalanced classes, and data leakage.

## 2.3 Data Pre-processing

The goal is to convert raw, messy data into a clean, model-ready form. It is often the most time-consuming step.

### (a) Data cleaning

- **Missing values:**
  - Deletion: listwise (drop rows) or column-wise (drop features) — simple but loses data.
  - Imputation: mean/median/mode, constant value, forward/backward fill (time series), or model-based (KNN, regression) imputation.
- **Noise removal:** binning, regression smoothing, clustering-based smoothing.
- **Outlier handling:** detect via z-score, IQR, or isolation forest; then remove, cap, or transform.
- **Duplicates:** detect and remove.

### (b) Data integration

Combine data from multiple sources; resolve schema conflicts, duplicate entities, and value conflicts.

### (c) Data transformation

- **Scaling:** min–max normalisation, standardisation (z-score), robust scaling.
- **Encoding categorical data:** one-hot, label/ordinal, binary, target encoding.
- **Non-linear transforms:** log, square root, Box–Cox, Yeo–Johnson (to reduce skew).
- **Discretisation:** convert continuous to bins (equal-width, equal-frequency, k-means).

### (d) Data reduction

- **Dimensionality reduction:** PCA, feature selection, autoencoders.
- **Numerosity reduction:** sampling, aggregation (e.g., daily → monthly).
- Reduces storage, computation, and overfitting risk.

### (e) Handling imbalanced data

- **Resampling:** random oversampling, random undersampling.
- **SMOTE** — synthesise new minority examples by interpolation.
- **Class weights** and choosing metrics other than accuracy (F1, recall, AUC).

### (f) Data splitting

- **Train / validation / test** (e.g., 60/20/20 or 70/15/15).
- **Cross-validation** for more reliable estimates.
- **Crucial:** fit scalers/encoders on training data only, then apply to validation/test, to avoid **data leakage**.

## 2.4 Model Selection and Training (Supervised Learning)

### Model selection

Choosing which algorithm (and which hyperparameters) to use. Considerations:

| Factor | Consideration |
|---|---|
| Problem type | Classification vs regression |
| Data size | Small → simple models; large → complex models/ensembles |
| Dimensionality | High-dim → regularised/SVM/tree models |
| Interpretability | Need explanation? → linear models, trees |
| Training time / compute | Real-time constraints |
| Noise / outliers | Robust models (trees) vs sensitive (linear) |
| Linearity | Linear (regression, SVM-linear) vs non-linear (trees, kernel SVM, NN) |

### Training

Supervised training means **learning model parameters** from labelled data by minimising an **objective/loss function**:

- Represent the model as `ŷ = f(x; θ)`, where `θ` are the parameters.
- Define a loss `L(y, ŷ)` measuring error (e.g., MSE for regression, cross-entropy for classification).
- Find `θ*` that minimises the average loss over training data:
  `θ* = argmin_θ (1/n) Σ L(yᵢ, f(xᵢ; θ))`
- Optimise using **gradient descent** (batch, stochastic, mini-batch), or closed-form solutions (e.g., normal equation for linear regression).

### Bias–variance trade-off

- **Bias** — error from overly simplistic assumptions (underfitting).
- **Variance** — error from sensitivity to the training sample (overfitting).
- **Total error ≈ Bias² + Variance + Irreducible error.**
- Goal: balance the two for best generalisation.

| | High bias (underfit) | High variance (overfit) |
|---|---|---|
| Train error | High | Low |
| Test error | High | High |
| Cause | Too simple model | Too complex model / too little data |
| Fix | More features, complex model | More data, simpler model, regularisation |

### Parameters vs hyperparameters

- **Parameters:** learned from data (weights, coefficients).
- **Hyperparameters:** set before training (learning rate, k in KNN, tree depth, C in SVM). Tuned via validation.

### Regularisation (prevents overfitting)

- **L1 (Lasso):** adds `λ Σ|θ|`; drives some coefficients to exactly zero → feature selection.
- **L2 (Ridge):** adds `λ Σθ²`; shrinks coefficients smoothly.
- **Elastic Net:** combines L1 + L2.
- Other: early stopping, dropout (NN), pruning (trees), max depth constraints.

### Model selection techniques

- Train/validation split, k-fold cross-validation.
- **Grid search** — try all combinations in a grid.
- **Random search** — sample combinations randomly (often more efficient).
- **Bayesian optimisation** — model the objective and choose promising points.

## 2.5 Model Representation and Interpretability

### Model representation

How a model encodes the relationship between input and output:

- **Linear (parametric):** `y = θ0 + θ1x1 + … + θnxn` — fixed functional form, few parameters.
- **Non-linear:** polynomial, kernel methods, tree structures, neural networks.
- **Instance-based (non-parametric):** the model *is* (part of) the training data, e.g., KNN.
- **Probabilistic:** models a probability distribution, e.g., Naïve Bayes, Gaussian Mixture.
- **Ensemble:** combines many models — bagging (Random Forest), boosting (AdaBoost, Gradient Boosting), stacking.

**Hypothesis space:** the set of all models a learning algorithm can represent. A richer space can fit more complex data but risks overfitting.

### Interpretability

The degree to which a human can understand *why* a model made a prediction.

| Interpretable ("white-box") | Less interpretable ("black-box") |
|---|---|
| Linear/logistic regression | Deep neural networks |
| Decision trees, decision rules | Large ensembles of trees |
| Naïve Bayes | Kernel SVM |
| k-NN with small k (locally) | Complex gradient boosting |

Why interpretability matters: trust, debugging, legal/regulatory compliance (e.g., credit decisions, healthcare), and scientific insight.

**Explainability (XAI) techniques:**
- **Feature importance** (tree-based, permutation).
- **LIME** — locally approximate the model with a simple interpretable model.
- **SHAP** — Shapley-value-based contributions of each feature.
- **Partial dependence plots** — effect of a feature on the prediction.
- **Saliency maps** for images.

### Trade-off

Often **accuracy vs interpretability**: simple models are transparent but may be less accurate; complex models are accurate but opaque. Model choice depends on the application.

## 2.6 Evaluating Machine Learning Algorithms and Performance Enhancement

### Classification metrics

- **Confusion matrix** (binary):

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

- **Accuracy** = (TP + TN) / (TP + TN + FP + FN) — misleading on imbalanced data.
- **Precision** = TP / (TP + FP) — of predicted positives, how many are correct.
- **Recall (Sensitivity/TPR)** = TP / (TP + FN) — of actual positives, how many were found.
- **Specificity (TNR)** = TN / (TN + FP).
- **F1-score** = 2 · (Precision · Recall) / (Precision + Recall) — harmonic mean.
- **ROC curve** — TPR vs FPR at all thresholds; **AUC** summarizes it (1.0 = perfect, 0.5 = random).
- **Precision–Recall curve** — better for highly imbalanced data.

### Regression metrics

- **MSE** = `(1/n) Σ(yᵢ − ŷᵢ)²`
- **RMSE** = `√MSE` (same units as target).
- **MAE** = `(1/n) Σ|yᵢ − ŷᵢ|` (robust to outliers).
- **R² (coefficient of determination)** = `1 − SS_res/SS_tot` — fraction of variance explained.
- **Adjusted R²** — penalises adding useless predictors.

### Resampling for reliable evaluation

- **Hold-out** — simple but high variance.
- **k-fold cross-validation** — split into k folds, train on k−1, test on 1, repeat; average results.
- **Stratified k-fold** — preserves class proportions per fold.
- **Leave-One-Out (LOOCV)** — k = n; expensive but low bias.
- **Repeated k-fold** — reduces variance of the estimate.
- **Time-series split** — respects temporal order (no future leakage).

### Performance enhancement of models

1. **More/better data** — often the biggest gain.
2. **Better features** — feature engineering, selection, transformation.
3. **Hyperparameter tuning** — grid/random/Bayesian search.
4. **Regularisation** — reduce overfitting (L1/L2, dropout, pruning).
5. **Ensembling:**
   - **Bagging** — train many models on bootstrap samples and average (Random Forest). Reduces variance.
   - **Boosting** — sequentially train models to correct previous errors (AdaBoost, XGBoost, LightGBM). Reduces bias.
   - **Stacking** — combine base-model predictions with a meta-model.
6. **Handling imbalance** — resampling, class weights, appropriate metrics.
7. **Algorithm selection** — try multiple families and compare with cross-validation.

## 2.7 What is Feature Engineering?

**Feature engineering** is the process of using domain knowledge and data-transformation techniques to create, transform, and select **features (input variables)** that make a machine-learning algorithm work better.

- **Raw data → informative features → better model.**
- It can improve accuracy more than choosing a fancier algorithm.
- It requires understanding the problem and the data.

### The feature-engineering process

1. **Brainstorm features** — think about what might be predictive.
2. **Feature construction** — create new features from existing ones (e.g., `BMI = weight/height²`, `age = current_year − birth_year`, `price_per_sqft`).
3. **Feature transformation** — change the scale/shape/encoding of features (Section 2.8).
4. **Feature selection** — keep the most useful features (Section 2.9).
5. **Evaluate** — measure whether the model improved.
6. **Iterate.**

### Types of feature-engineering operations

- **Aggregation** — summary stats over groups (mean, count).
- **Decomposition** — split a feature (date → day, month, year).
- **Interaction** — combine features (x1 · x2, x1 / x2).
- **Encoding** — represent categories numerically.
- **Scaling** — put features on comparable ranges.

## 2.8 Feature Transformation

Changing the representation of features without necessarily removing them.

### (a) Scaling / normalisation

| Method | Formula | Notes |
|---|---|---|
| Min–Max normalisation | `(x − min) / (max − min)` → [0,1] | Sensitive to outliers |
| Standardisation (z-score) | `(x − μ) / σ` → mean 0, std 1 | Preferred for many algorithms |
| Robust scaling | `(x − median) / IQR` | Robust to outliers |

**Why:** distance-based (KNN, k-Means, SVM) and gradient-based (NN, logistic regression) algorithms need features on comparable scales.

### (b) Encoding categorical variables

- **Label / ordinal encoding** — assign integers (good for ordinal data; can mislead models for nominal data).
- **One-hot encoding** — one binary column per category (avoids false ordering; increases dimensionality).
- **Binary encoding** — compact binary representation.
- **Target/mean encoding** — replace category with the mean target (powerful but risks leakage; use CV).

### (c) Non-linear transformations

- **log(x)**, **√x** — reduce right skew.
- **Box–Cox / Yeo–Johnson** — power transforms to make data more Gaussian.
- **Polynomial features** — create x², x³, x1x2 to model non-linear relationships.

### (d) Discretisation (binning)

Convert continuous features into categorical bins (equal width, equal frequency, or k-means). Useful for non-linear models and interpretability.

### (e) Dimensionality reduction (as transformation)

- **PCA** — project onto orthogonal components of maximum variance.
- **t-SNE / UMAP** — non-linear embeddings, mainly for visualisation.

### (f) Text-specific transformations

Bag of Words, TF-IDF, word embeddings (Word2Vec, GloVe).

## 2.9 Feature Subset Selection

Choosing a subset of the original features by **removing irrelevant or redundant** ones (without creating new ones).

### Why do it?

- Reduce **overfitting** (fewer features, less noise).
- Reduce **training time and computation**.
- Improve **interpretability**.
- Combat the **curse of dimensionality**.

### Three families of methods

| Family | How it works | Pros | Cons | Examples |
|---|---|---|---|---|
| **Filter** | Rank features by a statistical score, independent of the model | Fast, model-agnostic | Ignores feature interactions | Variance threshold, correlation, χ², mutual information, ANOVA F-test |
| **Wrapper** | Use the model's performance to search feature subsets | Accounts for interactions, often best accuracy | Computationally expensive | Forward selection, backward elimination, RFE, exhaustive search |
| **Embedded** | Selection happens during model training | Balanced cost/accuracy | Model-specific | Lasso (L1), Ridge, tree feature importance, Elastic Net |

### Wrapper search strategies

- **Forward selection** — start empty, add the best feature one at a time.
- **Backward elimination** — start with all, remove the worst one at a time.
- **Recursive Feature Elimination (RFE)** — repeatedly train and drop the weakest.
- **Exhaustive / bidirectional** — combinations.

### Key terms

- **Relevant feature** — affects the target.
- **Redundant feature** — duplicated information (correlated with another feature).
- **Irrelevant feature** — no useful information; adds noise.

---

# Module 3: Supervised Learning

## 3.1 Introduction to Regression

**Regression** is a supervised-learning task that predicts a **continuous numeric output** from one or more input variables.

- **Regression vs classification:** regression → continuous output (price, temperature); classification → discrete class (spam/ham).
- **Terminology:**
  - **Independent variables / predictors / features** (`X`).
  - **Dependent variable / target / response** (`y`).
- **Types:** simple linear, multiple linear, polynomial, ridge/lasso, logistic (for classification), etc.
- **Goal:** find a function `y = f(X)` that fits the data and generalises to new inputs.

**Regression assumptions (for linear models):**

1. **Linearity** — relationship is linear in the parameters.
2. **Independence** — observations are independent.
3. **Homoscedasticity** — constant error variance.
4. **Normality** — residuals are normally distributed (mainly for inference).
5. **No/low multicollinearity** — predictors not highly correlated.

## 3.2 Simple Linear Regression

Models the linear relationship between **one** predictor `x` and a continuous target `y`.

### Model

`y = b0 + b1·x + ε`

- `b0` — **intercept** (value of y when x = 0).
- `b1` — **slope / regression coefficient** (change in y per unit change in x).
- `ε` — error/residual (unexplained variation).

The fitted line: `ŷ = b0 + b1·x`.

### Learning the parameters — Ordinary Least Squares (OLS)

Minimise the **sum of squared errors (SSE)**:

`SSE = Σ (yᵢ − ŷᵢ)² = Σ (yᵢ − b0 − b1·xᵢ)²`

Closed-form solutions:

- `b1 = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)² = Cov(x,y) / Var(x)`
- `b0 = ȳ − b1·x̄`

### Worked example

Data: (x, y) = (1, 2), (2, 4), (3, 5), (4, 4), (5, 5).

- `x̄ = 3`, `ȳ = 4`.
- `Σ(xᵢ − x̄)(yᵢ − ȳ)`: (−2)(−2)+(−1)(0)+(0)(1)+(1)(0)+(2)(1) = 4+0+0+0+2 = 6.
- `Σ(xᵢ − x̄)²`: 4+1+0+1+4 = 10.
- `b1 = 6/10 = 0.6`; `b0 = 4 − 0.6·3 = 2.2`.
- Fitted model: `ŷ = 2.2 + 0.6x`.
- For `x = 6`, prediction `ŷ = 2.2 + 3.6 = 5.8`.

### Evaluation

- **R²** = 1 − SSE/SST, where `SST = Σ(yᵢ − ȳ)²`. Fraction of variance explained (0 to 1).
- **RMSE / MAE** — error magnitude in target units.
- Residual plots to check assumptions.

### Pros/cons

- **Pros:** simple, interpretable, fast; coefficients show effect size.
- **Cons:** only linear relationships; sensitive to outliers; needs scaled considerations for interpretation (not fitting).

## 3.3 Multiple Linear Regression

Models the relationship between **two or more** predictors and a continuous target.

### Model

`y = b0 + b1·x1 + b2·x2 + … + bn·xn + ε`

- Each `bi` is the effect of `xi` **holding the other predictors constant**.
- Partial regression coefficients.

### Matrix form

`y = X·b + ε`

- `X` — design matrix (rows = samples, columns = features, plus a column of 1s for the intercept).
- OLS solution (**normal equation**): `b = (XᵀX)⁻¹ Xᵀy` (when `XᵀX` is invertible).

### Worked mini-example

Predict house price from area (`x1`) and age (`x2`):
`price = b0 + b1·area + b2·age`.
If `b1 = 50` (price per unit area) and `b2 = −2` (price lost per year of age), then a 1000-unit, 10-year-old house predicts `b0 + 50000 − 20`.

### Key issues

- **Multicollinearity** — predictors highly correlated → unstable coefficients. Detect with a **correlation matrix** or **Variance Inflation Factor (VIF)**; `VIF = 1/(1 − Rᵢ²)` (VIF > 5–10 is a concern). Fix by removing/combining features, or using ridge/PCA.
- **Overfitting** — too many predictors; use feature selection, adjusted R², or regularisation.
- **Adjusted R²** penalises the addition of non-informative predictors:
  `R²_adj = 1 − (1 − R²)(n − 1)/(n − p − 1)`.

### Categorical predictors

Encode using **dummy/one-hot variables** (drop one category to avoid the **dummy-variable trap** of perfect collinearity).

### Cautions

- Scale of coefficients depends on feature scale (standardise to compare effects).
- Assumes additivity unless interaction terms are added (`x1·x2`).

## 3.4 Logistic Regression

Despite its name, logistic regression is a **classification** algorithm (usually binary).

### Why not linear regression for classification?

A straight line can output values far outside [0,1] and is not suited to probabilities. Logistic regression maps a linear combination to a probability via the **sigmoid** function.

### The sigmoid (logistic) function

`σ(z) = 1 / (1 + e^(−z))`, which maps any real `z` to (0, 1).

### Model

- Linear combination: `z = b0 + b1x1 + … + bnxn`.
- Probability: `P(y = 1 | x) = σ(z) = 1 / (1 + e^(−z))`.
- `P(y = 0 | x) = 1 − P(y = 1 | x)`.

### Logit / log-odds

The inverse of sigmoid is the **log-odds (logit)**:

`log( p / (1 − p) ) = b0 + b1x1 + … + bnxn`

So logistic regression is **linear in the log-odds**. A positive `bi` increases the odds of class 1.

### Decision rule

Predict class 1 if `P(y=1|x) ≥ threshold` (default 0.5), else class 0. The threshold can be tuned (e.g., for imbalanced data).

### Learning the parameters

Parameters are found by **maximum likelihood estimation (MLE)** / minimising **log-loss (binary cross-entropy)**:

`J = −(1/n) Σ [ yᵢ·log(p̂ᵢ) + (1 − yᵢ)·log(1 − p̂ᵢ) ]`

Optimised with gradient descent (no closed-form solution).

### Variants

- **Binomial** — two classes.
- **Multinomial** — more than two classes (softmax).
- **Ordinal** — ordered categories.

### Evaluation

Confusion matrix, accuracy, precision, recall, F1, ROC-AUC; interpret coefficients as odds ratios (`e^bi`).

### Pros/cons

- **Pros:** interpretable, gives probabilities, efficient, works well for linearly separable classes.
- **Cons:** assumes linear decision boundary (in log-odds); needs feature engineering for complex non-linear problems; sensitive to outliers and multicollinearity.

## 3.5 Bayes' Theorem and Naïve Bayes

### Conditional probability and Bayes' theorem

- Conditional probability: `P(A|B) = P(A ∩ B) / P(B)`.
- **Bayes' theorem:**

`P(A|B) = P(B|A) · P(A) / P(B)`

where:
- `P(A)` = **prior** (belief before seeing B),
- `P(B|A)` = **likelihood**,
- `P(A|B)` = **posterior** (updated belief),
- `P(B)` = **evidence** (normalising constant).

For classification with features `X = (x1,…,xn)` and class `C`:

`P(C | X) = P(X | C) · P(C) / P(X)`

Prediction = class with the maximum posterior:

`Ĉ = argmax_C P(X | C) · P(C)`   (since `P(X)` is the same for all classes).

### Naïve Bayes classifier

**"Naïve" assumption:** features are **conditionally independent** given the class:

`P(X | C) = P(x1 | C) · P(x2 | C) · … · P(xn | C)`

Therefore:

`Ĉ = argmax_C P(C) · Π P(xᵢ | C)`

This assumption is rarely strictly true, yet Naïve Bayes often performs well, is very fast, and needs little data.

### Types of Naïve Bayes

| Variant | Feature type | Model for `P(xᵢ|C)` |
|---|---|---|
| **Gaussian** | Continuous | Normal distribution (mean, variance per class) |
| **Multinomial** | Counts (e.g., word counts) | Multinomial distribution |
| **Bernoulli** | Binary/boolean | Bernoulli distribution |
| **Categorical** | Categorical | Category probabilities |

### Laplace (additive) smoothing

To avoid zero probabilities when a feature value never appears in a class:

`P(xᵢ | C) = (count(xᵢ, C) + α) / (count(C) + α·k)`

where `α` (often 1) is the smoothing parameter and `k` is the number of possible values.

### Worked example (Play Tennis, simplified)

Suppose, among records with `Outlook = Sunny`: 3 "No" and 2 "Yes", and the prior `P(Yes) = 9/14`, `P(No) = 5/14`. Then:

`P(Yes | Sunny) ∝ P(Sunny|Yes)·P(Yes) = (2/9)(9/14) = 2/14`
`P(No | Sunny) ∝ P(Sunny|No)·P(No) = (3/5)(5/14) = 3/14`

Since `3/14 > 2/14`, predict **No** for a sunny day (using this single feature). Adding more features multiplies the likelihoods.

### Pros/cons

- **Pros:** very fast, simple, works with small data, handles many features, good for text (spam filtering, sentiment).
- **Cons:** strong independence assumption; probability estimates are often poorly calibrated; sensitive to correlated features.

## 3.6 K-Nearest Neighbours (KNN)

KNN is a **non-parametric, instance-based (lazy) learning** algorithm: it stores the training data and defers computation until prediction time.

### Core idea

To predict for a new point, find the **k nearest training points** (neighbours) and:
- **Classification:** assign the **majority class** among the k neighbours (plurality vote).
- **Regression:** assign the **average (or weighted average)** of the neighbours' values.

### Steps

1. Choose the number of neighbours `k` and a distance metric.
2. Compute the distance from the query point to all training points.
3. Select the `k` points with the smallest distance.
4. Aggregate: majority vote (classification) or mean (regression).

### Distance metrics

- **Euclidean:** `√( Σ (xᵢ − yᵢ)² )`  (most common)
- **Manhattan (city-block):** `Σ |xᵢ − yᵢ|`
- **Minkowski:** `( Σ |xᵢ − yᵢ|^p )^(1/p)` (p=2 → Euclidean, p=1 → Manhattan)
- **Cosine** — for high-dimensional vectors/text.
- **Hamming** — for categorical/binary features.

### Choosing k

- **Small k** (e.g., 1): sensitive to noise/outliers → high variance, low bias (overfitting).
- **Large k**: smoother decision boundary → high bias, low variance; may blur class boundaries.
- **Rule of thumb:** `k ≈ √n`; choose odd k for binary problems to avoid ties; select by cross-validation.

### Weighted KNN

Give closer neighbours more influence — weight by `1/distance` (or `1/distance²`). Helps when the neighbourhood is unevenly distributed.

### The curse of dimensionality

As dimensionality grows, distances between points become similar, so "nearest" loses meaning. Mitigate with feature selection, PCA, or a domain-specific distance metric. **Always scale features** before using KNN (distance is scale-sensitive).

### Worked example

Training points (1D): class A = {1, 2, 3}, class B = {7, 8, 9}. Query x = 4.
- With `k = 3`, the three nearest are {1, 2, 3} → all class A → predict **A**.
- With `k = 5`, neighbours are {1, 2, 3, 7, 8} → A: 3 votes, B: 2 votes → predict **A**.
- With `k = 1`, nearest is 3 → predict **A**.

### Pros/cons

- **Pros:** simple, no training phase, naturally handles multi-class, non-linear boundaries, easy to understand.
- **Cons:** slow at prediction (must scan all data), memory-heavy, sensitive to irrelevant features and scale, needs a good `k`, no model/interpretability of global structure.

---

# Cross-Module Summary Tables

## Learning paradigms at a glance

| Paradigm | Labels | Output | Main tasks | Example algorithms |
|---|---|---|---|---|
| Supervised | Yes | Prediction | Classification, regression | Linear/Logistic Regression, KNN, SVM, Trees |
| Unsupervised | No | Structure | Clustering, dim. reduction | k-Means, PCA, Apriori |
| Semi-supervised | Partial | Prediction | Classification | Self-training, co-training |
| Reinforcement | Rewards | Action policy | Control/decisions | Q-Learning, DQN |

## Module 3 algorithms at a glance

| Algorithm | Task | Model type | Key idea | Main strength | Main weakness |
|---|---|---|---|---|---|
| Simple linear regression | Regression | Parametric | Fit a straight line (OLS) | Simple, interpretable | Only linear, single feature |
| Multiple linear regression | Regression | Parametric | Fit a hyperplane | Handles several features | Multicollinearity, linearity |
| Logistic regression | Classification | Parametric | Sigmoid of linear combo | Probabilities, interpretable | Linear decision boundary |
| Naïve Bayes | Classification | Probabilistic | Bayes + independence | Fast, small data, text | Independence assumption |
| KNN | Both | Instance-based | Majority/average of k neighbours | Simple, non-linear | Slow, scale/curse sensitive |

## Evaluation metrics quick reference

| Task | Metrics |
|---|---|
| Classification | Accuracy, Precision, Recall, F1, ROC-AUC (confusion matrix) |
| Regression | MSE, RMSE, MAE, R², Adjusted R² |
| Clustering | Silhouette score, Davies–Bouldin, inertia |
| General | Cross-validation accuracy, bias–variance analysis |

---

*End of notes. These cover the full scope of Modules 1–3 of the syllabus.*
