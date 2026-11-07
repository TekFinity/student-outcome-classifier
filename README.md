# Student Outcome Predictor: ML Assignment 2

- **Submitted by:** Asmita Chhanchure (BITS ID: 2026K05172)
- **Live app:** https://student-outcome-classifier.streamlit.app/
- **GitHub repo:** https://github.com/TekFinity/student-outcome-classifier

---

## a. Problem Statement

Universities lose many students every year, and the earlier an at-risk student is identified, the sooner
support (mentoring, fee relief, academic counselling) can be offered. The goal of this project is to
**predict a student's academic outcome (Dropout, Enrolled or Graduate)** from information available
at enrolment and from their first-year academic performance. This is a **3-class classification**
problem. Five classical ML classifiers are trained, compared using six evaluation metrics, and served
through an interactive Streamlit web application.

## b. Dataset Description

| Item | Detail |
|---|---|
| Name | *Predict Students' Dropout and Academic Success* |
| Source | UCI Machine Learning Repository, ID 697 (Realinho, Machado, Baptista & Martins, 2021) https://archive.ics.uci.edu/dataset/697 |
| Instances | **4,424** students from a Portuguese higher-education institution |
| Features | **36** raw features (+ 6 engineered = 42 used by the models) |
| Target | `Target`: **Dropout** (1,421 · 32.1 %), **Enrolled** (794 · 17.9 %), **Graduate** (2,209 · 49.9 %) |
| Missing values | None |

**Feature groups**

- *Demographic:* marital status, nationality, gender, age at enrolment, displaced, international
- *Socio-economic:* mother's/father's qualification and occupation, debtor, tuition fees up to date, scholarship holder
- *Academic background:* application mode/order, course, day/evening attendance, previous qualification and grade, admission grade
- *Academic performance:* curricular units credited/enrolled/evaluated/approved/grade/without evaluation for the 1st and 2nd semester
- *Macro-economic:* unemployment rate, inflation rate, GDP

**Engineered features** (in `features.py`): `pass_rate_sem1`, `pass_rate_sem2` (approved ÷ enrolled units),
`pass_rate_trend`, `grade_trend` (semester 2 minus semester 1), `total_units_approved`, `financial_risk` (debtor OR fees not up to date).

**Split:** stratified 80 / 20 train/test (`random_state = 3621`). The 885-row test split
(Graduate 442, Dropout 284, Enrolled 159) is saved as `test_data.csv`.

## c. Repository Structure

```
student-outcome-classifier/
├── app.py                  # Streamlit app (loads saved models, never retrains)
├── train_models.py         # trains + tunes + evaluates the 5 models, saves artifacts
├── features.py             # shared cleaning & feature engineering (train + app)
├── requirements.txt
├── README.md
├── test_data.csv           # 20 % hold-out test set (features + Target)
├── data/students_raw.csv   # original UCI file (';' separated)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    ├── feature_columns.json
    └── metrics_summary.csv
```

## d. Models Used & Comparison Table

All models are sklearn `Pipeline`s. Scale-sensitive models include a `StandardScaler`.
Precision, Recall and F1 are **weighted averages**. AUC is **one-vs-rest, weighted**.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7311 | 0.8892 | **0.7663** | 0.7311 | 0.7426 | 0.5784 |
| Decision Tree | 0.7446 | 0.8686 | 0.7650 | 0.7446 | 0.7437 | 0.5858 |
| KNN | 0.7514 | 0.8610 | 0.7465 | 0.7514 | 0.7366 | 0.5884 |
| Naive Bayes (Gaussian) | 0.7175 | 0.8388 | 0.7091 | 0.7175 | 0.7099 | 0.5315 |
| Support Vector Machine (RBF) | **0.7650** | **0.8944** | 0.7556 | **0.7650** | **0.7535** | **0.6096** |

**Tuned hyper-parameters**

| Model | Best parameters |
|---|---|
| Logistic Regression | C = 0.3, class_weight = balanced |
| Decision Tree | criterion = gini, max_depth = 4, min_samples_leaf = 5 |
| KNN | k = 7, Manhattan distance (p = 1), distance weights |
| Naive Bayes | GaussianNB, var_smoothing = 1e-3 |
| SVM | RBF kernel, C = 3.0, gamma = 0.01 |

**Per-class recall on the test set** (supports the observations below)

| Model | Dropout | Enrolled | Graduate |
|---|---|---|---|
| Logistic Regression | 0.697 | **0.610** | 0.796 |
| Decision Tree | 0.627 | 0.528 | 0.898 |
| KNN | 0.673 | 0.365 | **0.941** |
| Naive Bayes | 0.690 | 0.371 | 0.860 |
| SVM | **0.739** | 0.396 | 0.914 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Good overall, but not the most accurate (73 %). It is the **best at spotting students who are still "Enrolled"** (finds 61 % of them), because we told it to give equal importance to the smaller class. The downside: it wrongly labels some graduates as "Enrolled". |
| Decision Tree | A simple, easy-to-understand model (only 4 levels of yes/no questions). Accuracy is 74 %. It is **good at finding graduates** but **misses many dropouts**: 56 students who actually dropped out were predicted as "Graduate". |
| KNN | Predicts by looking at the 7 most similar students. Accuracy is 75 %, but it **mostly guesses "Graduate"** because that is the biggest group. It finds 94 % of graduates but only 37 % of enrolled students. |
| Naive Bayes | **The weakest model** (72 % accuracy, lowest on every metric). It assumes every feature is unrelated to the others, which is not true here (for example, 1st and 2nd semester marks are closely linked). It is fast but too simple for this data. |
| Support Vector Machine | **The best model**: highest accuracy (76.5 %), F1 and MCC. It can learn complex patterns in the data and is the **best at catching students who will drop out** (finds 74 % of them), which is the most useful thing for a university. Like the others, it finds it hard to identify "Enrolled" students. |
| **Overall winner for this dataset** | **Support Vector Machine.** It scored the best on 5 out of 6 metrics and is the best at finding dropouts. Logistic Regression comes second and is better if finding "Enrolled" students is more important. All models found the **"Enrolled" group hardest** to predict, because it is the smallest group and these students are "in between" dropouts and graduates. |

## e. Streamlit App Features

- **CSV upload** of test data (`test_data.csv`)
- **Model selection dropdown** for the 5 classifiers
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC
- **Confusion matrix** and **classification report** for the selected model
- **Comparison table** of all 5 models on the uploaded test data

## f. How to Run Locally

```bash
pip install -r requirements.txt
python train_models.py      # optional: regenerates model/ and test_data.csv
streamlit run app.py
```
