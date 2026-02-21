# 🎫 AI Support Ticket Classifier

> Automatically classify IT service tickets into **8 categories** and predict **priority levels** for customer support tickets using NLP and Machine Learning — built end-to-end in Python.

<br>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-4B8BBE?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

---

## 📌 Project Overview

This project builds a **complete machine learning pipeline** from raw support tickets to a live, deployable classifier. It covers:

- 🏷️ **IT Ticket Classification** — 8 categories (Hardware, HR Support, Access, Storage, etc.)
- 🚨 **Priority Prediction** — High / Medium / Low for customer support tickets
- 🧹 **NLTK Preprocessing Pipeline** — tokenize → filter → lemmatize → TF-IDF
- 🤖 **4 ML models trained and benchmarked** side by side
- 📊 **Executive Dashboard** with KPI cards, confusion matrices, and CV results
- 💾 **Exported models** (pickle) + downloadable ZIP of all outputs

**Built for:** Future Interns — Machine Learning Task 2 (2026)

---

## 📊 Results

| Metric | Value |
|---|---|
| 📂 Total Tickets Processed | **56,306** |
| 🏷️ IT Category — Weighted F1 | **84.5%** |
| 🔁 5-Fold Cross-Validation F1 | **85.1% ± 0.2%** |
| ⚡ Inference Time | **< 5ms per ticket** |
| 🏆 Best Model | **Logistic Regression** |

### Model Comparison — IT Category Classification

| Model | Accuracy | Precision | Recall | Weighted F1 |
|---|---|---|---|---|
| ⭐ Logistic Regression | 0.8450 | 0.8455 | 0.8450 | **0.8451** |
| Linear SVM | 0.8423 | 0.8430 | 0.8423 | 0.8424 |
| Random Forest | 0.8342 | 0.8350 | 0.8342 | 0.8343 |
| Naive Bayes | 0.7713 | 0.7719 | 0.7713 | 0.7712 |

---

## 🗂️ Repository Structure

```
support-ticket-classifier/
│
├── 📓 support_ticket_classifier.ipynb   ← Full notebook (13 steps)
│
├── 📊 outputs/
│   ├── 01_eda_overview.png              ← Dataset EDA charts
│   ├── 02_wordclouds.png                ← Top keywords per category
│   ├── 03_confusion_matrices.png        ← IT + CS evaluation heatmaps
│   ├── 04_model_comparison.png          ← 4-model benchmark bar chart
│   ├── 05_cross_validation.png          ← 5-fold CV line chart
│   ├── 06_feature_importance.png        ← TF-IDF top terms per class
│   ├── 07_executive_dashboard.png       ← Full KPI dashboard
│   └── 08_nltk_pipeline.png             ← Preprocessing flow diagram
│
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 .gitignore
```

> ⚠️ **Datasets not included** (too large for GitHub). See the [Datasets](#-datasets) section below.

---

## 🗃️ Datasets

Two datasets are used. Upload them when running in Google Colab:

| Dataset | Rows | Cols | Target |
|---|---|---|---|
| `all_tickets_processed_improved_v3.csv` | 47,837 | 2 | `Topic_group` — 8 classes |
| `customer_support_tickets.csv` | 8,469 | 17 | `Ticket Priority` — 3 classes |

**IT Ticket Categories (8):**
Hardware · HR Support · Access · Miscellaneous · Storage · Purchase · Internal Project · Administrative Rights

**Priority Levels (3):**
🔴 High · 🟡 Medium · 🟢 Low *(Critical merged into High)*

---

## 🔧 Tech Stack

| Layer | Library / Tool |
|---|---|
| Language | Python 3.10 |
| NLP | `nltk` — word_tokenize, stopwords, WordNetLemmatizer, PorterStemmer |
| Features | `TfidfVectorizer` — 15K features, ngram(1,3), sublinear TF |
| Models | LogisticRegression, LinearSVC, RandomForestClassifier, MultinomialNB |
| Evaluation | classification_report, confusion_matrix, StratifiedKFold, cross_val_score |
| Visualization | matplotlib, matplotlib.gridspec, seaborn, wordcloud |
| Environment | Jupyter Notebook / Google Colab |
| Export | pickle, zipfile |

---

## 🧹 NLTK Preprocessing Pipeline (Step 5)

```
Raw Ticket Text
      │
      ▼   re.sub() — remove {placeholders}, URLs, emails, numbers
      │
      ▼   word_tokenize()         ← NLTK punkt tokenizer
      │
      ▼   Filter stopwords        ← NLTK corpus + 23 domain-specific words
      │                              (ticket, support, regards, dear, ...)
      ▼   WordNetLemmatizer()     ← Lemmatization (default mode)
      │   PorterStemmer()         ← Optional stemming mode
      │
      ▼   TfidfVectorizer()       ← 15,000 features, ngram(1,3), sublinear_tf
      │
      ▼   ML Model → Prediction + Confidence Score
```

**Real example from the notebook:**

```python
Input     : "My laptop keyboard stopped working after the Windows update yesterday."
Tokens    : ['my', 'laptop', 'keyboard', 'stopped', 'working', 'after', ...]
Filtered  : ['laptop', 'keyboard', 'stopped', 'working', 'windows', 'update']
Lemmatized: ['laptop', 'keyboard', 'stop', 'work', 'window', 'update']
TF-IDF    : [0.0, 0.82, 0.71, 0.0, 0.63, ...]

→ Prediction: 🖥️  Hardware  (confidence: 87.3%)
```

---

## 📓 Notebook Walkthrough — 13 Steps

| Step | Title | What It Does |
|---|---|---|
| 1 | Install & Import Libraries | `pip install nltk wordcloud`; all sklearn/matplotlib imports; dark theme palette |
| 2 | Load Datasets | Upload both CSVs via Colab file picker; auto-detects IT vs CS dataset by column names |
| 3 | Exploratory Data Analysis | Category bar chart, ticket-type pie, priority bars, word-length histogram |
| 4 | Data Quality Audit | Cross-tab analysis; discovers CS text fields carry no signal (synthetically generated) |
| 5 | Text Preprocessing with NLTK | `clean_text()` — regex cleaning → word_tokenize → stopword filter → lemmatize |
| 6 | Word Clouds | Per-category word clouds showing most discriminative terms |
| 7 | Model Training — IT Category | TF-IDF + 4 models; stratified 80/20 split; full accuracy/F1 score table |
| 8 | Model Training — CS Priority | TF-IDF + one-hot metadata → combined sparse feature matrix; 4 models |
| 9 | Evaluation | `classification_report` for both tasks; styled confusion matrices with % annotations |
| 9b | Cross-Validation | 5-fold `StratifiedKFold` on a `Pipeline`; fill-between confidence bands plotted |
| 10 | Feature Importance | Logistic Regression coefficients → top 10 TF-IDF terms per category (2×4 grid) |
| 11 | Live Inference | `classify_it_ticket()` + `classify_cs_priority()` with top-3 confidence scores |
| 12 | Executive Dashboard | 4 KPI cards + category bars + CV plot + per-class F1 heatmap in one figure |
| 13 | Save & Download | Pickle all models + vectorizers + label encoders → ZIP download |

---

## 🚀 Quick Start

### Option A — Google Colab (Recommended)

1. Upload `support_ticket_classifier.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Run **Step 1** to install dependencies
3. Run **Step 2** — upload both CSV files when prompted
4. Run all remaining steps top-to-bottom

### Option B — Local Jupyter

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/support-ticket-classifier.git
cd support-ticket-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch notebook
jupyter notebook support_ticket_classifier.ipynb
```

> **Local tip:** Replace the Colab upload cells (Steps 2–3) with:
> ```python
> df_it = pd.read_csv('all_tickets_processed_improved_v3.csv')
> df_cs = pd.read_csv('customer_support_tickets.csv')
> ```

---

## 🔍 Live Inference (Step 11)

Two ready-to-use functions are provided:

```python
# Classify an IT service ticket → category + confidence
result = classify_it_ticket(
    "My laptop screen is flickering after a Windows update."
)
# Returns:
# {
#   'category'  : 'Hardware',
#   'confidence': '89.2%',
#   'top3'      : [('Hardware','89.2%'), ('Access','5.1%'), ('Misc','3.4%')]
# }

# Predict priority for a customer support ticket
result = classify_cs_priority(
    "My account was charged twice. I need an immediate refund.",
    ticket_type='Billing inquiry',
    ticket_subject='Payment issue',
    channel='Chat',
    product='Apple AirPods'
)
# Returns:
# {
#   'priority'  : 'High',
#   'badge'     : '🔴 HIGH',
#   'confidence': '91.4%'
# }
```

---

## 📈 Output Charts

| File | Description |
|---|---|
| `01_eda_overview.png` | IT category horizontal bars · CS ticket-type pie · priority bar · ticket-length histogram |
| `02_wordclouds.png` | Top keywords for each of the 8 IT categories |
| `03_confusion_matrices.png` | Side-by-side heatmaps — IT (8×8) + CS priority (3×3) with count and % annotations |
| `04_model_comparison.png` | Grouped bar chart: Accuracy + Weighted F1 for all 4 models across both tasks |
| `05_cross_validation.png` | 5-fold CV scores per fold with mean line and confidence band |
| `06_feature_importance.png` | Top 10 LR coefficients per IT category shown in a 2×4 subplot grid |
| `07_executive_dashboard.png` | Full KPI dashboard: 4 cards + all charts in one publication-ready figure |
| `08_nltk_pipeline.png` | Visual step-by-step diagram of the NLTK preprocessing pipeline |

---

## 💾 Exported Files (Step 13)

```
saved_models/
├── tfidf_it.pkl         ← TF-IDF vectorizer for IT tickets   (15K features)
├── tfidf_cs.pkl         ← TF-IDF vectorizer for CS tickets    (8K features)
├── model_it_cat.pkl     ← Best IT category classifier (Logistic Regression)
├── model_cs_pri.pkl     ← Best CS priority classifier
├── le_it.pkl            ← LabelEncoder for 8 IT categories
└── le_pri.pkl           ← LabelEncoder for 3 priority levels

model_metrics.csv              ← All model scores in one CSV
ticket_classifier_outputs.zip  ← Everything above bundled for download
```

---

## 📦 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
wordcloud
scipy
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🤝 Connect

Built with ❤️ as part of the **Future Interns ML Internship — 2026**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/YOUR_USERNAME)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
