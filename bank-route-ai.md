# Bank-Route

**AI-driven classification and priority-based routing of banking customer complaints.**

Bank-Route reads a customer complaint written in plain English, predicts which financial product it is about, and flags how urgent it is, so that serious cases such as fraud or unauthorized transactions can reach the right team first.

Built as a Machine Learning course project at Sukkur IBA University (Instructor: Dr Ismail Mangrio).

---

## The Problem

Banks receive thousands of complaints every day. Staff read each one manually, decide its type and urgency, and forward it to a department. This is slow, and different employees make different decisions. Urgent cases can sit in the queue.

## The Solution

A supervised machine learning pipeline that takes the complaint text and predicts:

1. **Category**: which financial product the complaint is about (for example Mortgage or Credit card).
2. **Priority**: High, Medium or Low.
3. **Department**: where the complaint should be routed.

```
Complaint text -> Text cleaning -> TF-IDF -> ML models -> Category + Priority -> Department
```

The system only **suggests** a result to staff. The final decision stays with a human.

---

## Results

Dataset split: 80% training (13,496 complaints), 20% testing (3,374 complaints).

### Category prediction (9 classes)

| Model | Accuracy | Weighted F1 |
|---|---|---|
| **SVM (LinearSVC)** | **0.8388** | **0.8386** |
| Logistic Regression | 0.8370 | 0.8375 |
| Random Forest | 0.8079 | 0.8064 |
| Naive Bayes | 0.7543 | 0.7241 |

The SVM is the deployed category model. It predicts Mortgage and Student loan best (F1 = 0.93) and Payday loan worst (F1 = 0.69). Most errors happen between categories that share vocabulary, such as Consumer Loan and Debt collection.

### Priority prediction (High / Medium / Low)

| Model | Accuracy | Weighted F1 | Recall of High |
|---|---|---|---|
| **Logistic Regression** | **0.6621** | **0.6639** | **0.7174** |
| Random Forest | 0.6517 | 0.6462 | 0.4112 |
| SVM (LinearSVC) | 0.6381 | 0.6386 | 0.5668 |
| Naive Bayes | 0.6218 | 0.5927 | 0.1506 |

Recall of the High class is tracked separately because missing an urgent complaint is worse than wrongly flagging a normal one.

---

## Dataset

- **Name:** US Consumer Finance Complaints (CFPB), on Kaggle
- **Link:** https://www.kaggle.com/datasets/kaggle/us-consumer-finance-complaints
- **License:** U.S. Government work, public domain
- **Raw size:** 555,957 complaints, of which 66,806 include the customer's written narrative. Only those rows can be used for text classification.
- **Final sample:** 16,870 complaints across 9 bank-related product categories, with each large class capped at 2,500 to reduce imbalance.
- **Excluded categories:** Credit reporting (concerns credit bureaus, not banks) and Other financial service (only 109 rows).

The dataset is not included in this repository. Download it from Kaggle to re-run the notebook.

### How priority labels were created

The dataset has no priority column, so we built a transparent rule-based rubric from the `issue` field:

- **High:** fraud, unauthorized transactions, identity theft, illegal threats, foreclosure, debt not owed
- **Medium:** billing disputes, fees, payments, account closing, loan servicing, ATM problems
- **Low:** everything else

The models never see the `issue` column. They learn urgency only from the complaint text.

### Department routing

The category is mapped to a department using a fixed table defined by the team:

| Category | Department |
|---|---|
| Bank account or service | Accounts & Deposits Team |
| Credit card, Prepaid card | Card Services Team |
| Consumer Loan, Payday loan | Retail Lending Team |
| Mortgage | Mortgage & Home Loans Team |
| Student loan | Education Loans Team |
| Debt collection | Recovery & Collections Team |
| Money transfers | Payments & Remittances Team |

---

## Repository Structure

```
.
├── app.py                                          Streamlit web app
├── category_model.pkl                              Trained category classifier (SVM)
├── tfidf_vectorizer.pkl                            Trained TF-IDF vectorizer
├── usa-endtoend-consumer-complaints-project.ipynb  Full pipeline: EDA, training, evaluation
└── AzharAli_ProjectProposal.pdf                    Original project proposal
```

---

## Run the App Locally

**1. Install dependencies**

```bash
pip install streamlit scikit-learn joblib
```

**2. Start the app** (from the folder that contains `app.py` and both `.pkl` files)

```bash
streamlit run app.py
```

**3. Use it:** paste a complaint into the text box and click **Route & Classify Complaint**.

Note: the `.pkl` files were saved with scikit-learn on Kaggle. If loading fails, install the same scikit-learn version that was used for training.

---

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, TF-IDF (unigrams and bigrams, 20,000 features), Streamlit, Joblib

---

## Current Status and Limitations

- **The Streamlit app uses a rule-based priority engine.** It matches keywords such as "fraud" or "unauthorized" in the text. The ML priority model (Logistic Regression) was evaluated in the notebook but is not saved or loaded by the app. If a `priority_model.pkl` file is added, the app uses it automatically.
- **The app shows category and priority.** The department mapping is implemented in the notebook's `triage()` function but is not shown in the app yet.
- **Priority labels are proxies.** They were created by our own rubric, not by real bank staff, so priority accuracy (about 66%) should be treated as a suggestion only.
- **US data.** The model is trained on CFPB complaints, not on Pakistani bank complaints, so wording and categories differ from local systems.
- **Single-label output.** Complaints that describe several problems at once are hard to classify with one category.
- **Department mapping is an assumption.** Real banks organize their teams differently.

## Future Work

- Save the trained priority model and use it in the app.
- Show the routed department in the app.
- Train on real Pakistani complaints, including Urdu and Roman Urdu text.
- Replace TF-IDF with transformer embeddings.
- Add human review for low-confidence predictions.

---

## Team

| Name | Role |
|---|---|
| Azhar Ali | Team Lead |
| Khadija Shahid | Data Analyst |
| Faryal Gul | System Design |
