# Bank-Route
### AI-Driven Classification and Priority-Based Routing of Banking Customer Complaints

Bank-Route is a machine learning project that analyzes customer complaints written in plain English, predicts their financial product category, estimates their priority, and suggests the appropriate banking department.

**Institution:** Sukkur IBA University  
**Course:** Machine Learning  
**Instructor:** Dr. Ismail Mangrio

---

## 1. Problem Statement

Banks receive thousands of customer complaints that require manual classification, priority assessment, and department assignment.

This process can be time-consuming and inconsistent, potentially delaying urgent cases involving fraud or unauthorized transactions.

Bank-Route explores how machine learning can automate the initial complaint triage process.

## 2. Proposed Solution

A supervised machine learning pipeline that processes complaint text and produces three outputs:

- **Category:** Identifies the financial product associated with the complaint.
- **Priority:** Classifies urgency as High, Medium, or Low.
- **Department:** Suggests the responsible banking department based on the predicted category.

### ML Pipeline

```text
Customer Complaint
        |
        ▼
   Text Cleaning
        |
        ▼
  TF-IDF Vectorization
        |
        ▼
 Machine Learning Models
        |
   ┌────┴────┐
   ▼         ▼
Category  Priority
   |
   ▼
Department Mapping
```

The system provides recommendations for human review. Final decisions remain with banking staff.

---

## 3. Model Performance

**Dataset split:** 80% Training (13,496) | 20% Testing (3,374)

### Category Classification

Nine financial product categories.

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| **Linear SVM** | **83.88%** | **83.86%** |
| Logistic Regression | 83.70% | 83.75% |
| Random Forest | 80.79% | 80.64% |
| Naive Bayes | 75.43% | 72.41% |

**Selected model:** Linear SVM

- Best-performing categories: Mortgage and Student Loan (F1 = 0.93).
- Weakest category: Payday Loan (F1 = 0.69).
- Common confusion: Consumer Loan and Debt Collection.

### Priority Classification

Three priority levels: High, Medium, Low.

| Model | Accuracy | Weighted F1 | High Recall |
|---|---:|---:|---:|
| **Logistic Regression** | **66.21%** | **66.39%** | **71.74%** |
| Random Forest | 65.17% | 64.62% | 41.12% |
| Linear SVM | 63.81% | 63.86% | 56.68% |
| Naive Bayes | 62.18% | 59.27% | 15.06% |

**Selected model:** Logistic Regression

High-priority recall is evaluated separately because missing an urgent complaint is more consequential than incorrectly flagging a normal complaint.

---

## 4. Dataset

**Source:** [US Consumer Financial Protection Bureau (CFPB) Complaints Dataset](https://www.kaggle.com/datasets/kaggle/us-consumer-finance-complaints)

| Dataset Detail | Count |
|---|---:|
| Raw records | 555,957 |
| Records with narratives | 66,806 |
| Final selected dataset | 16,870 |
| Product categories | 9 |
| Training records | 13,496 |
| Testing records | 3,374 |

### Preprocessing

- Removed records without customer narratives.
- Selected nine relevant financial product categories.
- Capped large categories at 2,500 records.
- Cleaned complaint text.
- Applied TF-IDF with unigrams and bigrams.
- Used 20,000 maximum features.

### Priority Label Creation

The original dataset does not contain priority labels. A rule-based rubric was developed using the `issue` field.

| Priority | Examples |
|---|---|
| High | Fraud, unauthorized transactions, identity theft, foreclosure, debt not owed |
| Medium | Billing disputes, fees, payments, account closing, ATM problems |
| Low | Remaining issues |

The `issue` field was used only to generate priority labels. The priority model uses complaint narratives as input.

---

## 5. Department Mapping

Predicted product categories are mapped to departments using a predefined routing table.

| Product Category | Department |
|---|---|
| Bank Account or Service | Accounts & Deposits |
| Credit Card / Prepaid Card | Card Services |
| Consumer Loan / Payday Loan | Retail Lending |
| Mortgage | Mortgage & Home Loans |
| Student Loan | Education Loans |
| Debt Collection | Recovery & Collections |
| Money Transfers | Payments & Remittances |

---

## 6. Technology Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **NLP:** TF-IDF Vectorization
- **Algorithms:** Linear SVM, Logistic Regression, Random Forest, Naive Bayes
- **Visualization:** Matplotlib
- **Model Persistence:** Joblib
- **Web Application:** Streamlit

---

## 7. Repository Structure

```text
Bank-Route/
│
├── app.py
├── category_model.pkl
├── tfidf_vectorizer.pkl
├── usa-endtoend-consumer-complaints-project.ipynb
└── AzharAli_ProjectProposal.pdf
```

| File | Description |
|---|---|
| `app.py` | Streamlit web application |
| `category_model.pkl` | Trained SVM classifier |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `.ipynb` | Data analysis, preprocessing, training and evaluation |
| `.pdf` | Original project proposal |

---

## 8. Run Locally

**Clone the repository**

```bash
git clone YOUR_REPOSITORY_URL
cd Bank-Route
```

**Install dependencies**

```bash
pip install streamlit scikit-learn joblib pandas numpy matplotlib
```

**Run the application**

```bash
streamlit run app.py
```

Enter a customer complaint and click **Route & Classify Complaint**.

The dataset is not included in the repository. Download it separately from Kaggle to reproduce the notebook.

---

## 9. Current Limitations

- The Streamlit application uses Linear SVM for category prediction.
- Priority prediction currently uses keyword-based rules. The evaluated Logistic Regression model is not integrated.
- Department mapping is implemented in the notebook but not displayed in the application.
- Priority labels are based on a team-designed rubric, not actual bank-assigned priorities.
- The dataset contains US complaints and may not represent Pakistani banking complaints.
- The system supports single-label classification.

---

## 10. Team

| Name | Role |
|---|---|
| Azhar Ali | Team Lead |
| Khadija Shahid | Data Analyst |
| Faryal Gul | System Design |

**Instructor:** Dr. Ismail Mangrio  
**Institution:** Sukkur IBA University

---

**Bank-Route | Machine Learning | Natural Language Processing | Financial Technology**
