import streamlit as st
import joblib
import re

# Page Configuration
st.set_page_config(
    page_title="BankRoute - AI Complaint Gateway",
    page_icon="🏦",
    layout="centered"
)

# 1. Text Preprocessing Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"x{2,}", " ", text)
    text = re.sub(r"\{\$?[\d.,]+\}", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 2. Rule-based Fallback Priority Function (Agar priority .pkl file na ho)
# Updated & Smart Rule Engine Logic
def get_rule_priority(text):
    text_lower = text.lower()
    
    # 1. High Priority Triggers (Fraud, money loss, unauthorized access)
    high_keywords = [
        'fraud', 'stolen', 'unauthorized', 'without my permission', 
        'lost money', 'lost', 'lawsuit', 'urgent', 'scam', 
        'identity theft', 'hacked', 'stolen money', 'unapproved'
    ]
    
    # 2. Medium Priority Triggers (Fees, delays, updates)
    medium_keywords = ['delay', 'fee', 'charge', 'pending', 'issue', 'wrong amount']
    
    # Priority Evaluation
    if any(word in text_lower for word in high_keywords):
        return "🔴 HIGH"
    elif any(word in text_lower for word in medium_keywords) or len(text) > 400:
        return "🟡 MEDIUM"
    else:
        return "🟢 LOW"

# 3. Load Saved Assets
@st.cache_resource
def load_models():
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    cat_model = joblib.load('category_model.pkl')
    
    try:
        priority_model = joblib.load('priority_model.pkl')
    except Exception:
        priority_model = None
        
    return tfidf, cat_model, priority_model

tfidf_vec, category_clf, priority_clf = load_models()

# 4. Interface Header
st.title("🏦 BankRoute")
st.caption("AI-Driven Classification & Priority Routing Gateway")
st.write("Customer narrative input .")

st.divider()

# Input UI
user_narrative = st.text_area(
    "Customer Complaint Narrative:", 
    height=150, 
    placeholder="Type or paste the complaint text here..."
)

# Execution Button
if st.button("Route & Classify Complaint", type="primary"):
    if user_narrative.strip():
        # Clean and vectorize
        cleaned_input = clean_text(user_narrative)
        vectorized_input = tfidf_vec.transform([cleaned_input])
        
        # Predict Category
        predicted_cat = category_clf.predict(vectorized_input)[0]
        
        # Display Category
        st.subheader("📌 Classification Result")
        st.success(f"**Predicted Category:** {predicted_cat}")
        
        # Display Priority (Model base ya Rule base)
        if priority_clf:
            predicted_priority = priority_clf.predict(vectorized_input)[0]
            st.info(f"**Priority Level (ML Model):** {predicted_priority}")
        else:
            rule_priority = get_rule_priority(user_narrative)
            st.info(f"**Priority Level (Rule Engine):** {rule_priority}")
            
    else:
        st.warning("⚠️ Please enter a complaint narrative before classifying.")