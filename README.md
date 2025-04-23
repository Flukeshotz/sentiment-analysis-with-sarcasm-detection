# Sentiment Analysis with Sarcasm Detection

This project combines **Sentiment Analysis** and **Sarcasm Detection** to build a robust NLP pipeline that can analyze the underlying sentiment of text while intelligently identifying sarcastic content.

## 🧠 Project Overview

Sentiment analysis can often be skewed by sarcastic language, which expresses negative emotions using positive wording. This project addresses that by:

- Merging multiple datasets from Twitter, Reddit, and sarcasm headlines  
- Preprocessing and balancing the data  
- Training a **Logistic Regression** model using **TF-IDF** vectorization  
- Integrating **VADER Sentiment Analysis** for sentiment scoring  
- Predicting both sarcasm and sentiment from custom user input

---

## 📁 Dataset Sources

The project utilizes four main datasets:
1. `Reddit_Data.csv`
2. `Twitter_Data.csv`
3. `Cleaned_Sarcasm_Headlines_Dataset.json`
4. `Sarcasm_Headlines_Dataset_v2.json`

Each dataset is cleaned, normalized, and combined into a unified dataframe for training.

---

## 🔧 Tech Stack

- **Language**: Python 3  
- **Libraries**:  
  - `pandas`, `numpy`  
  - `sklearn`  
  - `nltk` (with VADER Sentiment Analyzer)  
- **Model**: Logistic Regression  
- **Feature Engineering**: TF-IDF Vectorizer  

---

## ⚙️ Key Steps

1. **Data Loading & Cleaning**: Load and standardize formats across all datasets  
2. **Upsampling**: Handle class imbalance using resampling  
3. **Feature Extraction**: Convert text data into numerical form with `TfidfVectorizer`  
4. **Model Training**: Use `LogisticRegression` in a scikit-learn pipeline  
5. **Evaluation**: Assess performance with accuracy and classification report  
6. **Custom Prediction**: Input text is analyzed for both sentiment and sarcasm  

---

## 📈 Results

The model achieved:
- **Accuracy**: 92%  
- **Precision/Recall/F1-Score**: Well-balanced across sarcastic and non-sarcastic classes  

**Classification Report**:
          precision    recall  f1-score   support

     0.0       0.90      0.94      0.92     16571
     1.0       0.94      0.89      0.91     16725

accuracy                           0.92     33296

macro avg       0.92      0.92      0.92     33296
weighted avg       0.92      0.92      0.92     33296

---

## 💬 Try It Yourself

After training, run the script and enter your own text:

```bash
Enter text to check for sarcasm and sentiment (or 'exit' to quit):
Input: "Oh great, another Monday morning!"
Prediction: Sarcastic
Sentiment: Negative

📚 Future Improvements
	•	Implement Deep Learning (e.g., LSTM or BERT) for better context understanding
	•	Build a web interface using Flask/Streamlit
	•	Multilingual sarcasm detection support

⸻

👨‍💻 Author

Harsh Vardhan Singh
⸻

📜 License

This project is licensed under the MIT License.

⸻

🌐 Acknowledgements
	•	NLTK
	•	Scikit-learn
	•	VADER Sentiment
	•	Sarcasm datasets from Kaggle and other open sources

⸻
