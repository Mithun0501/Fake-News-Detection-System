# 📰 Fake News Detection Using Machine Learning

## 📌 Project Overview

This project is a Machine Learning and Natural Language Processing (NLP) based web application that detects whether a news article is **Real** or **Fake**.

The system is trained on a large dataset containing real and fake news articles and uses TF-IDF vectorization with Logistic Regression for classification.

---

## 🚀 Features

* Detects Fake and Real News
* User-friendly Web Interface
* Confidence Score Display
* Machine Learning Based Prediction
* Fast and Accurate Results
* Flask Web Application

---

## 🛠️ Technologies Used

* Python
* Flask
* Pandas
* Scikit-Learn
* TF-IDF Vectorizer
* Logistic Regression
* HTML
* CSS
* Bootstrap

---

## 📂 Dataset

Dataset Files:

* Fake.csv
* True.csv

Dataset Columns:

* title
* text
* subject
* date

Labels:

* 0 → Fake News
* 1 → Real News

---

## ⚙️ Machine Learning Workflow

### 1. Data Collection

Load Fake.csv and True.csv datasets.

### 2. Data Preprocessing

* Combine title and text columns
* Remove missing values
* Convert text into numerical features

### 3. Feature Extraction

TF-IDF Vectorization is used to transform text data into numerical vectors.

### 4. Model Training

Logistic Regression is used for classification.

### 5. Prediction

The trained model predicts whether the entered news article is Real or Fake.

---

## 📁 Project Structure

```text
Fake_News_Detection
│
├── app.py
├── train.py
├── evaluate.py
├── Fake.csv
├── True.csv
├── model.pkl
├── vectorizer.pkl
│
├── templates
│   └── index.html
│
└── static
```

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Fake-News-Detection.git
```

Install required packages:

```bash
pip install pandas scikit-learn flask joblib
```

---

## ▶️ Run the Project

Train the model:

```bash
python train.py
```

Start Flask server:

```bash
python app.py
```

Open Browser:

```text
http://127.0.0.1:5000
```

---

## 📸 Sample Output

Input:

```text
The Federal Reserve announced that interest rates will remain unchanged following its latest policy meeting.
```

Output:

```text
🟢 Real News
Confidence: 96.5%
```

Input:

```text
Aliens have secretly taken control of all world governments.
```

Output:

```text
🔴 Fake News
Confidence: 98.3%
```

---

## 🎯 Future Enhancements

* Deep Learning Models (LSTM, BERT)
* Real-Time News Verification
* News Source Credibility Analysis
* Multi-Language Support
* News URL Verification

---

## 👨‍💻 Author

Mithun V

Artificial Intelligence & Machine Learning Internship Project

---

## 📜 Conclusion

This project successfully classifies news articles as Real or Fake using Machine Learning and NLP techniques. The system demonstrates how text classification can be applied to combat misinformation and improve information reliability.
