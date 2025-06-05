# 🔍 **Fake Review Identification System**

## 💡 Objective

The core aim of this project is to **identify fake product reviews** from a large dataset encompassing reviews across multiple categories such as *Home and Office*, *Sports*, etc. Each entry in the dataset comprises the review text, associated rating, and a tag specifying its nature – either CG (Computer Generated) or OR (Original, written by a human).

The central goal is to determine whether a review is **genuine or artificially generated**. Reviews labeled as CG are flagged as fake, while OR-tagged ones are considered authentic.

---

## 🧾 Dataset Details

The dataset consists of **40,000 total reviews**, with an even split:

* **20,000 genuine reviews** (OR)
* **20,000 computer-generated (fake) reviews** (CG)

<ul>
  <li>OR = Authentic product reviews written by real users</li>
  <li>CG = Synthetic reviews generated through automated systems</li>
</ul>

---

## 📦 Python Modules & Dependencies Utilized

<ul>
  <li><strong>numpy</strong> – For numerical operations</li>
  <li><strong>pandas</strong> – For data manipulation and analysis</li>
  <li><strong>matplotlib.pyplot</strong> – For visualizing data</li>
  <li><strong>seaborn</strong> – For advanced plotting and statistical visuals</li>
  <li><strong>warnings</strong> – To suppress warning messages</li>
  <li><strong>nltk</strong> and <strong>nltk.corpus</strong> – For natural language processing and corpus tools</li>
  <li><strong>string</strong> – For character and string manipulation</li>
  <li><strong>sklearn.naive_bayes</strong> – For Naive Bayes classification</li>
  <li><strong>sklearn.feature_extraction</strong> – To extract features from text</li>
  <li><strong>sklearn.model_selection</strong> – For model training and evaluation splits</li>
  <li><strong>sklearn.ensemble</strong> – For ensemble learning models</li>
  <li><strong>sklearn.tree</strong> – For decision tree-based classifiers</li>
  <li><strong>sklearn.linear_model</strong> – For linear classifiers like logistic regression</li>
  <li><strong>sklearn.svc</strong> – For support vector classification</li>
  <li><strong>sklearn.neighbors</strong> – For implementing K-Nearest Neighbors</li>
</ul>

---

## 🧹 Text Cleaning & Preprocessing Methods

To enhance the quality and relevance of the text data before feeding it into ML models, the following preprocessing steps were applied:

<ul>
  <li>Elimination of punctuation marks</li>
  <li>Conversion of all characters to lowercase</li>
  <li>Filtering out stopwords (common but insignificant words)</li>
  <li>Application of stemming to reduce words to their base/root form</li>
  <li>Lemmatization to bring words to their dictionary form</li>
  <li>Removal of numerical digits from the reviews</li>
</ul>

---

## 🔄 Text Vectorization and Feature Engineering Techniques

Textual content was transformed into numerical form using the following techniques:

<ul>
  <li><strong>CountVectorizer</strong> – Implements the Bag of Words approach for text-to-number conversion</li>
  <li><strong>TF-IDF (Term Frequency-Inverse Document Frequency)</strong> – Assigns weights to terms based on their importance across documents</li>
</ul>

---

## 🤖 Algorithms Implemented for Review Classification

The following machine learning models were used to classify the reviews:

<ol>
  <li><strong>Logistic Regression</strong></li>
  <li><strong>K-Nearest Neighbors (KNN)</strong></li>
  <li><strong>Support Vector Classifier (SVC)</strong></li>
  <li><strong>Decision Tree Classifier</strong></li>
  <li><strong>Random Forest Classifier</strong></li>
  <li><strong>Multinomial Naive Bayes</strong></li>
</ol>

---

## 📊 Model Performance and Accuracy Comparison

<p>The <strong>Support Vector Machine (SVC)</strong> achieved the highest accuracy, predicting fake reviews with a precision of approximately <strong>88%</strong>. Next in performance was <strong>Logistic Regression</strong>, which achieved slightly over <strong>86%</strong>. Both the <strong>Random Forest Classifier</strong> and the <strong>Multinomial Naive Bayes</strong> model followed closely, with an accuracy rate near <strong>84%</strong>. The <strong>Decision Tree</strong> classifier yielded satisfactory results with just over <strong>73%</strong> accuracy. In contrast, the <strong>K-Nearest Neighbors</strong> model underperformed, with an accuracy of nearly <strong>58%</strong>, making it the least effective among the tested models.</p>

---

Would you like a **README.md** file version of this text or a **voice-over script** for your YouTube explanation next?
