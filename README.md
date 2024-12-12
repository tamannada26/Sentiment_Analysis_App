

# Amazon Fine Food Reviews Sentiment Analysis 

## Project Overview
This project analyzes sentiment in the **Amazon Fine Food Reviews** dataset using Natural Language Processing (NLP) and Machine Learning techniques. The dataset consists of customer reviews, and the goal is to predict sentiment (positive, negative, neutral) while addressing data imbalance challenges.

## Dataset
The dataset is sourced from Kaggle and includes over **500,000 reviews** of fine food products sold on Amazon. Each review contains the following attributes:
- **Id**: Unique identifier for the review
- **ProductId**: Unique identifier for the product
- **UserId**: Unique identifier for the user
- **ProfileName**: Profile name of the reviewer
- **Helpfulness**: Ratio of helpful votes to total votes
- **Score**: Numerical rating of the product
- **Time**: Time of the review (UNIX timestamp)
- **Summary**: Summary of the review
- **Text**: Full text of the review

## Project Workflow
### 1. Data Preprocessing
- **Text Cleaning**: Removed HTML tags, special characters, and stopwords.
- **Tokenization**: Split text into individual words for analysis.
- **Lemmatization**: Reduced words to their base forms.
- **Vectorization**: Converted text data into numerical format using **TF-IDF Vectorizer**.

### 2. Data Visualization
- Explored class distribution, word frequencies, and review length.
- Visualized insights using libraries such as **matplotlib** and **seaborn**.

### 3. Addressing Class Imbalance
- Handled imbalanced data using **SMOTE** (Synthetic Minority Oversampling Technique).

### 4. Sentiment Analysis Models
Two models were implemented for sentiment classification:
1. **Logistic Regression**
2. **Random Forest Classifier**

### 5. Model Evaluation
- Metrics: **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
- Results showed that the **Random Forest Classifier** outperformed Logistic Regression in all metrics.

## Key Results
| Metric         | Logistic Regression | Random Forest |
|----------------|----------------------|---------------|
| Accuracy       | 82%                 | 87%           |
| Precision      | 63% (neutral)       | 73% (neutral) |
| Recall         | 62% (neutral)       | 68% (neutral) |
| F1-Score       | 63% (neutral)       | 70% (neutral) |

## Technologies Used
- **Python**: Core programming language
- **Libraries**: pandas, NumPy, scikit-learn, matplotlib, seaborn, nltk, imbalanced-learn
- **Tools**: Jupyter Notebook, Git, Kaggle

## File Structure
```
project-folder/
├── data/
│   ├── amazon_fine_food_reviews.csv
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── evaluation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
├── results/
│   ├── visualizations/
│   ├── metrics_summary.txt
└── README.md
```

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/tamannada26/amazon-sentiment-analysis.git
   ```
2. Navigate to the project folder:
   ```bash
   cd amazon-sentiment-analysis
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebooks in the `notebooks/` directory to preprocess data, train models, and evaluate results.

## Future Work
- Explore additional models such as **Gradient Boosting** and **Neural Networks**.
- Implement more advanced techniques for text preprocessing and feature engineering.
- Experiment with sentiment-specific embeddings like **BERT**.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- Kaggle for the dataset
- Scikit-learn and NLTK for NLP and machine learning libraries
