# Naive-Bayes-spam-classifier
This project is a machine learning-based spam detection system built using Python and Scikit-Learn.

The model is trained on SMS messages and classifies incoming messages as either **Spam** or **Ham (Not Spam)** using the Naive Bayes algorithm.

## Features

* SMS spam detection
* Text preprocessing and feature extraction
* Bag-of-Words representation using CountVectorizer
* Multinomial Naive Bayes classification
* Model evaluation using classification metrics
* Custom message prediction

## Dataset

This project uses the SMS Spam Collection dataset, which contains labeled SMS messages categorized as:

* **Spam** – unwanted or promotional messages
* **Ham** – legitimate messages

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* CountVectorizer
* Multinomial Naive Bayes

## Project Workflow

1. Load and explore the dataset
2. Extract features and labels
3. Convert text into numerical features using CountVectorizer
4. Split data into training and testing sets
5. Train a Multinomial Naive Bayes model
6. Evaluate model performance
7. Predict labels for new messages

## Example

**Input**

```text
Congratulations! You have won a free iPhone.
```

**Prediction**

```text
Spam
```

**Input**

```text
Hey, are we still meeting tomorrow?
```

**Prediction**

```text
Ham
```

## Results

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd naive-bayes-spam-classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python main.py
```

Enter a message or modify the sample messages in the source code to test predictions.

## Learning Outcomes

This project demonstrates practical understanding of:

* Supervised Learning
* Binary Classification
* Text Vectorization
* Feature Engineering
* Model Training and Evaluation
* Natural Language Processing Fundamentals
* Machine Learning Workflow with Scikit-Learn

## Future Improvements

* TF-IDF vectorization
* Model comparison and hyperparameter tuning
* FastAPI REST API integration
* Web-based user interface
* Email spam detection and Gmail integration
* Real-time prediction service

## License

This project is intended for educational and portfolio purposes.
