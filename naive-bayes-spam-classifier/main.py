
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#loading the data
df = pd.read_csv("spam.csv", encoding="latin-1")

# print(df.head())

#correcting colums v1,v2 to labels, message
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

#features and labels
x = df["message"]
y = df["label"]


#explore the data
# print("\n", df.shape)
# print("\n", df.columns)

# print(df["label"].value_counts())

#Convert text into Numbers
vectorizer = CountVectorizer()

X_vectorized = vectorizer.fit_transform(x)

# print(vectorizer.get_feature_names_out()[:20])

#Train/Test Split
X_train, X_test,  y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

#Train Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

#making the prediction
predictions = model.predict(X_test)

#evaluation
accuracy = accuracy_score(y_test, predictions)

# print("Accuracy:", accuracy)

#My experiments
message = [
    "Son of a bitch, I am working on the api. wait for my sake"
]

message_vector = vectorizer.transform(message)

result = model.predict(message_vector)

print(result)