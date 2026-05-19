import numpy as np
from spam_assassin.utils_fer import email2TokenList
from sklearn.model_selection import train_test_split
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def email_2_binary_vector(email, vocab):
    tokens = email2TokenList(email)
    
    vector = np.zeros(len(vocab))
    
    for token in tokens:
        if token in vocab:
            vector[vocab[token] - 1] = 1
            
    return vector

from spam_assassin.utils_fer import getVocabDict
vocab = getVocabDict()

email1 = open('spam_assassin/spam/0001.txt', 'r', encoding= 'utf_8').read()
features1 = email_2_binary_vector(email1, vocab)
print(features1.shape)

def load_email_folder(folder_path, label, vocab):
    X = []
    y = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                email_content = f.read()
            
            vector = email_2_binary_vector(email_content, vocab)
            X.append(vector)
            y.append(label)
            
    return np.array(X), np.array(y)

X_spam, y_spam = load_email_folder('spam_assassin/spam', 1, vocab)
X_easy_ham, y_easy_ham = load_email_folder('spam_assassin/easy_ham', 0, vocab)
X_hard_ham, y_hard_ham = load_email_folder('spam_assassin/hard_ham', 0, vocab)

X = np.vstack((X_spam, X_easy_ham, X_hard_ham))
y = np.concatenate((y_spam, y_easy_ham, y_hard_ham))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print(f"Dimensiones de X_train: {X_train.shape}")
print(f"Dimensiones de X_test: {X_test.shape}")

# Merge ham (¡Usamos np.concatenate para juntar los vectores 'y' (1D)!)
X_ham = np.vstack([X_easy_ham, X_hard_ham])
y_ham = np.concatenate([y_easy_ham, y_hard_ham])

# Merge all datasets in a single dataset
X = np.vstack([X_spam, X_ham])
y = np.concatenate([y_spam, y_ham])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)

clf = LinearSVC(C=1.0)
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)

print(f"Train accuracy: {acc_train:.4f}")
print(f"Test accuracy: {acc_test:.4f}")

pesos = clf.coef_.ravel()
