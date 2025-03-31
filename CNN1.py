# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('heart.csv')

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Data Visualization
sns.countplot(x=df['target'])
plt.title("Target Distribution")
plt.show()

plt.figure(figsize=(12, 6), dpi=100)
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# Splitting dataset into features and target
X = df.drop('target', axis=1).values
y = df['target'].values

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshaping for CNN input (adding a channel dimension)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Defining CNN Model
cnn = Sequential()
cnn.add(Conv1D(64, 3, padding="same", activation="relu", input_shape=(13, 1)))
cnn.add(MaxPooling1D(pool_size=2))
cnn.add(Flatten())
cnn.add(Dense(128, activation="relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(1, activation="sigmoid"))

# Compile Model
cnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# Train Model
history_cnn = cnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16, verbose=1)

# Model Evaluation
y_pred = (cnn.predict(X_test) > 0.5).astype("int32")  # Convert probabilities to binary labels

# Accuracy, Precision, Recall, F1 Score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
error_rate = 1 - accuracy

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Error Rate: {error_rate:.4f}")

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Training & Validation Accuracy and Loss Visualization
acc = history_cnn.history['accuracy']
val_acc = history_cnn.history['val_accuracy']
loss = history_cnn.history['loss']
val_loss = history_cnn.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy', color='cyan')
plt.plot(val_acc, label='Validation Accuracy', color='purple')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Training & Validation Accuracy', fontsize=14, weight='bold')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss', color='cyan')
plt.plot(val_loss, label='Validation Loss', color='purple')
plt.ylabel('Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.title('Training & Validation Loss', fontsize=14, weight='bold')
plt.legend(fontsize=12)
plt.show()
