import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

# 1. Ngarkimi i dataset-it
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalizimi dhe reshaping
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 2. Ndërtimi i modelit CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Trajnimi i modelit
model.fit(X_train, y_train_cat, epochs=5, batch_size=64, validation_split=0.2, verbose=2)

# 4. Predikimi mbi test set
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = y_test

# 5. Llogaritja e metrikave
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

# 6. Rezultati në DataFrame për raport
results_df = pd.DataFrame([{
    "Dataset": "Fashion MNIST",
    "Algoritëm": "CNN",
    "Accuracy": round(acc,4),
    "Precision": round(prec,4),
    "Recall": round(rec,4),
    "F1-Score": round(f1,4)
}])

print(results_df.to_string(index=False))
