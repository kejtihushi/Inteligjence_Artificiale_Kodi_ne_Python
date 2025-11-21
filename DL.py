from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Emri i dataset-it
dataset_name = "Digits Dataset"

# Ngarkimi i dataset-it
digits = load_digits()
X, y = digits.data, digits.target

# Normalizimi i të dhënave
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ndarja në train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Krijimi i modelit MLP
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300)

# Trajnimi
mlp.fit(X_train, y_train)

# Predikimi
y_pred = mlp.predict(X_test)

# Matja e performancës
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Printimi i rezultateve
print(f"Dataset: {dataset_name}")
print("Algoritëm: MLP (Deep Learning)")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")
