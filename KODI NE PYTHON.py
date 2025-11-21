KODI NE PYTHON 

import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Dataset-et
datasets = {
    "Iris": load_iris(),
    "Wine": load_wine(),
    "Breast Cancer": load_breast_cancer(),
    "Digits": load_digits()
}

# Algoritmet
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Artificial Neural Network": MLPClassifier(max_iter=300)
}

# Lista për rezultatet
results = []

for dname, dataset in datasets.items():
    X, y = dataset.data, dataset.target
    # Normalizim vetëm për KNN, SVM dhe MLP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    for mname, model in models.items():
        # Wine dataset nuk përdor KNN sipas planit, mund të kontrollohet
        if dname == "Wine" and mname == "K-Nearest Neighbors":
            continue
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            "Dataset": dname,
            "Algoritëm": mname,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4)
        })

# Konvertimi në DataFrame dhe rregullimi i renditjes
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by=["Dataset", "Algoritëm"]).reset_index(drop=True)

# Shfaqja në një format të bukur
print(df_results.to_string(index=False))