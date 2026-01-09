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

# Algoritmet (me random_state për qëndrueshmëri)
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Artificial Neural Network": MLPClassifier(max_iter=300, random_state=42)
}

# Modelet që kërkojnë normalizim
models_need_scaling = [
    "K-Nearest Neighbors",
    "Support Vector Machine",
    "Artificial Neural Network"
]

results = []

for dname, dataset in datasets.items():
    X, y = dataset.data, dataset.target

    # Split fillestar (pa scaling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for mname, model in models.items():

        # Wine dataset nuk përdor KNN sipas planit
        if dname == "Wine" and mname == "K-Nearest Neighbors":
            continue

        # Normalizim vetëm kur është e nevojshme
        if mname in models_need_scaling:
            scaler = StandardScaler()
            X_train_used = scaler.fit_transform(X_train)
            X_test_used = scaler.transform(X_test)
        else:
            X_train_used = X_train
            X_test_used = X_test

        # Trajnimi dhe parashikimi
        model.fit(X_train_used, y_train)
        y_pred = model.predict(X_test_used)

        # Metrikat
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

# Rezultatet finale
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by=["Dataset", "Algoritëm"]).reset_index(drop=True)

print(df_results.to_string(index=False))
