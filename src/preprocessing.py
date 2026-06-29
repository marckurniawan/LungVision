import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# Path dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "lung_cancer_dataset.csv"


def load_preprocessed_data():

    
    df = pd.read_csv(DATA_PATH)

    print("Dataset berhasil dibaca.\n")
    print(df.head())

    print("\nUkuran Dataset:", df.shape)

    print("\nJumlah Missing Value:")
    print(df.isnull().sum())

    
    df.drop(
        columns=["Patient Id", "Level"],
        errors="ignore",
        inplace=True
    )

    
    encoder = LabelEncoder()

    categorical_columns = df.select_dtypes(include=["object"]).columns

    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])

   
    numeric_columns = df.select_dtypes(include=np.number).columns

    imputer = SimpleImputer(strategy="median")

    df[numeric_columns] = imputer.fit_transform(
        df[numeric_columns]
    )

    
    X = df.drop(columns=["Result"])

    y = df["Result"]

    print("\nJumlah Feature :", X.shape[1])

    print("\nTarget:")
    print(y.value_counts())

   
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    print("\nData Train :", X_train.shape)

    print("Data Test  :", X_test.shape)

    
    print("\nKolom Object Setelah Encoding:")

    print(df.select_dtypes(include=["object"]).columns)

   
    print("\nMissing Value Setelah Preprocessing:")

    print(df.isnull().sum().sum())

    

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled
    )