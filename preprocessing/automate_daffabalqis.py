# automate_daffabalqis.py

"""
File ini berisi fungsi otomatisasi preprocessing dataset Heart Disease UCI sesuai notebook eksperimen manual.
Fungsi utama:
- preprocess_data(filepath): menerima path csv dataset raw, menghapus duplikat, melakukan preprocessing dengan pipeline, dan mengembalikan DataFrame hasil preprocess.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    X = df_clean.drop(columns=["target"])
    y = df_clean["target"]
    
    # Membuat pipeline preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    X_processed = numeric_transformer.fit_transform(X)
    df_processed = pd.DataFrame(X_processed, columns=X.columns)
    
    df_final = pd.concat([df_processed, y.reset_index(drop=True)], axis=1)
    return df_final

if __name__ == "__main__":
    input_path = "heart.csv"  # Sesuai lokasi download dataset di workflow
    output_path = "preprocessing/heart_preprocessing.csv"  # Simpan di folder preprocessing

    df_preprocessed = preprocess_data(input_path)
    df_preprocessed.to_csv(output_path, index=False)
    print(f"Hasil preprocessing tersimpan di: {output_path}")
