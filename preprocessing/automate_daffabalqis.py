# automate_daffabalqis.py

"""
File ini berisi fungsi otomatisasi preprocessing dataset Heart Disease UCI sesuai notebook eksperimen manual.
Fungsi utama:
- preprocess_data(filepath): menerima path csv dataset raw, menghapus duplikat, melakukan scaling, dan mengembalikan DataFrame hasil preprocess.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    X = df_clean.drop(columns=["target"])
    y = df_clean["target"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    df_processed = pd.concat([df_scaled, y.reset_index(drop=True)], axis=1)
    return df_processed

if __name__ == "__main__":
    input_path = "heart.csv"  # Sesuai lokasi download dataset di workflow
    output_path = "preprocessing/heart_preprocessed.csv"  # Simpan di folder preprocessing

    df_preprocessed = preprocess_data(input_path)
    df_preprocessed.to_csv(output_path, index=False)
    print(f"Hasil preprocessing tersimpan di: {output_path}")


