# automate_daffabalqis.py

"""
File ini berisi fungsi otomatisasi preprocessing dataset Heart Disease UCI sesuai notebook eksperimen manual.
Fungsi utama:
- preprocess_data(filepath): menerima path csv dataset raw, menghapus duplikat, melakukan scaling, dan mengembalikan DataFrame hasil preprocess.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import os 

def preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Membaca dataset csv, hapus duplikat, lakukan scaling fitur numerik, kembalikan DataFrame siap train.
    """
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
    input_path = "heart.csv"  # atau folder/heart.csv yang sesuai lokasi file
    output_dir = "preprocessing"
    output_path = os.path.join(output_dir, "heart_preprocessed.csv")
    
    df_processed = preprocess_data(input_path)

    os.makedirs(output_dir, exist_ok=True)
    df_processed.to_csv(output_path, index=False)

    print(f"Preprocessing selesai. Dataset hasil preprocess disimpan di: {output_path}")

