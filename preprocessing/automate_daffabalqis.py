# automate_daffabalqis.py

"""
File ini berisi fungsi otomatisasi preprocessing dataset Heart Disease UCI sesuai notebook eksperimen manual.
Fungsi utama:
- preprocess_data(filepath): menerima path csv dataset raw, menghapus duplikat, melakukan scaling, dan mengembalikan DataFrame hasil preprocess.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(filepath: str) -> pd.DataFrame:
    # Check if the input file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    X = df_clean.drop(columns=["target"], errors='ignore')
    y = df_clean.get("target", pd.Series())  # Handle missing 'target' column gracefully
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    df_processed = pd.concat([df_scaled, y.reset_index(drop=True)], axis=1)
    return df_processed

if __name__ == "__main__":
    input_path = "preprocessing/heart.csv"  # Correct input path
    output_path = "preprocessing/heart_preprocessed.csv"

    # Ensure the preprocessing directory exists
    os.makedirs("preprocessing", exist_ok=True)

    try:
        # Preprocess the data and save the output
        df_preprocessed = preprocess_data(input_path)
        df_preprocessed.to_csv(output_path, index=False)
        print(f"Hasil preprocessing tersimpan di: {output_path}")
    except Exception as e:
        print(f"ERROR: {e}")



