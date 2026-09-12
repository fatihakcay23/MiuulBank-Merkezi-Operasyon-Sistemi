"""
Veri Hazırlama Betiği
----------------------
BankChurners.csv (Kaggle "Credit Card Customer Churn Prediction" veri seti)
üzerinde temel temizleme işlemlerini uygular ve app.py'nin SQL modülünün
beklediği BankChurners_Cleaned.csv dosyasını üretir.

Kullanım:
    python data_prep.py
    # veya farklı bir girdi dosyası için:
    python data_prep.py --input /path/to/BankChurners.csv
"""

import argparse
import os

import pandas as pd


def clean_bank_churners(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    # 1. Gereksiz (çöp) sütunların temizlenmesi
    # Veri setinin sonunda Naive Bayes sınıflandırıcı çıktısı olan ve
    # modelleme için gereksiz olan iki sütun bulunuyor.
    cols_to_drop = [c for c in df.columns if "Naive_Bayes" in c]
    df = df.drop(columns=cols_to_drop)

    # 2. Sütun isimlerinin standartlaştırılması (SQL ve Python uyumu için)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # 3. Hedef değişkenin (target) hazırlanması
    # attrition_flag: "Existing Customer" / "Attrited Customer" -> 0 / 1
    df["churn_label"] = df["attrition_flag"].apply(
        lambda x: 1 if x == "Attrited Customer" else 0
    )

    # 4. Metin sütunlarındaki gereksiz boşlukların temizlenmesi
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].str.strip()

    null_counts = df.isnull().sum().sum()

    df.to_csv(output_path, index=False)

    print(f"Veri temizlendi ve '{output_path}' olarak kaydedildi.")
    print(f"Toplam Satır: {df.shape[0]}")
    print(f"Toplam Sütun: {df.shape[1]}")
    print(f"Toplam Eksik Değer (Null): {null_counts}")
    print("\nSütun İsimleri (Yeni Hali):")
    print(list(df.columns))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BankChurners.csv veri temizleme betiği")
    parser.add_argument(
        "--input",
        default=os.environ.get("MIUUL_DATA_PATH", "BankChurners.csv"),
        help="Ham BankChurners.csv dosyasının yolu (varsayılan: proje klasöründeki BankChurners.csv)",
    )
    parser.add_argument(
        "--output",
        default="BankChurners_Cleaned.csv",
        help="Temizlenmiş çıktı dosyasının adı",
    )
    args = parser.parse_args()

    clean_bank_churners(args.input, args.output)
