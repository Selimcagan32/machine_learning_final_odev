"""
Müşteri Kayıp (Churn) Tahmini - Uçtan Uca Makine Öğrenmesi Projesi
=====================================================================

AMAÇ
-----
Bu proje, bir telekom şirketinin müşteri verilerini kullanarak hangi
müşterilerin hizmeti bırakma (churn) olasılığının yüksek olduğunu tahmin
etmeyi amaçlar. Problem bir SINIFLANDIRMA (classification) problemidir;
hedef değişken müşterinin "Churn" (Evet/Hayır) durumudur.

Veri seti, Kaggle üzerindeki "Telco Customer Churn" veri setidir
(https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
Bu dosya, "WA_Fn-UseC_-Telco-Customer-Churn.csv" adıyla proje ile aynı
klasörde bulunmalıdır; kod bu dosyayı okur ve üzerinde çalışır.

KULLANILAN KÜTÜPHANELER
-------------------------
- pandas, numpy            : veri okuma / işleme
- matplotlib, seaborn       : görselleştirme
- scikit-learn               : ön işleme, modelleme, değerlendirme,
                                hiperparametre arama

KURULUM
--------
Gerekli kütüphaneleri kurmak için terminalde şu komutu çalıştırın:

    pip install pandas numpy matplotlib seaborn scikit-learn

(Aynı klasördeki requirements.txt dosyasıyla da kurulabilir:
    pip install -r requirements.txt
)

ÇALIŞTIRMA ADIMLARI
----------------------
1. Kaggle'dan "WA_Fn-UseC_-Telco-Customer-Churn.csv" dosyasını indirip
   bu script ile aynı klasöre (Colab'da /content/ dizinine) koyun.
2. Yukarıdaki kurulum adımını uygulayın (kütüphaneler daha önce
   kurulmadıysa).
3. Google Colab veya yerel Python ortamında bu dosyayı çalıştırın:
       python musteri_kayip_tahmini.py
4. Çıktılar konsola yazdırılır; grafikler hem ekranda bir pencerede
   gösterilir hem de "outputs/" klasörüne PNG olarak kaydedilir.

Proje aşağıdaki adımları sırasıyla uygular:
    Veri okuma -> Keşifsel analiz -> Eksik değer temizliği ->
    Kategorik encoding -> Aykırı değer incelemesi -> Ölçekleme ->
    Öznitelik mühendisliği -> Öznitelik seçimi -> Train/Val/Test ayrımı ->
    Model eğitimi (3 model) -> Validation karşılaştırması ->
    Hiperparametre ayarlama -> Test değerlendirmesi -> Yorumlama ->
    Açıklanabilirlik (bonus)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# 2) VERİ SETİNİ OKUMA
# =====================================================================
# Çözülen problem: Bir telekom şirketinin müşterilerinin hizmeti bırakıp
# bırakmayacağını (churn) tahmin etmek. Bu sayede şirket, kaybetme riski
# yüksek müşterilere önceden aksiyon alabilir.
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("\n" + "=" * 70)
print("3) HEDEF DEĞİŞKEN VE PROBLEM TÜRÜ")
print("=" * 70)
print("Hedef değişken: 'Churn' (Evet/Hayır)")
print("Problem türü: SINIFLANDIRMA (binary classification)")


# =====================================================================
# 4) TEMEL VERİ İNCELEME
# =====================================================================
print("\n" + "=" * 70)
print("4) TEMEL VERİ İNCELEME")
print("=" * 70)
print("\nİlk 5 satır:")
print(df.head())
print(f"\nSatır sayısı: {df.shape[0]}, Sütun sayısı: {df.shape[1]}")
print("\nVeri tipleri:")
print(df.dtypes)
print("\nSayısal değişkenlerin temel istatistikleri:")
print(df.describe())


# =====================================================================
# 5) EKSİK DEĞER KONTROLÜ
# =====================================================================
print("\n" + "=" * 70)
print("5) EKSİK DEĞER KONTROLÜ")
print("=" * 70)

# TotalCharges bazen Kaggle veri setinde boşluk karakteri olarak gelir, sayıya çeviriyoruz
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("Eksik değer sayıları:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# TotalCharges'taki eksik değerleri medyan ile dolduruyoruz (aykırı değerlere dayanıklı)
median_total = df["TotalCharges"].median()
df["TotalCharges"] = df["TotalCharges"].fillna(median_total)
print(f"\n'TotalCharges' sütunundaki eksik değerler medyan ({median_total:.2f}) ile dolduruldu.")

# customerID modelleme için anlamsız, kaldırıyoruz
df = df.drop(columns=["customerID"])


# =====================================================================
# 6) KATEGORİK DEĞİŞKENLERİ ENCODE ETME
# =====================================================================
print("\n" + "=" * 70)
print("6) KATEGORİK DEĞİŞKENLERİN ENCODE EDİLMESİ")
print("=" * 70)

# Hedef değişkeni ayrı encode ediyoruz (Yes=1, No=0)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

categorical_cols = df.select_dtypes(include="object").columns.tolist()
print(f"Kategorik sütunlar: {categorical_cols}")

# İkili (binary) kategorik sütunlar için Label Encoding,
# çok sınıflı olanlar için One-Hot Encoding kullanıyoruz
binary_cols = [c for c in categorical_cols if df[c].nunique() == 2]
multi_cols = [c for c in categorical_cols if df[c].nunique() > 2]

le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
print(f"İkili sütunlar (Label Encoding): {binary_cols}")
print(f"Çok sınıflı sütunlar (One-Hot Encoding): {multi_cols}")
print(f"Encoding sonrası sütun sayısı: {df.shape[1]}")


# =====================================================================
# 7) AYKIRI DEĞER İNCELEMESİ
# =====================================================================
print("\n" + "=" * 70)
print("7) AYKIRI DEĞER İNCELEMESİ")
print("=" * 70)

numeric_check_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
for col in numeric_check_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} aykırı değer bulundu (IQR yöntemiyle).")

print(
    "\nYorum: Churn veri setinde tenure ve ücret değişkenleri doğal olarak "
    "geniş bir aralığa yayılır (yeni ile uzun süreli müşteriler arası fark). "
    "Aykırı değerler iş açısından anlamlı olduğu (örn. çok yüksek faturalı "
    "müşteriler) için silinmedi; bunun yerine ağaç tabanlı modeller "
    "(Random Forest, Decision Tree) aykırı değerlere doğal olarak dayanıklı "
    "olduğundan veri olduğu gibi bırakıldı, sadece ölçekleme uygulanacak "
    "modellerde (LR, KNN) StandardScaler ile etkisi azaltıldı."
)

# Görselleştirme
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, numeric_check_cols):
    sns.boxplot(y=df[col], ax=ax, color="skyblue")
    ax.set_title(col)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "outlier_boxplots.png"))
plt.show()


# =====================================================================
# 9) ÖZNİTELİK MÜHENDİSLİĞİ (en az 2 yeni öznitelik)
# =====================================================================
print("\n" + "=" * 70)
print("9) ÖZNİTELİK MÜHENDİSLİĞİ")
print("=" * 70)

# 1) Ortalama aylık harcama oranı: TotalCharges / (tenure + 1)  -> sıfıra bölmeyi önlemek için +1
df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

# 2) Tenure grubu: müşteri kıdemini kategorilere ayırıyoruz (0-1yıl, 1-3yıl, 3-5yıl, 5+yıl)
def tenure_group(t):
    if t <= 12:
        return 0  # yeni müşteri
    elif t <= 36:
        return 1  # orta kıdem
    elif t <= 60:
        return 2  # kıdemli
    else:
        return 3  # çok kıdemli

df["tenure_group"] = df["tenure"].apply(tenure_group)

# 3) (Bonus ek öznitelik) Toplam hizmet sayısı: müşterinin aldığı ek hizmet sayısı
service_cols = [c for c in df.columns if any(
    s in c for s in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                      "TechSupport", "StreamingTV", "StreamingMovies"]
) and "No" not in c]
if service_cols:
    df["total_services"] = df[service_cols].sum(axis=1)

print("Eklenen yeni öznitelikler: 'avg_monthly_spend', 'tenure_group', 'total_services'")


# =====================================================================
# 8) ÖLÇEKLEME (scaling gerektiren modeller için)
# =====================================================================
print("\n" + "=" * 70)
print("8) ÖLÇEKLEME")
print("=" * 70)

X = df.drop(columns=["Churn"])
y = df["Churn"]

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend"]
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
print(f"Ölçeklenen sayısal sütunlar: {numeric_cols}")
print("(Not: Random Forest / Decision Tree ölçeklemeye ihtiyaç duymaz, "
      "Logistic Regression ve KNN için ölçekli veri kullanılacak.)")


# =====================================================================
# 10) ÖZNİTELİK SEÇİMİ
# =====================================================================
print("\n" + "=" * 70)
print("10) ÖZNİTELİK SEÇİMİ")
print("=" * 70)

# Korelasyon tabanlı öznitelik seçimi: hedefle en düşük korelasyona sahip
# ve birbirleriyle çok yüksek korelasyonlu (>0.95) tekrarlayan sütunları çıkar
corr_with_target = X_scaled.corrwith(y).abs().sort_values(ascending=False)
print("Hedef değişkenle en yüksek korelasyona sahip 10 öznitelik:")
print(corr_with_target.head(10))

# Neredeyse sıfır varyanslı sütunları eleme
low_variance_cols = [c for c in X_scaled.columns if X_scaled[c].var() < 0.01]
if low_variance_cols:
    print(f"\nDüşük varyanslı olduğu için elenen sütunlar: {low_variance_cols}")
    X_scaled = X_scaled.drop(columns=low_variance_cols)
    X = X.drop(columns=low_variance_cols)
else:
    print("\nDüşük varyanslı sütun bulunamadı.")

print(f"Öznitelik seçimi sonrası kalan öznitelik sayısı: {X_scaled.shape[1]}")


# =====================================================================
# 11) TRAIN / VALIDATION / TEST AYRIMI
# =====================================================================
print("\n" + "=" * 70)
print("11) TRAIN / VALIDATION / TEST AYRIMI")
print("=" * 70)

X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=RANDOM_STATE, stratify=y_temp
)  # 0.1765 * 0.85 ~ 0.15 -> yaklaşık %70/%15/%15 ayrım

print(f"Train: {X_train.shape[0]} satır")
print(f"Validation: {X_val.shape[0]} satır")
print(f"Test: {X_test.shape[0]} satır")
print(f"Train churn oranı: {y_train.mean():.3f} | Val: {y_val.mean():.3f} | Test: {y_test.mean():.3f}")


# =====================================================================
# 12) MODEL EĞİTİMİ (3+ model)
# =====================================================================
print("\n" + "=" * 70)
print("12) MODEL EĞİTİMİ")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} eğitildi.")


# =====================================================================
# 13) VALIDATION KARŞILAŞTIRMASI
# =====================================================================
print("\n" + "=" * 70)
print("13) VALIDATION SONUÇLARINA GÖRE MODEL KARŞILAŞTIRMASI")
print("=" * 70)

val_results = []
for name, model in trained_models.items():
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
    val_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, preds),
        "Precision": precision_score(y_val, preds),
        "Recall": recall_score(y_val, preds),
        "F1-Score": f1_score(y_val, preds),
        "ROC-AUC": roc_auc_score(y_val, proba) if proba is not None else np.nan,
    })

val_df = pd.DataFrame(val_results).sort_values("F1-Score", ascending=False)
print(val_df.to_string(index=False))

best_model_name = val_df.iloc[0]["Model"]
print(f"\nValidation F1-Score'a göre en iyi model: {best_model_name}")


# =====================================================================
# 14) HİPERPARAMETRE AYARLAMA (GridSearch)
# =====================================================================
print("\n" + "=" * 70)
print("14) HİPERPARAMETRE AYARLAMA (GRID SEARCH)")
print("=" * 70)

param_grids = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    },
    "K-Nearest Neighbors": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
    },
    "Decision Tree": {
        "max_depth": [3, 5, 6, 8, 10],
        "min_samples_split": [2, 5, 10],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 8, 12, None],
        "min_samples_split": [2, 5],
    },
}

base_model = models[best_model_name].__class__()
grid = GridSearchCV(
    base_model,
    param_grids[best_model_name],
    scoring="f1",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    n_jobs=-1,
)
# Grid search icin train+validation birlestirilerek cross-validation yapiliyor
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])
grid.fit(X_train_full, y_train_full)

print(f"En iyi hiperparametreler ({best_model_name}): {grid.best_params_}")
print(f"En iyi cross-validation F1-Score: {grid.best_score_:.4f}")

best_model = grid.best_estimator_


# =====================================================================
# 15) TEST VERİSİ ÜZERİNDE DEĞERLENDİRME
# =====================================================================
print("\n" + "=" * 70)
print("15) TEST VERİSİ ÜZERİNDE FİNAL DEĞERLENDİRME")
print("=" * 70)

test_preds = best_model.predict(X_test)
test_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

acc = accuracy_score(y_test, test_preds)
prec = precision_score(y_test, test_preds)
rec = recall_score(y_test, test_preds)
f1 = f1_score(y_test, test_preds)
auc = roc_auc_score(y_test, test_proba) if test_proba is not None else np.nan

print(f"Test Accuracy : {acc:.4f}")
print(f"Test Precision: {prec:.4f}")
print(f"Test Recall   : {rec:.4f}")
print(f"Test F1-Score : {f1:.4f}")
print(f"Test ROC-AUC  : {auc:.4f}")

cm = confusion_matrix(y_test, test_preds)
print("\nConfusion Matrix:")
print(cm)
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, test_preds, target_names=["Kalan (No)", "Kayıp (Yes)"]))

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Kalan", "Kayıp"], yticklabels=["Kalan", "Kayıp"])
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title(f"Confusion Matrix - {best_model_name} (Test)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()


# =====================================================================
# 16) MODEL SONUCUNUN YORUMLANMASI
# =====================================================================
print("\n" + "=" * 70)
print("16) MODEL SONUCUNUN YORUMLANMASI")
print("=" * 70)
print(f"""
- Validation aşamasında en yüksek F1-Score'u '{best_model_name}' modeli verdi
  ve hiperparametre ayarlaması sonrası test verisinde {f1:.3f} F1-Score,
  {acc:.3f} accuracy elde edildi.
- Recall ({rec:.3f}) değeri, churn riski taşıyan müşterilerin ne kadarının
  doğru yakalandığını gösterir; churn tahmininde recall'u yüksek tutmak
  genelde işletme için daha kritiktir (kaybedilecek müşteriyi kaçırmamak).
- Precision ({prec:.3f}) ise "churn" dediğimiz müşterilerin gerçekten kaç
  tanesinin churn olduğunu gösterir; düşükse gereksiz müşteri koruma
  kampanyalarına bütçe harcanabilir.
- Modelin sınırlılıkları: Veri setinde sadece demografik ve hizmet bilgisi
  var; müşteri memnuniyet anketleri, çağrı merkezi şikayetleri gibi
  davranışsal veriler eklenirse tahmin gücü artabilir.
""")


# =====================================================================
# 17) BONUS: AÇIKLANABİLİRLİK (Feature Importance)
# =====================================================================
print("\n" + "=" * 70)
print("17) BONUS - AÇIKLANABİLİRLİK")
print("=" * 70)

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X_scaled.columns)
    importances = importances.sort_values(ascending=False).head(10)
    print("En önemli 10 öznitelik (feature importance):")
    print(importances)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances.values, y=importances.index, color="steelblue")
    plt.title(f"{best_model_name} - Öznitelik Önem Dereceleri")
    plt.xlabel("Önem Skoru")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.show()

elif hasattr(best_model, "coef_"):
    coefs = pd.Series(best_model.coef_[0], index=X_scaled.columns)
    coefs = coefs.sort_values(key=abs, ascending=False).head(10)
    print("En etkili 10 katsayı (Logistic Regression):")
    print(coefs)
    print("""
Yorum: Pozitif katsayılar churn olasılığını artıran, negatif katsayılar
churn olasılığını azaltan değişkenlerdir. Örneğin kontrat tipi
'Month-to-month' genellikle churn'ü artırırken, 'tenure' (kıdem) genellikle
churn'ü azaltır.
""")
else:
    print("Seçilen model için doğrudan bir açıklanabilirlik çıktısı (importance/coef) bulunmuyor.")
    print("Alternatif olarak mutual information skorları kullanılabilir:")
    mi_scores = mutual_info_classif(X_train_full, y_train_full, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi_scores, index=X_scaled.columns).sort_values(ascending=False).head(10)
    print(mi_series)

print("\n" + "=" * 70)
print(f"Tüm grafikler '{OUTPUT_DIR}/' klasörüne kaydedildi.")
print("Proje tamamlandı.")
print("=" * 70)
