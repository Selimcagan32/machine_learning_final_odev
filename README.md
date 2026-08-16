# Müşteri Kayıp (Churn) Tahmini

## Amaç
Bir telekom şirketinin müşteri verilerini kullanarak hangi müşterilerin
hizmeti bırakma (churn) olasılığının yüksek olduğunu tahmin etmek.
Problem bir **sınıflandırma (classification)** problemidir; hedef
değişken `Churn` (Evet/Hayır).

## Veri Seti
[Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(`WA_Fn-UseC_-Telco-Customer-Churn.csv`)

7043 müşteri, 21 sütun. Demografik bilgiler (cinsiyet, yaş grubu, eş/bakmakla
yükümlü kişi durumu), abone olunan hizmetler (internet, telefon, güvenlik,
yedekleme vb.), sözleşme/fatura bilgileri ve hedef değişken olarak churn
durumunu içerir.

## Nasıl Çalıştırılır
1. Veri setini yukarıdaki linkten indirin ve `WA_Fn-UseC_-Telco-Customer-Churn.csv`
   adıyla proje klasörüne koyun.
2. Gerekli kütüphaneleri kurun:
   ```
   pip install -r requirements.txt
   ```
3. Scripti çalıştırın:
   ```
   python musteri_kayip_tahmini.py
   ```
4. Çıktılar konsola yazdırılır; grafikler ekranda gösterilir ve
   `outputs/` klasörüne PNG olarak kaydedilir.

## Yöntem
Veri okuma → keşifsel analiz → eksik değer temizliği → kategorik encoding →
aykırı değer incelemesi → ölçekleme → öznitelik mühendisliği (avg_monthly_spend,
tenure_group, total_services) → öznitelik seçimi → train/validation/test
ayrımı (stratify ile) → 4 model eğitimi (Logistic Regression, KNN, Decision
Tree, Random Forest) → validation karşılaştırması → GridSearchCV ile
hiperparametre ayarlama → test değerlendirmesi → açıklanabilirlik (feature
importance / katsayı yorumu).

## Sonuçlar
Validation performansına göre en iyi model **Logistic Regression** oldu.
Hiperparametre ayarlaması sonrası test verisinde:

| Metrik | Değer |
|---|---|
| Accuracy | 0.81 |
| Precision | 0.68 |
| Recall | 0.56 |
| F1-Score | 0.61 |
| ROC-AUC | 0.86 |

**Yorum:** Churn oranı veri setinde %26.5 ile dengesiz olduğu için ROC-AUC
(0.86) en güvenilir metrik; model churn eden ile etmeyen müşteriyi iyi
ayırt edebiliyor. Recall (0.56) görece düşük — bu, gerçek churn eden
müşterilerin bir kısmının kaçırıldığı anlamına gelir ve iş açısından
iyileştirilmesi öncelikli bir alandır. En etkili öznitelikler arasında
sözleşme tipi (iki yıllık sözleşme churn'ü azaltıyor), fiber internet
hizmeti (churn'ü artırıyor) ve müşteri kıdemi (tenure, churn'ü azaltıyor)
öne çıkıyor. Modelin sınırlılığı: veri setinde yalnızca demografik ve
hizmet bilgisi var; müşteri memnuniyeti veya şikayet verileri eklenirse
tahmin gücü artabilir.
