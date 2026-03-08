#  Lineer Regresyon Analizi — Çok Veri Setli Karşılaştırma

Üç farklı veri seti üzerinde **Lineer Regresyon** uygulayarak veri yapısının model performansına etkisini inceleyen denetimli öğrenme projesi. Her veri seti aynı tam ML pipeline'ından geçirilmiş ve DummyRegressor baseline'ı ile karşılaştırılmıştır.

---

##  Proje Özeti

| Veri Seti | Hedef Değişken | R² Skoru |
|---|---|---|
|  Ames Housing | `SalePrice` (ev fiyatı) | 0.05 |
|  Wine Quality | `class` (kalite skoru) | 0.38 |
|  CPU Activity | `usr` (CPU kullanım %) | 0.88 |

---

##  Pipeline (Her Veri Setine Uygulanan)

```
Ham Veri → EDA → Ön İşleme → Train/Test Split → Ölçeklendirme → Model Eğitimi → Değerlendirme
```

**1. Keşifsel Veri Analizi (EDA)**
- Veri seti boyutu ve tip incelemesi
- Eksik değer tespiti
- Hedef değişken dağılımı
- Sayısal değişkenler arası korelasyon ısı haritası

**2. Veri Ön İşleme**
- Sayısal sütunlar için medyan, kategorik sütunlar için mod ile eksik değer doldurma
- Kategorik değişkenler için One-Hot Encoding (Ames Housing)
- Hedef değişkende IQR yöntemiyle aykırı değer temizleme

**3. Model Eğitimi**
- %80 eğitim, %20 test ayrımı (`random_state=42`)
- `StandardScaler` ile özellik ölçeklendirme
- Baseline: `DummyRegressor(strategy='mean')`
- Ana model: `LinearRegression()`

**4. Değerlendirme**
- MAE, RMSE, R² metrikleri
- Gerçek vs. Tahmin scatter plot
- Residual plot
- Veri setleri arası karşılaştırma grafikleri

---

##  Temel Bulgular

- **CPU Activity** en yüksek performansı sergilemiştir (R² = 0.88); özellikler ile CPU kullanımı arasında güçlü bir doğrusal ilişki mevcuttur.
- **Wine Quality** orta düzey performans göstermiştir (R² = 0.38); hedef değişkenin tam sayılardan oluşması lineer regresyonun uyumunu kısıtlamaktadır.
- **Ames Housing** en düşük performansı vermiştir (R² = 0.05); One-Hot Encoding sonrası 79 özelliğin yarattığı yüksek boyutluluk lineer regresyonu olumsuz etkilemiştir.

Üç veri setinde de Lineer Regresyon, baseline modeli anlamlı ölçüde geride bırakmıştır.

---

##  Gereksinimler

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

##  Kullanım

```bash
git clone https://github.com/<kullanici-adi>/lineer-regresyon-analizi.git
cd lineer-regresyon-analizi
jupyter notebook lineerRegresyonAnalizi.ipynb
```

---

## 📁 Depo Yapısı

```
├── lineerRegresyonAnalizi.ipynb   # Ana notebook
└── README.md
```

---

## 🛠 Kullanılan Teknolojiler

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green?logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
