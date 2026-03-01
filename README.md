# ConsensusFS

> **Tek bir feature selection metoduna güvenme — hepsini çalıştır, en iyisini seç.**

ConsensusFS, makine öğrenmesi projelerinde özellik seçimi (feature selection) için geliştirilmiş bir **ensemble / konsensüs kütüphanesidir**. SHAP, LOFO, Permutation Importance ve Korelasyon gibi farklı metodları **aynı anda paralel** olarak çalıştırır; ardından sonuçları akıllıca birleştirerek güvenilir bir **Meta-Importance skoru** üretir.

Scikit-Learn uyumludur: `fit`, `transform`, `fit_transform` ve Pipeline desteği ile gelir.

---

## 📋 İçindekiler

- [Neden ConsensusFS?](#neden-consensusfs)
- [Kurulum](#kurulum)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Nasıl Çalışır?](#nasıl-çalışır)
- [API Referansı](#api-referansı)
- [Desteklenen Metodlar](#desteklenen-metodlar)
- [Aggregation Stratejileri](#aggregation-stratejileri)
- [Gelişmiş Kullanım](#gelişmiş-kullanım)
- [Sklearn Pipeline ile Kullanım](#sklearn-pipeline-ile-kullanım)
- [Görselleştirme](#görselleştirme)
- [Bağımlılıklar](#bağımlılıklar)
- [Sık Sorulan Sorular](#sık-sorulan-sorular)

---

## Neden ConsensusFS?

Her feature selection metodu farklı bir şeyi ölçer ve farklı zayıf noktaları vardır:

| Metod | Gördüğü Şey | Kör Noktası |
|-------|------------|------------|
| Correlation | Lineer ilişkiler | Non-lineer ilişkileri kaçırır |
| Permutation | Model performansına etkisi | Overfitting modelde yanıltıcı olabilir |
| SHAP | Her özelliğin tahmine katkısı | Model bağımlıdır |
| LOFO | CV skoru üzerindeki etkisi | Yavaştır, küçük veride gürültülüdür |

**ConsensusFS** bu metodları bir araya getirerek:
- Hiçbir metodun körü körüne güven açığına düşmez
- Birden fazla metodun hemfikir olduğu özellikleri seçer
- Bireysel metodlara göre daha **kararlı (stable)** ve **güvenilir** sonuçlar üretir

---

## Kurulum

```bash
# Repoyu klonlayın ve dizine girin
git clone https://github.com/kullanici_adi/consensusfs.git
cd consensusfs

# Yerel olarak kurun
pip install .
```

**Tüm bağımlılıkları ayrıca kurmak isterseniz:**

```bash
pip install scikit-learn>=1.0.0 pandas>=1.0.0 numpy>=1.18.0 shap>=0.40.0 lofo-importance>=0.3.0 joblib>=1.0.0 seaborn>=0.11.0 matplotlib>=3.3.0
```

> **Python Gereksinimi:** >= 3.8

---

## Hızlı Başlangıç

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from consensusfs import ConsensusSelector

# Veri hazırla
X, y = make_classification(n_samples=500, n_features=20, n_informative=5, random_state=42)
X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(20)])

# Model ve selector tanımla
model = RandomForestClassifier(n_estimators=100, random_state=42)

selector = ConsensusSelector(
    estimator=model,
    methods=['correlation', 'permutation', 'shap'],  # kullanılacak metodlar
    n_features_to_select=10,                          # kaç özellik seçilsin
    n_jobs=-1                                         # tüm CPU çekirdeklerini kullan
)

# Eğit ve dönüştür
X_selected = selector.fit_transform(X_df, y)

print("Seçilen özellikler:", selector.best_features_)
print("Yeni boyut:", X_selected.shape)

# Sonuçları görselleştir
selector.plot(top_n=10)
```

---

## Nasıl Çalışır?

```
Veri (X, y)
    │
    ▼
┌─────────────────────────────────────────────────┐
│           Paralel Hesaplama (joblib)             │
│                                                  │
│  Correlation │ Permutation │  SHAP  │   LOFO    │
│  (opsiyonel) │ (opsiyonel) │(opsy.) │ (opsy.)  │
└──────┬───────┴──────┬──────┴───┬────┴─────┬─────┘
       │              │          │           │
       └──────────────┴────┬─────┘───────────┘
                           ▼
              Aggregation (rank_mean / minmax_mean)
                      + Ağırlıklandırma
                           │
                           ▼
                  Meta-Skor Tablosu (importance_df_)
                           │
                           ▼
                  En İyi N Özellik (best_features_)
```

### Aggregation Detayı (`rank_mean`)

1. Her metod kendi skorunu üretir (örn. SHAP değerleri, korelasyon katsayıları)
2. Her metod için özellikler **sıralanır** (1 = en önemli, N = en önemsiz)
3. Her özelliğin tüm sıraların **ağırlıklı ortalaması** alınır → `meta_score`
4. En düşük `meta_score` → en iyi özellik

---

## API Referansı

### `ConsensusSelector(estimator, methods, aggregation, n_features_to_select, weights, n_jobs, scoring)`

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `estimator` | sklearn estimator | **Zorunlu** | Kullanılacak ML modeli. SHAP ve Permutation için eğitilir. |
| `methods` | `list[str]` | `['correlation', 'permutation', 'shap']` | Kullanılacak özellik seçim metodlarının listesi. |
| `aggregation` | `str` | `'rank_mean'` | Skor birleştirme stratejisi. `'rank_mean'` veya `'minmax_mean'`. |
| `n_features_to_select` | `int` veya `None` | `None` | Seçilecek özellik sayısı. `None` ise tüm özellikler sıralanır. |
| `weights` | `dict` veya `None` | `None` | Her metoda verilecek ağırlık. `{'shap': 2.0, 'correlation': 0.5}` gibi. |
| `n_jobs` | `int` | `-1` | Joblib paralel iş sayısı. `-1` tüm CPU çekirdeklerini kullanır. |
| `scoring` | `str` | `'roc_auc'` | LOFO metodu için skorlama metriği. |

---

### Metodlar

#### `fit(X, y) → self`
Tüm feature selection hesaplamalarını yapar, `importance_df_` ve `best_features_` atributlarını doldurur.

- `X`: `pd.DataFrame` veya `np.ndarray` — girdi özellikleri
- `y`: `pd.Series` veya `np.ndarray` — hedef değişken

#### `transform(X) → pd.DataFrame veya np.ndarray`
`fit()` ile seçilmiş özellikleri içeren veri setini döndürür.

- `X`'in tipi korunur: DataFrame girerse DataFrame döner, ndarray girerse ndarray döner.

#### `fit_transform(X, y) → pd.DataFrame veya np.ndarray`
`fit(X, y)` ardından `transform(X)` çalıştırır.

#### `plot(top_n=15, title="Consensus Feature Selection Heatmap")`
En önemli `top_n` özelliği için her metodun verdiği önemi gösteren Isı Haritası çizer.

---

### Atributlar (fit() sonrası)

| Atribut | Tip | Açıklama |
|---------|-----|----------|
| `importance_df_` | `pd.DataFrame` | Her metodun ve meta skorun sütun olduğu, özellik sıralı tablo |
| `best_features_` | `list[str]` | Seçilen en iyi özellik isimleri (sıralanmış) |
| `feature_names_` | `list[str]` | Eğitim sırasındaki tüm özellik isimleri |

---

## Desteklenen Metodlar

### `'correlation'`
Hedef değişken ile her özellik arasındaki **mutlak Pearson korelasyonu** hesaplar. Model gerektirmez, çok hızlıdır. Lineer olmayan ilişkileri kaçırabilir.

### `'permutation'`
Scikit-learn'in `permutation_importance` fonksiyonunu kullanır. Her özelliğin değerleri karıştırıldığında model skoru ne kadar düşüyor? Düşüş fazlaysa özellik önemlidir.

### `'shap'`
SHAP (SHapley Additive exPlanations) değerlerini hesaplar. Model tipine göre otomatik olarak doğru Explainer seçilir:
1. Ağaç tabanlı modeller (XGBoost, LightGBM, RandomForest) → `TreeExplainer`
2. Doğrusal modeller → `LinearExplainer`
3. Diğer tüm modeller → `KernelExplainer` (100 örnek ile örnekleme yapılır)

### `'lofo'`
LOFO (Leave One Feature Out) Importance: Her özellik sırası ile veri setinden çıkarıldığında cross-validation skoru ne kadar değişiyor? En yorumlayıcı ama en yavaş metoddur.

> ⚠️ LOFO büyük veri setlerinde uzun sürebilir. `scoring` parametresi ile metriği değiştirebilirsiniz.

---

## Aggregation Stratejileri

### `rank_mean` (Önerilen)
Her metodun sıra bilgisi kullanılır. Ölçek farklılıklarından etkilenmez.

```
meta_score = weighted_average(rank_per_method)
```

- **Düşük** `meta_score` → daha iyi özellik
- Ağırlık verilirse `numpy.average()` ile uygulanır (sıralama bozulmaz)

### `minmax_mean`
Ham skorlar 0-1 arasına normalize edilip ortalaması alınır.

```
normalized = (score - min) / (max - min)
meta_score = weighted_average(normalized)
```

- **Yüksek** `meta_score` → daha iyi özellik
- Metodların orijinal skor büyüklükleri önemlidir, aşırı baskın metodlar riski vardır.

---

## Gelişmiş Kullanım

### Özel Ağırlıklar

```python
# SHAP'a daha fazla güven, korelasyona daha az
custom_weights = {
    'shap': 2.0,
    'lofo': 1.5,
    'permutation': 1.0,
    'correlation': 0.5
}

selector = ConsensusSelector(
    estimator=model,
    methods=['correlation', 'permutation', 'shap', 'lofo'],
    weights=custom_weights,
    n_features_to_select=10
)
```

### Sadece Hızlı Metodlar (LOFO Olmadan)

```python
# Hızlı çalışma için sadece correlation + shap
selector = ConsensusSelector(
    estimator=model,
    methods=['correlation', 'shap'],
    aggregation='rank_mean',
    n_features_to_select=15,
    n_jobs=-1
)
```

### Tüm Özellikleri Sıralı Almak

```python
# n_features_to_select=None ile tüm özellikler önem sırasıyla listelenir
selector = ConsensusSelector(estimator=model)
selector.fit(X, y)

print(selector.importance_df_)         # Tam skor tablosu
print(selector.best_features_)         # Tüm özellikler, en iyiden en kötüye
```

### Regresyon Problemleri için

```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(random_state=42)

selector = ConsensusSelector(
    estimator=model,
    methods=['correlation', 'permutation', 'shap'],
    scoring='r2',          # Regresyon için uygun metrik
    n_features_to_select=8
)
selector.fit(X_train, y_train)
```

---

## Sklearn Pipeline ile Kullanım

`ConsensusSelector`, Scikit-Learn Pipeline ile tam uyumludur:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from consensusfs import ConsensusSelector

# Not: Pipeline içinde estimator olarak basit bir model kullanın;
# Pipeline'ın son adımında farklı bir model kullanabilirsiniz.
inner_model = RandomForestClassifier(n_estimators=50, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', ConsensusSelector(
        estimator=inner_model,
        methods=['correlation', 'shap'],
        n_features_to_select=10
    )),
    ('classifier', LogisticRegression())
])

pipe.fit(X_train, y_train)
print("Test Skoru:", pipe.score(X_test, y_test))
```

---

## Görselleştirme

```python
# fit() çağrısından sonra kullanın
selector.plot(top_n=15)

# Özel başlıkla
selector.plot(top_n=10, title="Proje X — Özellik Önem Haritası")
```

Isı haritası, her metodun her özelliğe verdiği önemi **0-1 arasında normalize ederek** gösterir. Koyu renk = daha önemli.

---

## Bağımlılıklar

| Kütüphane | Minimum Sürüm | Kullanım Amacı |
|-----------|--------------|----------------|
| `scikit-learn` | ≥ 1.0.0 | BaseEstimator, permutation_importance |
| `pandas` | ≥ 1.0.0 | DataFrame işlemleri |
| `numpy` | ≥ 1.18.0 | Sayısal hesaplamalar |
| `shap` | ≥ 0.40.0 | SHAP değerleri |
| `lofo-importance` | ≥ 0.3.0 | LOFO hesaplaması |
| `joblib` | ≥ 1.0.0 | Paralel hesaplama |
| `matplotlib` | ≥ 3.3.0 | Görselleştirme |
| `seaborn` | ≥ 0.11.0 | Isı haritası |

---

## Sık Sorulan Sorular

**LOFO çok yavaş, ne yapmalıyım?**
`methods` listesinden `'lofo'`'yu çıkarın. Diğer üç metod yeterince güçlüdür.

**Model olarak ne kullanmalıyım?**
SHAP için ağaç tabanlı modeller (RandomForest, XGBoost, LightGBM) hem en hızlı hem en doğru sonucu verir. Ancak herhangi bir Scikit-Learn uyumlu model çalışır.

**`n_features_to_select` nasıl seçmeliyim?**
Önce `None` bırakıp `importance_df_`'e bakın; `meta_score`'un belirgin şekilde arttığı nokta iyi bir kesme noktasıdır.

**NumPy array kullanabilir miyim, DataFrame zorunlu mu?**
Her ikisi de çalışır. NumPy array verildiğinde özellik isimleri otomatik olarak `feature_0`, `feature_1`, ... şeklinde atanır.

**`weights` vermediğimde ne olur?**
Tüm metodlar eşit ağırlıkla (1.0) değerlendirilir; `rank_mean` için basit ortalama sıra hesaplanır.

**Pipeline'da `fit` edilmiş modeli tekrar eğitiyor mu?**
Evet. `ConsensusSelector.fit()` her çağrıldığında estimatoru **yeniden eğitir**. Pipeline içinde bu beklenen davranıştır.

---

## Lisans

MIT License — Dilediğiniz gibi kullanın, değiştirin ve dağıtın.

---

*Geliştirici: **Ulaş Taylan Met** — umet9711@gmail.com*