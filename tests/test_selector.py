import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Proje kök dizinini sisteme ekle (consensusfs klasörünün bulunduğu yer)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from consensusfs import ConsensusSelector
    print(">>> ConsensusFS Kütüphanesi Başarıyla Yüklendi.\n")
except ImportError as e:
    print(f">>> Kütüphane Yükleme Hatası: {e}")
    sys.exit(1)

def run_comprehensive_test():
    print("="*60)
    print("CONSENSUS FS: KAPSAMLI SİSTEM TESTİ BAŞLIYOR")
    print("="*60)

    # 1. VERİ HAZIRLIĞI
    # 20 özellikli bir veri seti (5'i anlamlı, 15'i gürültü/gereksiz)
    X, y = make_classification(
        n_samples=200, 
        n_features=20, 
        n_informative=5, 
        n_redundant=2, 
        random_state=42
    )
    feature_names = [f"Özellik_{i:02d}" for i in range(20)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=42)

    # 2. MODEL TANIMLAMA
    model = RandomForestClassifier(n_estimators=50, random_state=42)

    # 3. KÜTÜPHANE AYARLARI (Tüm Metotlar ve Ağırlıklandırma Testi)
    # SHAP'a 2.0, Korelasyona 0.5 ağırlık veriyoruz.
    # LOFO dahil edildiği için n_jobs=-1 ile paralel çalıştırıyoruz.
    methods_to_test = ['correlation', 'permutation', 'shap', 'lofo']
    custom_weights = {
        'shap': 2.0, 
        'lofo': 1.5, 
        'permutation': 1.0, 
        'correlation': 0.5
    }

    selector = ConsensusSelector(
        estimator=model,
        methods=methods_to_test,
        aggregation='rank_mean',      # Sıralama tabanlı birleştirme
        n_features_to_select=5,       # En iyi 5 özelliği seç
        weights=custom_weights,       # Özel ağırlıklar
        n_jobs=-1,                    # Tüm CPU çekirdeklerini kullan
        scoring='roc_auc'             # Metrik tercihi
    )

    # 4. FIT İŞLEMİ (Eğitim ve Hesaplama)
    print(f"[*] Hesaplamalar başlatıldı: {methods_to_test}")
    print("[*] Paralel işlemci çekirdekleri kullanılıyor (n_jobs=-1)...")
    
    try:
        selector.fit(X_train, y_train)
        print("\n[+] Fit işlemi başarıyla tamamlandı.")
    except Exception as e:
        print(f"\n[!] Fit sırasında hata oluştu: {e}")
        return

    # 5. SONUÇLARIN ANALİZİ
    print("\n" + "-"*30)
    print("META-IMPORTANCE RAPORU (Top 10)")
    print("-"*30)
    print(selector.importance_df_.head(10))

    print(f"\n[+] Seçilen En İyi 5 Özellik: {selector.best_features_}")

    # 6. TRANSFORM TESTİ
    print("\n[*] Veri seti dönüştürülüyor (Transform)...")
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    print(f"[+] Orijinal Boyut: {X_train.shape}")
    print(f"[+] Yeni Boyut:      {X_train_selected.shape}")

    assert X_train_selected.shape[1] == 5, "Hata: Seçilen özellik sayısı yanlış!"
    assert list(X_train_selected.columns) == selector.best_features_, "Hata: Sütun isimleri eşleşmiyor!"

    # 7. FARKLI BİR AGGREGATION TESTİ (MinMax Mean)
    print("\n[*] Min-Max Toplulaştırma Testi Yapılıyor...")
    selector_minmax = ConsensusSelector(
        estimator=model,
        methods=['correlation', 'shap'],
        aggregation='minmax_mean'
    )
    selector_minmax.fit(X_train, y_train)
    print("[+] Min-Max Meta Skorları (İlk 3):")
    print(selector_minmax.importance_df_['meta_score'].head(3))

    # 8. GÖRSELLEŞTİRME TESTİ
    print("\n[*] Görselleştirme Heatmap hazırlanıyor...")
    # Not: Bu satır ekranında bir grafik penceresi açacaktır.
    try:
        selector.plot(top_n=10)
        print("[+] Görselleştirme başarılı.")
    except Exception as e:
        print(f"[!] Görselleştirme hatası (Display kaynaklı olabilir): {e}")

    print("\n" + "="*60)
    print("TÜM TESTLER BAŞARIYLA TAMAMLANDI!")
    print("="*60)

if __name__ == "__main__":
    run_comprehensive_test()