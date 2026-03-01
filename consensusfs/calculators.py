import numpy as np
import pandas as pd
import shap
import uuid
import warnings
from sklearn.inspection import permutation_importance
from lofo import Dataset, LOFOImportance

def calc_correlation(X, y, **kwargs):
    """Hedef değişken ile özellikler arasındaki mutlak Pearson korelasyonunu hesaplar."""
    corr = X.apply(lambda col: col.corr(y)).abs()
    corr = corr.fillna(0)
    return corr.values

def calc_permutation(estimator, X, y, **kwargs):
    """Scikit-Learn tabanlı Permutation Importance hesaplar."""
    # joblib çakışmalarını önlemek için n_jobs=1 kullanıyoruz
    result = permutation_importance(estimator, X, y, n_repeats=5, random_state=42, n_jobs=1)
    return result.importances_mean

def calc_shap(estimator, X, y, **kwargs):
    """SHAP değerlerini hesaplar. Modeline göre otomatik Explainer seçer."""
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Sadece bu blok için SHAP uyarılarını gizle
        
        try:
            # Ağaç tabanlı modeller için (XGBoost, RandomForest, LightGBM vb.)
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X)
        except Exception:
            try:
                # Doğrusal modeller için
                explainer = shap.LinearExplainer(estimator, X)
                shap_values = explainer.shap_values(X)
            except Exception:
                # Diğer tüm modeller için (KernelExplainer yavaş olduğu için sample alıyoruz)
                sample_X = shap.sample(X, 100) if len(X) > 100 else X
                explainer = shap.KernelExplainer(estimator.predict, sample_X)
                shap_values = explainer.shap_values(X)
            
    # Eğer çok sınıflı sınıflandırma (multi-class) ise shap_values bir liste döner
    if isinstance(shap_values, list):
        # Tüm sınıfların ortalama mutlak etkisini al
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # Tekil çıktı (regresyon veya binary)
        if len(shap_values.shape) == 3: # Yeni SHAP versiyonları için
             mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
             mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
    return mean_abs_shap

def calc_lofo(estimator, X, y, feature_names, scoring="roc_auc", **kwargs):
    """LOFO (Leave One Feature Out) Importance hesaplar."""
    df = X.copy()
    # UUID ile benzersiz geçici sütun adı oluştur (X'teki sütunlarla çakışmayı önler)
    target_name = f'__lofo_target_{uuid.uuid4().hex}__'
    df[target_name] = y.values
    
    dataset = Dataset(df=df, target=target_name, features=feature_names)
    
    # LOFO hesaplama (n_jobs=1 nested paralel çakışmasını önler)
    lofo_imp = LOFOImportance(dataset, model=estimator, scoring=scoring, cv=3, n_jobs=1)
    importance_df = lofo_imp.get_importance()
    
    # Skorları orijinal özellik sırasına göre eşleştir ve array olarak dön
    importance_df = importance_df.set_index('feature')
    scores = importance_df.loc[feature_names, 'importance_mean'].values
    return scores