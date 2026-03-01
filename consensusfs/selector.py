import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed

# Kendi modüllerimizi içe aktarıyoruz
from consensusfs.calculators import calc_correlation, calc_permutation, calc_shap, calc_lofo
from consensusfs.aggregation import aggregate_scores
from consensusfs.plotting import plot_consensus_heatmap

class ConsensusSelector(BaseEstimator, TransformerMixin):
    """
    Consensus Feature Selection Kütüphanesi Ana Sınıfı.
    Scikit-Learn uyumlu (fit, transform) çalışır.
    """
    def __init__(self, estimator, methods=None, aggregation='rank_mean', 
                 n_features_to_select=None, weights=None, n_jobs=-1, scoring="roc_auc"):
        self.estimator = estimator
        # Varsayılan metodlar (LOFO yavaş olduğu için varsayılanlara eklemedik, istenirse yazılır)
        self.methods = methods if methods is not None else['correlation', 'permutation', 'shap']
        self.aggregation = aggregation
        self.n_features_to_select = n_features_to_select
        self.weights = weights
        self.n_jobs = n_jobs
        self.scoring = scoring
        
    def fit(self, X, y):
        """Özellik önemlerini hesaplar ve meta skor üretir."""
        # 1. Veri Doğrulama ve DataFrame formatına dönüştürme
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.Series(y, name='target')
            
        self.feature_names_ = X.columns.tolist()
        
        # 2. Modeli eğit (SHAP ve Permutation için eğitilmiş model şarttır)
        self.estimator.fit(X, y)
        
        # 3. Hesaplanacak görevleri belirle ve Paralel Çalıştır (standart generator pattern)
        method_names = []
        delayed_tasks = []
        for method in self.methods:
            if method == 'correlation':
                method_names.append(method)
                delayed_tasks.append(delayed(calc_correlation)(X, y))
            elif method == 'permutation':
                method_names.append(method)
                delayed_tasks.append(delayed(calc_permutation)(self.estimator, X, y))
            elif method == 'shap':
                method_names.append(method)
                delayed_tasks.append(delayed(calc_shap)(self.estimator, X, y))
            elif method == 'lofo':
                method_names.append(method)
                delayed_tasks.append(delayed(calc_lofo)(self.estimator, X, y, self.feature_names_, scoring=self.scoring))
            else:
                raise ValueError(f"Geçersiz metod: {method}. Desteklenenler: correlation, permutation, shap, lofo")

        # 4. Paralel Hesaplama (Joblib) — generator pattern ile doğru kullanım
        results_list = Parallel(n_jobs=self.n_jobs)(t for t in delayed_tasks)

        # Sonuçları sözlüğe çevir
        results_dict = dict(zip(method_names, results_list))
        
        # 5. Skorları Toplulaştırma (Aggregation)
        self.importance_df_ = aggregate_scores(
            results_dict=results_dict,
            feature_names=self.feature_names_,
            method=self.aggregation,
            weights=self.weights
        )
        
        # 6. Seçilecek en iyi özelliklerin listesini kaydet
        n_select = self.n_features_to_select if self.n_features_to_select else len(self.feature_names_)
        self.best_features_ = self.importance_df_.head(n_select).index.tolist()
        
        # Estimator'ın fit edildiğini işaretlemek için sklearn standartı
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Sadece en iyi özellikleri içeren veri setini filtreler ve döndürür."""
        check_is_fitted(self, 'is_fitted_')
        
        if isinstance(X, pd.DataFrame):
            # Eğer DataFrame verildiyse sütun isimlerinden filtrele
            missing_cols = set(self.best_features_) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Transform işlemindeki veride eksik sütunlar var: {missing_cols}")
            return X[self.best_features_]
        else:
            # NumPy array ise, eğitimdeki sütun indekslerini bul ve filtrele
            indices =[self.feature_names_.index(f) for f in self.best_features_]
            return X[:, indices]
            
    def fit_transform(self, X, y=None, **fit_params):
        """Önce fit() sonra transform() çalıştırır."""
        return self.fit(X, y).transform(X)
        
    def plot(self, top_n=15, title="Consensus Feature Selection Heatmap"):
        """Sonuçları ısı haritası olarak gösterir."""
        check_is_fitted(self, 'is_fitted_')
        actual_top_n = min(top_n, len(self.feature_names_))
        plot_consensus_heatmap(self.importance_df_, top_n=actual_top_n, title=title)