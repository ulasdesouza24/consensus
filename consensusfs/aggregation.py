import pandas as pd
import numpy as np

def aggregate_scores(results_dict, feature_names, method='rank_mean', weights=None):
    """
    Farklı algoritmalardan gelen skorları birleştirir.
    Dönen DataFrame EN İYİ özellikten EN KÖTÜ özelliğe doğru sıralanmış olur.
    """
    # NaN / Inf değerlere karşı koruma (#9)
    df = pd.DataFrame(results_dict, index=feature_names)
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    
    if method == 'rank_mean':
        # Her algoritma için en yüksek skoru alan özelliği 1. sıraya koy (ascending=False)
        rank_df = df.rank(ascending=False, method='min')
        
        # Ağırlıklandırma — np.average ile uygun ölçek korunarak uygulanır (#8)
        if weights:
            weight_values = [weights.get(col, 1.0) for col in rank_df.columns]
            df['meta_score'] = np.average(rank_df.values, axis=1, weights=weight_values)
        else:
            df['meta_score'] = rank_df.mean(axis=1)
        
        # En düşük puana (en iyi sıraya) göre küçükten büyüğe sırala
        return df.sort_values('meta_score', ascending=True)
        
    elif method == 'minmax_mean':
        # (x - min) / (max - min) formülü ile tüm değerleri 0-1 arasına çek
        minmax_df = (df - df.min()) / (df.max() - df.min() + 1e-9)
        
        # Ağırlıklandırma — np.average ile uygun ölçek korunarak uygulanır (#8)
        if weights:
            weight_values = [weights.get(col, 1.0) for col in minmax_df.columns]
            df['meta_score'] = np.average(minmax_df.values, axis=1, weights=weight_values)
        else:
            df['meta_score'] = minmax_df.mean(axis=1)
            
        # En yüksek puana göre büyükten küçüğe sırala
        return df.sort_values('meta_score', ascending=False)
        
    else:
        raise ValueError("Desteklenmeyen birleştirme yöntemi. Lütfen 'rank_mean' veya 'minmax_mean' kullanın.")