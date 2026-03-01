import matplotlib.pyplot as plt
import seaborn as sns

def plot_consensus_heatmap(importance_df, top_n=15, title="Consensus Feature Selection Heatmap"):
    """
    Farklı metriklerin özelliklere verdiği önem derecelerini Isı Haritası olarak çizer.
    """
    # İstenen sayıda en iyi özelliği seç
    plot_df = importance_df.head(top_n).copy()
    
    # Sadece algoritma skorlarını görselleştir (meta_score sütununu çıkar)
    if 'meta_score' in plot_df.columns:
        plot_df = plot_df.drop(columns=['meta_score'])
        
    # Her sütunu (algoritmayı) kendi içinde 0-1 arasına sıkıştır ki renkler düzgün görünsün
    plot_df_scaled = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min() + 1e-9)
    
    plt.figure(figsize=(10, max(6, top_n * 0.4))) # Özellik sayısına göre dinamik yükseklik
    ax = sns.heatmap(plot_df_scaled, annot=False, cmap='viridis', linewidths=.5)
    
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel("Özellikler (En İyiden En Kötüye)", fontsize=12)
    plt.xlabel("Metrikler", fontsize=12)
    
    # Eksen etiketlerini döndür
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()