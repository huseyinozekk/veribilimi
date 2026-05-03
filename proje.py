
# ─────────────────────────────────────────────
# Veri Bilimi Projesi: Adult Income Dataset ile Sınıflandırma
# ─────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. RENK PALETİ & ESTETİK STİL
# ─────────────────────────────────────────────

# Her modele ayırt edici bir renk atıyoruz, grafikler daha okunabilir olsun diye
MODEL_COLORS = {
    "Rastgele Orman":   "#4F9CF9",
    "Karar Ağacı":       "#A78BFA",
    "K-En Yakın Komşu": "#34D399",
    "Lineer Regresyon":  "#FB923C",
    "Basit Bayes":       "#F472B6",
    "YSA (ANN)":        "#FFD700",
    "CNN (1D)":         "#FF4500"
}

# Koyu tema renkleri
BG       = "#0F1117"   # arka plan
SURFACE  = "#181C27"   # kart/panel yüzeyi
BORDER   = "#2A2F3F"   # kenarlıklar
TEXT     = "#E2E8F0"   # ana metin
MUTED    = "#8892A4"   # ikincil metin, eksenler

# Matplotlib'i koyu temaya göre ayarlıyoruz
plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

def title_style(ax, title, subtitle=None):
    """Estetik başlık ekleme yardımcısı"""
    ax.set_title(title, color=TEXT, fontsize=14, fontweight="bold", pad=14)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                ha="center", color=MUTED, fontsize=9)

# ─────────────────────────────────────────────
# 1. VERİ YÜKLEME & ÖN İŞLEME
# ─────────────────────────────────────────────
print("Veri yükleniyor ve ön işleme yapılıyor.")

# Sütun adlarını manuel veriyoruz çünkü dosyada başlık satırı yok
COLS = ["age","workclass","fnlwgt","education","education_num",
        "marital_status","occupation","relationship","race","sex",
        "capital_gain","capital_loss","hours_per_week","native_country","income"]

df = pd.read_csv("adult.data", header=None, names=COLS, na_values=" ?")
df.dropna(inplace=True)

# Gelir sütununu ikili (0/1) formata çeviriyoruz
df["income"] = df["income"].str.strip().map({"<=50K": 0, ">50K": 1})
df.dropna(subset=["income"], inplace=True)
df["income"] = df["income"].astype(int)

# Kategorik değişkenleri sayısal hale getiriyoruz
le = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# fnlwgt özelliği genellikle anlamsız katkı sağlar, çıkarıyoruz
FEATURE_NAMES = [col for col in COLS[:-1] if col != "fnlwgt"]
X = df[FEATURE_NAMES]
y = df["income"]

# Ölçeklendirme — KNN ve Lojistik Regresyon gibi mesafeye duyarlı modeller için gerekli
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_NAMES)

X_train,  X_test,  y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Xs_train, Xs_test, _,       _      = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Grafiklerde İngilizce özellik adları yerine Türkçe kullanmak için
TR_NAMES = {
    "age": "Yaş", "workclass": "İşveren Türü", "fnlwgt": "Temsil katsayısı",
    "education": "Eğitim", "education_num": "Eğitim Yılı",
    "marital_status": "Medeni Hal", "occupation": "Meslek",
    "relationship": "İlişki Durumu", "race": "Irk", "sex": "Cinsiyet",
    "capital_gain": "Sermaye Kazancı", "capital_loss": "Sermaye Kaybı",
    "hours_per_week": "Haftalık Çalışma Saati", "native_country": "Ülke",
}

# ─────────────────────────────────────────────
# 2. MODEL EĞİTİMİ (KLASİK + DERİN ÖĞRENME)
# ─────────────────────────────────────────────
print("Modeller eğitiliyor (Biraz zaman alabilir).")
results = {}

# Ölçeklendirilmiş/ölçeklendirilmemiş veri her modele göre ayrı veriliyor
KLASIK_MODELS = {
    "Rastgele Orman":   (RandomForestClassifier(n_estimators=100, random_state=42), X_train, X_test),
    "Karar Ağacı":       (DecisionTreeClassifier(random_state=42), X_train, X_test),
    "K-En Yakın Komşu": (KNeighborsClassifier(n_neighbors=5), Xs_train, Xs_test),
    "Lineer Regresyon":  (LogisticRegression(max_iter=2000, random_state=42), Xs_train, Xs_test),
    "Basit Bayes":       (GaussianNB(), X_train, X_test),
}

# Tüm klasik modelleri eğit, tahminleri ve metrikleri kaydet
for name, (model, xtr, xte) in KLASIK_MODELS.items():
    model.fit(xtr, y_train)
    y_pred, y_prob = model.predict(xte), model.predict_proba(xte)[:, 1]
    results[name] = {"Doğruluk": accuracy_score(y_test, y_pred)*100, "Kesinlik": precision_score(y_test, y_pred)*100, "Duyarlılık": recall_score(y_test, y_pred)*100, "F1 Skoru": f1_score(y_test, y_pred)*100, "AUC-ROC": roc_auc_score(y_test, y_prob)*100, "y_pred": y_pred, "y_prob": y_prob}

# Yapay Sinir Ağı — basit bir mimari, fazla karmaşıklaştırmaya gerek yok
ysa = Sequential([Dense(64, activation='relu', input_shape=(X_train.shape[1],)), Dropout(0.2), Dense(32, activation='relu'), Dense(1, activation='sigmoid')])
ysa.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ysa.fit(Xs_train, y_train, epochs=15, batch_size=32, verbose=0)
p_ysa = ysa.predict(Xs_test, verbose=0).flatten()
pr_ysa = (p_ysa > 0.5).astype(int)
results["YSA (ANN)"] = {"Doğruluk": accuracy_score(y_test, pr_ysa)*100, "Kesinlik": precision_score(y_test, pr_ysa)*100, "Duyarlılık": recall_score(y_test, pr_ysa)*100, "F1 Skoru": f1_score(y_test, pr_ysa)*100, "AUC-ROC": roc_auc_score(y_test, p_ysa)*100, "y_pred": pr_ysa, "y_prob": p_ysa}

# CNN için veriyi 3 boyutlu hale getirmek gerekiyor (örnekler, özellikler, kanal)
X_tr_c, X_te_c = np.expand_dims(Xs_train.values, axis=2), np.expand_dims(Xs_test.values, axis=2)
cnn = Sequential([Conv1D(32, 2, activation='relu', input_shape=(X_train.shape[1], 1)), MaxPooling1D(2), Flatten(), Dense(64, activation='relu'), Dense(1, activation='sigmoid')])
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(X_tr_c, y_train, epochs=15, batch_size=32, verbose=0)
p_cnn = cnn.predict(X_te_c, verbose=0).flatten()
pr_cnn = (p_cnn > 0.5).astype(int)
results["CNN (1D)"] = {"Doğruluk": accuracy_score(y_test, pr_cnn)*100, "Kesinlik": precision_score(y_test, pr_cnn)*100, "Duyarlılık": recall_score(y_test, pr_cnn)*100, "F1 Skoru": f1_score(y_test, pr_cnn)*100, "AUC-ROC": roc_auc_score(y_test, p_cnn)*100, "y_pred": pr_cnn, "y_prob": p_cnn}

# ─────────────────────────────────────────────
# 3. İNTERAKTİF GRAFİK YÖNETİM FONKSİYONLARI
# ─────────────────────────────────────────────

def plot_grafik1(fig):
    # Tüm modellerin beş metrik üzerinden yan yana karşılaştırması
    ax = fig.add_subplot(111)
    ax.set_facecolor(SURFACE)
    metrics = ["Doğruluk", "Kesinlik", "Duyarlılık", "F1 Skoru", "AUC-ROC"]
    n_models, n_metrics = len(results), len(metrics)
    x, width = np.arange(n_metrics), 0.11

    for i, (name, color) in enumerate(MODEL_COLORS.items()):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, color=color, alpha=0.88, label=name, edgecolor=BG, linewidth=0.5, zorder=3)
        # Çubukların üstüne değerleri yaz
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{bar.get_height():.1f}", 
                    ha="center", va="bottom", fontsize=6.5, color=color, fontweight="bold", rotation=90)

    ax.set_xticks(x + width * (n_models - 1) / 2); ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Değer (%)", fontsize=11); ax.set_ylim(0, 115); ax.yaxis.grid(True, zorder=0)
    title_style(ax, "Algoritma Karşılaştırması — Tüm Metrikler", "Klasik ML ve Derin Öğrenme | Test Kümesi %20")
    ax.legend(loc="upper right", framealpha=0.15, edgecolor=BORDER, labelcolor=TEXT, fontsize=8, ncol=2)

def plot_grafik2(fig):
    # Her model için ayrı bir karmaşıklık matrisi çiziyoruz
    axes = [fig.add_subplot(1, 7, i+1) for i in range(7)]
    for ax, (name, color) in zip(axes, MODEL_COLORS.items()):
        cm = confusion_matrix(y_test, results[name]["y_pred"])
        # Mutlak sayı yerine yüzdesel gösterim daha karşılaştırılabilir oluyor
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        cmap = LinearSegmentedColormap.from_list("custom", [SURFACE, color], N=256)
        
        ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100)
        for r in range(2):
            for c in range(2):
                bright = cm_pct[r, c] > 55
                ax.text(c, r, f"{cm[r,c]:,}\n({cm_pct[r,c]:.1f}%)", ha="center", va="center", 
                        fontsize=8, color=BG if bright else TEXT, fontweight="bold")

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["≤50K", ">50K"], fontsize=8)
        ax.set_yticklabels(["≤50K", ">50K"], fontsize=8, rotation=90, va="center")
        ax.set_xlabel("Tahmin", fontsize=8, color=MUTED); ax.set_ylabel("Gerçek", fontsize=8, color=MUTED)
        ax.set_title(name, color=color, fontsize=9, fontweight="bold", pad=8)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
        
    fig.suptitle("Karmaşıklık Matrisleri (Confusion Matrix)", fontsize=13, fontweight="bold", color=TEXT, y=0.88)

def plot_grafik3(fig):
    # ROC eğrisi — eşik bağımsız model performansını görmek için iyi bir yöntem
    ax = fig.add_subplot(111)
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], "--", color=BORDER, linewidth=1.5, label="Rastgele (AUC = 0.500)")
    for name, color in MODEL_COLORS.items():
        fpr, tpr, _ = roc_curve(y_test, results[name]["y_prob"])
        auc = results[name]["AUC-ROC"] / 100
        ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{name} (AUC = {auc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.03, color=color)

    ax.set_xlabel("Yanlış Pozitif Oranı (FPR)", fontsize=11); ax.set_ylabel("Doğru Pozitif Oranı (TPR)", fontsize=11)
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.05]); ax.yaxis.grid(True); ax.xaxis.grid(True)
    title_style(ax, "ROC Eğrileri — Tüm Algoritmalar", "Adult Income Dataset | İkili Sınıflandırma")
    ax.legend(loc="lower right", framealpha=0.2, edgecolor=BORDER, labelcolor=TEXT, fontsize=9.5)

def plot_grafik4(fig):
    # Rastgele Orman'ın dahili önem skorlarını kullanıyoruz
    ax = fig.add_subplot(111)
    ax.set_facecolor(SURFACE)
    rf_model = KLASIK_MODELS["Rastgele Orman"][0]
    importances, indices = rf_model.feature_importances_, np.argsort(rf_model.feature_importances_)[::-1]
    feat_names = [FEATURE_NAMES[i] for i in indices]
    feat_vals = importances[indices]
    feat_names_tr = [TR_NAMES.get(f, f) for f in feat_names]

    gradient_colors = plt.cm.cool(np.linspace(0.2, 0.9, len(feat_vals)))
    bars = ax.barh(feat_names_tr[::-1], feat_vals[::-1], color=gradient_colors[::-1], edgecolor=BG, height=0.72)

    # Her çubuğun yanına sayısal değeri yaz
    for bar, val in zip(bars, feat_vals[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", 
                va="center", fontsize=8.5, color=TEXT, fontweight="bold")

    ax.set_xlabel("Önem Skoru", fontsize=11); ax.xaxis.grid(True)
    title_style(ax, "Özellik Önem Skoru — Rastgele Orman", "Sınıflandırma kararına en çok katkı sağlayan özellikler")

def plot_grafik5(fig):
    # Korelasyon matrisinin sadece alt üçgeni gösteriliyor, tekrar bilgiye gerek yok
    ax = fig.add_subplot(111)
    df_corr = df[FEATURE_NAMES + ["income"]].rename(columns=TR_NAMES)
    df_corr.rename(columns={"income": "Gelir"}, inplace=True)
    corr = df_corr.corr()

    cmap_corr = LinearSegmentedColormap.from_list("corr", ["#F472B6", SURFACE, "#4F9CF9"], N=256)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", linewidths=0.4, cmap=cmap_corr, center=0, 
                vmin=-1, vmax=1, annot_kws={"size": 7.5, "color": TEXT, "weight":"bold"}, linecolor=BG, ax=ax)

    ax.set_title("Korelasyon Isı Haritası\nAdult Income Dataset", color=TEXT, fontsize=14, fontweight="bold", pad=16)
    ax.tick_params(axis="x", labelsize=8.5, rotation=45, colors=MUTED)
    ax.tick_params(axis="y", labelsize=8.5, rotation=0, colors=MUTED)

class Navigator:
    def __init__(self, fig):
        self.fig = fig
        self.idx = 0  # hangi grafikte olduğumuzu takip ediyoruz
        self.plots = [plot_grafik1, plot_grafik2, plot_grafik3, plot_grafik4, plot_grafik5]
        
    def update(self):
        clear_plot_area(self.fig)
        self.plots[self.idx](self.fig)
        self.fig.canvas.draw_idle()
        
    def next_plot(self, event):
        self.idx = (self.idx + 1) % len(self.plots)
        self.update()
        
    def prev_plot(self, event):
        self.idx = (self.idx - 1) % len(self.plots)
        self.update()

# ─────────────────────────────────────────────
# 4. GÖRSELLEŞTİRME BAŞLATICI PENCERESİ
# ─────────────────────────────────────────────

fig = plt.figure(figsize=(18, 7), facecolor=BG)
plt.subplots_adjust(bottom=0.15)

# Navigasyon butonlarının konumlarını [sol, alt, genişlik, yükseklik] olarak ayarlıyoruz
ax_prev = fig.add_axes([0.3, 0.02, 0.15, 0.06])
ax_next = fig.add_axes([0.55, 0.02, 0.15, 0.06])

btn_prev = Button(ax_prev, 'Önceki', color=SURFACE, hovercolor=BORDER)
btn_next = Button(ax_next, 'Sonraki', color=SURFACE, hovercolor=BORDER)

btn_prev.label.set_color(TEXT)
btn_next.label.set_color(TEXT)

navigator = Navigator(fig)

btn_prev.on_clicked(navigator.prev_plot)
btn_next.on_clicked(navigator.next_plot)

def clear_plot_area(fig):
    # Buton eksenlerine dokunmadan sadece grafik alanını temizle
    axes_to_remove = [ax for ax in fig.axes if ax not in [ax_prev, ax_next]]
    for ax in axes_to_remove:
        ax.remove()

navigator.update()

print("Grafik yönetim penceresi başlatılıyor...")
plt.show()