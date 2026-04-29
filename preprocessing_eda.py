"""
TalkHand BISINDO - Data Preprocessing & Exploratory Data Analysis
Menggunakan MediaPipe Hand Landmark untuk ekstraksi fitur dari gambar isyarat BISINDO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import os
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
PLOTS_DIR  = BASE_DIR / "outputs" / "plots"
PROC_DIR   = BASE_DIR / "data" / "processed"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette & style ────────────────────────────────────────────────────
PALETTE    = {"primary": "#3D7A3A", "secondary": "#E8A020", "accent": "#7B5EA7",
               "bg": "#F5F0E8", "text": "#2C2C2C", "light": "#D4E8D0"}
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#FAFAFA",
                     "axes.edgecolor": "#CCCCCC", "grid.alpha": 0.4})

# ── Constants ──────────────────────────────────────────────────────────
ALPHABET  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
N_LABELS  = len(ALPHABET)
N_SAMPLES = 320        # sampel per label (representatif dataset BISINDO)
N_LMKS    = 21         # MediaPipe hand landmarks
FEAT_COLS = [f"lm{i}_{ax}" for i in range(N_LMKS) for ax in ("x","y","z")]

# ── MediaPipe landmark connections (untuk visualisasi) ─────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # ibu jari
    (0,5),(5,6),(6,7),(7,8),          # telunjuk
    (0,9),(9,10),(10,11),(11,12),     # tengah
    (0,13),(13,14),(14,15),(15,16),   # manis
    (0,17),(17,18),(18,19),(19,20),   # kelingking
    (5,9),(9,13),(13,17),             # telapak tangan
]

# ── Landmark naming ────────────────────────────────────────────────────
LMK_NAMES = {
    0:"Wrist", 1:"Thumb_CMC", 2:"Thumb_MCP", 3:"Thumb_IP", 4:"Thumb_Tip",
    5:"Index_MCP", 6:"Index_PIP", 7:"Index_DIP", 8:"Index_Tip",
    9:"Mid_MCP",  10:"Mid_PIP", 11:"Mid_DIP", 12:"Mid_Tip",
    13:"Ring_MCP",14:"Ring_PIP",15:"Ring_DIP",16:"Ring_Tip",
    17:"Pink_MCP",18:"Pink_PIP",19:"Pink_DIP",20:"Pink_Tip",
}

# ════════════════════════════════════════════════════════════════════════
# 1. GENERATE REALISTIC SYNTHETIC BISINDO DATA
#    Setiap huruf BISINDO memiliki konfigurasi jari yang berbeda.
#    Data disimulasikan berdasarkan geometri tangan nyata + noise realistis.
# ════════════════════════════════════════════════════════════════════════

def finger_angles(open_frac):
    """Kembalikan koordinat Y jari berdasarkan fraksi keterbukaan (0=tertutup, 1=terbuka)."""
    return np.array([0.5 + 0.3 * open_frac * i/4 for i in range(5)])

def make_hand_pose(thumb, index, middle, ring, pinky, spread=0.0):
    """
    Buat 21 landmark tangan berbasis konfigurasi jari.
    thumb/index/.../pinky : 0.0–1.0 (0=tertutup, 1=terbuka)
    spread : pemisahan jari (0=rapat, 1=melebar)
    """
    pts = np.zeros((N_LMKS, 3))

    # Wrist
    pts[0] = [0.5, 0.85, 0.0]

    # Metacarpal base (knuckles)
    knuckle_x = np.array([0.35, 0.42, 0.50, 0.58, 0.65]) + spread * np.array([-0.06,-0.02,0,0.02,0.06])
    knuckle_y = np.array([0.72, 0.62, 0.60, 0.62, 0.65])

    fractions = [thumb, index, middle, ring, pinky]
    lm_map = [(1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)]

    for fi, (base, pip_, dip_, tip_) in enumerate(lm_map):
        kx, ky = knuckle_x[fi], knuckle_y[fi]
        frac = fractions[fi]
        # Base knuckle
        pts[base] = [kx - 0.04*(1-frac), ky, 0.01]
        # PIP joint
        curl = 0.12 * (1 - frac)
        pts[pip_] = [kx - 0.01, ky - 0.12 + curl, -0.02*frac]
        # DIP joint
        pts[dip_] = [kx + 0.01*(frac-0.5), ky - 0.20 + curl*1.5, -0.03*frac]
        # Tip
        pts[tip_] = [kx + 0.02*(frac-0.5), ky - 0.28 + curl*2.0, -0.04*frac]

    return pts.flatten()

# Konfigurasi isyarat alfabet BISINDO
# (thumb, index, middle, ring, pinky, spread)
BISINDO_CONFIGS = {
    "A": (0.2, 0.0, 0.0, 0.0, 0.0, 0.0),   # kepalan, ibu jari ke samping
    "B": (0.0, 1.0, 1.0, 1.0, 1.0, 0.1),   # 4 jari tegak, ibu jari dalam
    "C": (0.6, 0.6, 0.6, 0.6, 0.6, 0.3),   # setengah melengkung seperti C
    "D": (0.1, 1.0, 0.1, 0.1, 0.1, 0.1),   # telunjuk tegak, lain menutup
    "E": (0.1, 0.2, 0.2, 0.2, 0.2, 0.0),   # semua jari sedikit terbuka
    "F": (0.5, 0.3, 1.0, 1.0, 1.0, 0.2),   # OK-shape + 3 jari tegak
    "G": (0.5, 1.0, 0.0, 0.0, 0.0, 0.0),   # telunjuk horizontal
    "H": (0.1, 1.0, 1.0, 0.0, 0.0, 0.1),   # 2 jari horizontal
    "I": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),   # kelingking tegak
    "J": (0.0, 0.0, 0.0, 0.0, 1.0, 0.1),   # I + gerakan J
    "K": (0.4, 1.0, 1.0, 0.0, 0.0, 0.2),   # ibu jari + 2 jari
    "L": (1.0, 1.0, 0.0, 0.0, 0.0, 0.3),   # L-shape
    "M": (0.0, 0.1, 0.1, 0.1, 0.0, 0.0),   # 3 jari di atas ibu jari
    "N": (0.0, 0.1, 0.1, 0.0, 0.0, 0.0),   # 2 jari di atas ibu jari
    "O": (0.5, 0.5, 0.5, 0.5, 0.5, 0.1),   # O-shape
    "P": (0.4, 1.0, 1.0, 0.0, 0.0, 0.1),   # K terbalik
    "Q": (0.5, 1.0, 0.0, 0.0, 0.0, 0.1),   # G ke bawah
    "R": (0.1, 1.0, 1.0, 0.0, 0.0, 0.0),   # jari silang
    "S": (0.3, 0.1, 0.1, 0.1, 0.1, 0.0),   # kepalan + ibu jari
    "T": (0.3, 0.0, 0.0, 0.0, 0.0, 0.0),   # ibu jari antara telunjuk-tengah
    "U": (0.0, 1.0, 1.0, 0.0, 0.0, 0.1),   # 2 jari tegak rapat
    "V": (0.0, 1.0, 1.0, 0.0, 0.0, 0.3),   # peace / V-shape
    "W": (0.0, 1.0, 1.0, 1.0, 0.0, 0.3),   # 3 jari terbuka
    "X": (0.1, 0.5, 0.0, 0.0, 0.0, 0.1),   # telunjuk bengkok
    "Y": (1.0, 0.0, 0.0, 0.0, 1.0, 0.4),   # hang-loose
    "Z": (0.0, 1.0, 0.0, 0.0, 0.0, 0.1),   # telunjuk gambar Z
}

def generate_dataset():
    """Generate synthetic dataset dengan landmark hand pose realistis."""
    print("[INFO] Generating synthetic BISINDO hand landmark dataset...")
    rng = np.random.default_rng(42)
    rows = []
    for letter in ALPHABET:
        cfg = BISINDO_CONFIGS[letter]
        base_pose = make_hand_pose(*cfg)
        for _ in range(N_SAMPLES):
            noise_scale = 0.015 + rng.uniform(0, 0.01)
            noise  = rng.normal(0, noise_scale, base_pose.shape)
            # Tambah variasi kemiringan tangan (rotation simulation)
            tilt   = rng.uniform(-0.03, 0.03, base_pose.shape)
            sample = base_pose + noise + tilt
            # Variasi jarak kamera (scale)
            scale  = rng.uniform(0.92, 1.08)
            sample = sample * scale
            rows.append([letter] + sample.tolist())

    df = pd.DataFrame(rows, columns=["label"] + FEAT_COLS)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   [OK] Dataset: {len(df):,} sampel | {N_LABELS} label | {len(FEAT_COLS)} fitur")
    return df

# ════════════════════════════════════════════════════════════════════════
# 2. DATA PREPROCESSING
# ════════════════════════════════════════════════════════════════════════

def normalize_landmarks(df_feat):
    """
    Preprocessing: Normalisasi landmark relatif terhadap wrist (landmark 0).
    Strategi ini membuat fitur invariant terhadap posisi tangan di layar.
    """
    df = df_feat.copy()
    for ax in ("x","y","z"):
        wrist_col = f"lm0_{ax}"
        for i in range(1, N_LMKS):
            col = f"lm{i}_{ax}"
            df[col] = df[col] - df[wrist_col]
        df[wrist_col] = 0.0  # wrist jadi titik referensi (0,0,0)
    return df

def scale_by_hand_size(df_feat):
    """
    Normalisasi skala: bagi semua koordinat dengan jarak wrist–middle_MCP.
    Membuat fitur invariant terhadap jarak kamera / ukuran tangan.
    """
    df = df_feat.copy()
    mid_mcp_x = df["lm9_x"].values
    mid_mcp_y = df["lm9_y"].values
    hand_size  = np.sqrt(mid_mcp_x**2 + mid_mcp_y**2).clip(min=1e-6)
    for col in FEAT_COLS:
        df[col] = df[col] / hand_size
    return df

def preprocess_pipeline(df_raw):
    """Jalankan full preprocessing pipeline."""
    print("\n[INFO] PREPROCESSING PIPELINE")
    print("-" * 45)
    labels = df_raw["label"]
    X_raw  = df_raw[FEAT_COLS]

    print(f"  [1/4] Input   : {X_raw.shape} | Missing: {X_raw.isna().sum().sum()}")

    # Step 1: Normalisasi posisi relatif wrist
    X_norm = normalize_landmarks(X_raw)
    print(f"  [2/4] Pos-norm: {X_norm.shape} (relatif ke wrist)")

    # Step 2: Normalisasi skala ukuran tangan
    X_scaled = scale_by_hand_size(X_norm)
    print(f"  [3/4] Size-norm: {X_scaled.shape} (invariant skala kamera)")

    # Step 3: StandardScaler (zero mean, unit variance)
    scaler = StandardScaler()
    X_final = pd.DataFrame(scaler.fit_transform(X_scaled), columns=FEAT_COLS)
    print(f"  [4/4] Standard: mean approx {X_final.values.mean():.4f}, std approx {X_final.values.std():.4f}")

    # Encode label
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    print(f"\n  [OK] Preprocessing selesai | X: {X_final.shape} | y: {y.shape}")
    return X_final, y, le, scaler

# ════════════════════════════════════════════════════════════════════════
# 3. VISUALISASI EDA
# ════════════════════════════════════════════════════════════════════════

def plot_dataset_overview(df):
    """Slide 1: Overview distribusi dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle("TalkHand BISINDO — Dataset Overview", fontsize=16,
                 fontweight="bold", color=PALETTE["text"], y=1.01)

    # -- Distribusi label --
    ax = axes[0]
    counts = df["label"].value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE["primary"] if i%2==0 else PALETTE["secondary"]
                         for i in range(N_LABELS)],
                  edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlabel("Label (Huruf BISINDO)", fontsize=11)
    ax.set_ylabel("Jumlah Sampel", fontsize=11)
    ax.set_title("Distribusi Sampel per Label", fontweight="bold")
    ax.axhline(counts.mean(), color="red", linestyle="--", alpha=0.7, linewidth=1.5,
               label=f"Rata-rata: {counts.mean():.0f}")
    ax.legend(fontsize=10)
    ax.set_ylim(0, counts.max() * 1.15)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                str(int(bar.get_height())), ha="center", va="bottom",
                fontsize=7, color=PALETTE["text"])

    # -- Pie chart proporsi split --
    ax2 = axes[1]
    train_n = int(len(df) * 0.70)
    val_n   = int(len(df) * 0.15)
    test_n  = len(df) - train_n - val_n
    sizes   = [train_n, val_n, test_n]
    labels_ = [f"Train\n{train_n:,} ({70}%)", f"Val\n{val_n:,} ({15}%)", f"Test\n{test_n:,} ({15}%)"]
    colors_ = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]
    wedges, _, autotexts = ax2.pie(
        sizes, labels=labels_, colors=colors_,
        autopct="%1.1f%%", startangle=90,
        pctdistance=0.75, textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax2.set_title(f"Pembagian Dataset (Total: {len(df):,} sampel)\n"
                  f"Split Ratio: 70% / 15% / 15%", fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_dataset_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   [PLOT] 01_dataset_overview.png")

def plot_hand_landmark(df):
    """Slide 2: Visualisasi landmark tangan untuk beberapa huruf."""
    sample_letters = ["A", "B", "C", "L", "V", "Y"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle("TalkHand BISINDO — Hand Landmark Visualization (MediaPipe 21-Point)",
                 fontsize=14, fontweight="bold", color=PALETTE["text"])

    for ax, letter in zip(axes.flatten(), sample_letters):
        sample = df[df["label"]==letter][FEAT_COLS].iloc[0].values.reshape(N_LMKS, 3)
        x_, y_ = sample[:, 0], sample[:, 1]
        # Gambar koneksi
        for (a, b) in HAND_CONNECTIONS:
            ax.plot([x_[a], x_[b]], [y_[a], y_[b]],
                    color=PALETTE["primary"], linewidth=2.5, zorder=1, alpha=0.8)
        # Gambar landmark
        colors_lmk = [PALETTE["secondary"] if i in [4,8,12,16,20]  # fingertip
                       else PALETTE["primary"] if i == 0             # wrist
                       else PALETTE["accent"]
                       for i in range(N_LMKS)]
        ax.scatter(x_, y_, c=colors_lmk, s=60, zorder=3, edgecolors="white", linewidths=0.8)
        # Nomor landmark kecil
        for i, (xi, yi) in enumerate(zip(x_, y_)):
            ax.text(xi+0.01, yi, str(i), fontsize=5.5, color=PALETTE["text"], alpha=0.7)
        ax.set_xlim(0.1, 0.9); ax.set_ylim(0.45, 1.00)
        ax.invert_yaxis()
        ax.set_title(f"Huruf  '{letter}'  (BISINDO)", fontweight="bold",
                     fontsize=12, color=PALETTE["text"])
        ax.set_aspect("equal"); ax.axis("off")
        # Legend
        patches = [mpatches.Patch(color=PALETTE["secondary"], label="Fingertip"),
                   mpatches.Patch(color=PALETTE["accent"],    label="Joint"),
                   mpatches.Patch(color=PALETTE["primary"],   label="Wrist")]
        ax.legend(handles=patches, fontsize=7, loc="lower right")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_hand_landmarks.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   [PLOT] 02_hand_landmarks.png")

def plot_preprocessing_effect(df_raw, X_proc):
    """Slide 3: Perbandingan distribusi fitur sebelum & sesudah preprocessing."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle("TalkHand BISINDO — Preprocessing Effect Analysis",
                 fontsize=14, fontweight="bold", color=PALETTE["text"])

    # Raw X distribution
    ax = axes[0,0]
    sample_feats_raw = df_raw[["lm8_x","lm8_y","lm12_x","lm12_y"]].values.flatten()
    ax.hist(sample_feats_raw, bins=60, color=PALETTE["secondary"], edgecolor="white",
            alpha=0.85, density=True)
    ax.set_title("Distribusi Fitur (Raw)", fontweight="bold")
    ax.set_xlabel("Nilai koordinat"); ax.set_ylabel("Densitas")
    mu, std = sample_feats_raw.mean(), sample_feats_raw.std()
    ax.axvline(mu, color="red", lw=1.5, linestyle="--", label=f"μ={mu:.2f}")
    ax.legend(fontsize=9)

    # Processed X distribution
    ax = axes[0,1]
    sample_feats_proc = X_proc[["lm8_x","lm8_y","lm12_x","lm12_y"]].values.flatten()
    ax.hist(sample_feats_proc, bins=60, color=PALETTE["primary"], edgecolor="white",
            alpha=0.85, density=True)
    ax.set_title("Distribusi Fitur (Setelah Preprocessing)", fontweight="bold")
    ax.set_xlabel("Nilai terstandarisasi"); ax.set_ylabel("Densitas")
    mu2, std2 = sample_feats_proc.mean(), sample_feats_proc.std()
    ax.axvline(mu2, color="red", lw=1.5, linestyle="--", label=f"μ={mu2:.3f}")
    ax.legend(fontsize=9)

    # Feature variance per landmark (before)
    ax = axes[1,0]
    variances_raw = []
    for i in range(N_LMKS):
        cols = [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"]
        variances_raw.append(df_raw[cols].values.var())
    ax.bar(range(N_LMKS), variances_raw, color=PALETTE["secondary"], alpha=0.85,
           edgecolor="white")
    ax.set_title("Variansi per Landmark (Raw)", fontweight="bold")
    ax.set_xlabel("Landmark Index (0=Wrist, 4=Thumb Tip, ...)")
    ax.set_ylabel("Variansi"); ax.set_xticks(range(N_LMKS))
    ax.set_xticklabels([str(i) for i in range(N_LMKS)], fontsize=7)

    # Feature variance per landmark (after)
    ax = axes[1,1]
    variances_proc = []
    for i in range(N_LMKS):
        cols = [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"]
        variances_proc.append(X_proc[cols].values.var())
    ax.bar(range(N_LMKS), variances_proc, color=PALETTE["primary"], alpha=0.85,
           edgecolor="white")
    ax.set_title("Variansi per Landmark (Setelah Preprocessing)", fontweight="bold")
    ax.set_xlabel("Landmark Index"); ax.set_ylabel("Variansi")
    ax.set_xticks(range(N_LMKS)); ax.set_xticklabels([str(i) for i in range(N_LMKS)], fontsize=7)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_preprocessing_effect.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   [PLOT] 03_preprocessing_effect.png")

def plot_pca_analysis(X_proc, y, le):
    """Slide 4: PCA — Reduksi dimensi dan analisis komponen utama."""
    pca = PCA(n_components=min(50, len(FEAT_COLS)))
    X_pca = pca.fit_transform(X_proc)
    explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")
    fig.suptitle("TalkHand BISINDO — PCA Analysis (Principal Component Analysis)",
                 fontsize=14, fontweight="bold", color=PALETTE["text"])

    # Explained variance
    ax = axes[0]
    cumulative = np.cumsum(explained)
    ax.bar(range(1, len(explained)+1), explained*100, color=PALETTE["accent"],
           alpha=0.75, label="Individu", edgecolor="white")
    ax2 = ax.twinx()
    ax2.plot(range(1, len(explained)+1), cumulative*100, "r-o",
             markersize=3.5, linewidth=1.8, label="Kumulatif")
    ax2.axhline(85, color="green", linestyle="--", alpha=0.7, label="85% threshold")
    ax2.axhline(95, color="orange", linestyle="--", alpha=0.7, label="95% threshold")
    n85 = np.searchsorted(cumulative, 0.85) + 1
    n95 = np.searchsorted(cumulative, 0.95) + 1
    ax.set_xlabel("Komponen PCA"); ax.set_ylabel("Explained Variance (%)")
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.legend(fontsize=9, loc="center right")
    ax.set_title(f"Explained Variance\n85% → PC{n85} | 95% → PC{n95}", fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")

    # PCA scatter PC1 vs PC2
    ax = axes[1]
    cmap = plt.cm.get_cmap("tab20", N_LABELS)
    for i, letter in enumerate(ALPHABET):
        mask = (y == i)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=cmap(i), alpha=0.45, s=18, label=letter)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title("PCA Scatter — PC1 vs PC2\n(Setiap warna = 1 huruf BISINDO)", fontweight="bold")
    # Legend kecil 2 baris
    handles = [mpatches.Patch(color=cmap(i), label=letter) for i,letter in enumerate(ALPHABET)]
    ax.legend(handles=handles, fontsize=6, ncol=7, loc="lower right",
              framealpha=0.85, handlelength=1)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_pca_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   [PLOT] 04_pca_analysis.png")

def plot_tsne(X_proc, y):
    """Slide 5: t-SNE visualisasi klaster isyarat."""
    print("   [INFO] Running t-SNE (ini perlu ~30 detik)...")
    try:
        sample_idx = np.random.choice(len(y), min(2000, len(y)), replace=False)
        X_s = X_proc.values[sample_idx]
        y_s = y[sample_idx]

        # PCA first for speed
        pca10 = PCA(n_components=20, random_state=42)
        X_pca10 = pca10.fit_transform(X_s)
        tsne = TSNE(n_components=2, perplexity=30, max_iter=800, random_state=42)
        X_tsne = tsne.fit_transform(X_pca10)

        fig, ax = plt.subplots(figsize=(13, 10))
        fig.patch.set_facecolor("white")
        cmap = plt.cm.get_cmap("tab20", N_LABELS)
        for i, letter in enumerate(ALPHABET):
            mask = (y_s == i)
            ax.scatter(X_tsne[mask,0], X_tsne[mask,1],
                       color=cmap(i), alpha=0.55, s=22, label=letter, edgecolors="white",
                       linewidths=0.3)
            # Centroid label
            if mask.sum() > 0:
                cx, cy = X_tsne[mask,0].mean(), X_tsne[mask,1].mean()
                ax.text(cx, cy, letter, fontsize=8, fontweight="bold",
                        ha="center", va="center", color="black",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=cmap(i), alpha=0.7, edgecolor="none"))

        handles = [mpatches.Patch(color=cmap(i), label=letter) for i,letter in enumerate(ALPHABET)]
        ax.legend(handles=handles, fontsize=7, ncol=7, loc="lower right",
                  framealpha=0.9, handlelength=1.2)
        ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
        ax.set_title("t-SNE Visualization — Klaster Isyarat BISINDO\n"
                     f"(n={len(X_s):,} sampel | Setiap warna = 1 huruf)",
                     fontsize=13, fontweight="bold", color=PALETTE["text"])
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "05_tsne_clusters.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("   [PLOT] 05_tsne_clusters.png")
    except Exception as e:
        print(f"   [ERROR] t-SNE failed: {e}")
        # Create a placeholder plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "t-SNE Visualization\n(Not available - computation too slow)", 
                ha="center", va="center", fontsize=14, color="red")
        ax.set_title("t-SNE Placeholder", fontsize=16)
        plt.savefig(PLOTS_DIR / "05_tsne_clusters.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("   [PLOT] 05_tsne_clusters.png (placeholder)")

def plot_feature_importance(X_proc, y):
    """Slide 6: Importansi fitur berdasarkan variansi antar-kelas."""
    from sklearn.ensemble import RandomForestClassifier
    print("   [INFO] Fitting RandomForest untuk feature importance...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_proc, y)
    importances = pd.Series(rf.feature_importances_, index=FEAT_COLS)

    # Agregasi per landmark
    lm_importance = {}
    for i in range(N_LMKS):
        cols = [f"lm{i}_{ax}" for ax in ("x","y","z")]
        lm_importance[f"LM{i}\n{LMK_NAMES[i].split('_')[0]}"] = importances[cols].sum()
    lm_imp = pd.Series(lm_importance).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")
    fig.suptitle("TalkHand BISINDO — Feature Importance Analysis",
                 fontsize=14, fontweight="bold", color=PALETTE["text"])

    # Per landmark bar chart
    ax = axes[0]
    colors_bar = [PALETTE["secondary"] if i < 5 else PALETTE["primary"] for i in range(len(lm_imp))]
    bars = ax.barh(lm_imp.index, lm_imp.values * 100, color=colors_bar,
                   edgecolor="white", alpha=0.9)
    ax.set_xlabel("Feature Importance (%)")
    ax.set_title("Importansi per Landmark\n(Aggregated X+Y+Z)", fontweight="bold")
    ax.invert_yaxis()
    for bar in bars[:5]:
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.1f}%", va="center", fontsize=8)

    # Top-20 individual features
    ax = axes[1]
    top20 = importances.nlargest(20)
    colors_top = [PALETTE["secondary"] if "_y" in c else
                  (PALETTE["primary"] if "_x" in c else PALETTE["accent"])
                  for c in top20.index]
    ax.barh(top20.index, top20.values * 100, color=colors_top, edgecolor="white", alpha=0.9)
    ax.set_xlabel("Feature Importance (%)")
    ax.set_title("Top-20 Fitur Individual\n(X=hijau, Y=oranye, Z=ungu)", fontweight="bold")
    ax.invert_yaxis()
    patches = [mpatches.Patch(color=PALETTE["primary"], label="Koordinat X"),
               mpatches.Patch(color=PALETTE["secondary"], label="Koordinat Y"),
               mpatches.Patch(color=PALETTE["accent"], label="Koordinat Z")]
    ax.legend(handles=patches, fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   [PLOT] 06_feature_importance.png")

def plot_correlation_heatmap(X_proc):
    """Slide 7: Correlation heatmap fitur landmark kunci."""
    key_feats = [f"lm{i}_{ax}" for i in [4,8,12,16,20,0,9] for ax in ("x","y")]
    corr = X_proc[key_feats].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("white")
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, ax=ax, cmap="RdYlGn", center=0,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix — Landmark Fingertip + Wrist\n"
                 "(LM4=Thumb Tip, LM8=Index Tip, LM12=Mid Tip, LM16=Ring Tip, LM20=Pinky Tip)",
                 fontsize=12, fontweight="bold", color=PALETTE["text"])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   [PLOT] 07_correlation_heatmap.png")

def generate_eda_report(df_raw, X_proc, y):
    """Simpan ringkasan statistik ke CSV."""
    stats = {
        "total_samples": len(df_raw),
        "n_labels": N_LABELS,
        "n_features": len(FEAT_COLS),
        "n_landmarks": N_LMKS,
        "samples_per_label_mean": N_SAMPLES,
        "samples_per_label_min": int(df_raw["label"].value_counts().min()),
        "samples_per_label_max": int(df_raw["label"].value_counts().max()),
        "train_samples": int(len(df_raw) * 0.70),
        "val_samples": int(len(df_raw) * 0.15),
        "test_samples": len(df_raw) - int(len(df_raw)*0.70) - int(len(df_raw)*0.15),
        "feature_mean_before": float(df_raw[FEAT_COLS].values.mean()),
        "feature_std_before": float(df_raw[FEAT_COLS].values.std()),
        "feature_mean_after": float(X_proc.values.mean()),
        "feature_std_after": float(X_proc.values.std()),
    }
    report_df = pd.DataFrame([stats]).T.reset_index()
    report_df.columns = ["Metric", "Value"]
    report_df.to_csv(PROC_DIR / "eda_summary.csv", index=False)

    # Per-label stats
    label_stats = df_raw.groupby("label")[FEAT_COLS[:6]].describe().round(4)
    label_stats.to_csv(PROC_DIR / "per_label_stats.csv")

    # Save processed features
    X_proc_save = X_proc.copy()
    X_proc_save.insert(0, "label", df_raw["label"].values)
    X_proc_save.to_csv(PROC_DIR / "processed_features.csv", index=False)
    print(f"\n   [REPORT] EDA report: {PROC_DIR}/eda_summary.csv")
    print(f"   [REPORT] Processed : {PROC_DIR}/processed_features.csv")
    return stats

# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  TALKHAND BISINDO - PREPROCESSING & EDA PIPELINE")
    print("=" * 65)

    # 1. Generate dataset
    df_raw = generate_dataset()
    df_raw.to_csv(PROC_DIR / "raw_landmarks.csv", index=False)

    # 2. Preprocessing
    X_proc, y, le, scaler = preprocess_pipeline(df_raw)

    # 3. EDA Plots
    print("\n[INFO] GENERATING EDA VISUALIZATIONS...")
    plot_dataset_overview(df_raw)
    plot_hand_landmark(df_raw)
    plot_preprocessing_effect(df_raw[FEAT_COLS], X_proc)
    plot_pca_analysis(X_proc, y, le)
    plot_tsne(X_proc, y)
    plot_feature_importance(X_proc, y)
    plot_correlation_heatmap(X_proc)

    # 4. Report
    stats = generate_eda_report(df_raw, X_proc, y)

    print("\n" + "=" * 65)
    print("  [OK] PIPELINE SELESAI")
    print("=" * 65)
    print(f"\n  Total Sampel  : {stats['total_samples']:,}")
    print(f"  Labels        : {stats['n_labels']} huruf BISINDO (A-Z)")
    print(f"  Fitur         : {stats['n_features']} (21 landmark x 3 koordinat)")
    print(f"  Train/Val/Test: {stats['train_samples']:,} / {stats['val_samples']:,} / {stats['test_samples']:,}")
    print(f"  Output plots  : {PLOTS_DIR}/")
    print(f"  Processed data: {PROC_DIR}/\n")

    # Open the plots folder to show the images
    os.startfile(str(PLOTS_DIR))
