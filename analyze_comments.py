#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse marketing des commentaires YouTube.
Classe les émotions primaires et génère des graphiques.

Émotions : colère, anticipation, dégoût, peur, joie, tristesse, surprise, confiance, neutre
"""
import argparse, glob, os, re, sys, datetime as dt
from collections import Counter
from typing import List

import pandas as pd
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import logging


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Tentative d'importer NRCLex ; si cela échoue plus tard lors de l'exécution en raison des corpus, nous reviendrons à la solution de secours.
try:
    from nrclex import NRCLex
    HAVE_NRC = True
except Exception:
    NRCLex = None
    HAVE_NRC = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from scipy import sparse

# ---- Matplotlib fallback pour les emoji/glyphs ----
matplotlib.rcParams['font.family'] = ['Segoe UI Emoji', 'Segoe UI Symbol', 'Arial Unicode MS', 'DejaVu Sans']
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


EMOJI_RE = re.compile(r'[\U00010000-\U0010FFFF]')
def strip_emoji(s):
    return EMOJI_RE.sub('', str(s))

EMOTION_ORDER = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust","neutral"]

URL_RE = re.compile(r"http[s]?://\S+")
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", re.UNICODE)

DEFAULT_STOPWORDS = set([
    "the","a","an","and","or","of","to","in","on","for","with","is","are","was","were","be","been","am",
    "i","you","he","she","it","we","they","me","him","her","them","my","your","his","its","our","their",
    "this","that","these","those","as","at","by","from","about","into","over","after","before","up","down",
    "not","no","yes","do","does","did","done","just","so","very","really","mais","les","des","une","un",
    "le","la","et","de","du","aux","au","pour","sur","dans","est","sont","été","être","je","tu","il","elle",
    "nous","vous","ils","elles","mon","ma","mes","tes","ses","leurs","leur","ce","cet","cette","ces","qui",
])

def preprocess_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = URL_RE.sub("", s)
    s = s.replace("\n", " ")
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s

def tokens_simple(s: str):
    return [t.lower() for t in TOKEN_RE.findall(s)]

# --- Fallback lexicon for offline emotion (Seulement si le NRC échoue) ---
BASIC_EMO_LEXICON = {
    "joy": {"love","great","awesome","amazing","happy","fantastic","enjoy","nice","cool","bravo","wonderful","géniale","super","adore"},
    "anger": {"angry","mad","furious","rage","hate","annoy","stupid","idiot","trash","worst","bête","nul","pire"},
    "sadness": {"sad","unhappy","upset","depressed","cry","tears","sorry","miss","triste","désolé","manque"},
    "fear": {"afraid","scared","fear","terrified","panic","worried","anxious","nervous","peur","inquiet","angoissé"},
    "disgust": {"disgust","gross","nasty","ew","yuck","cringe","dégueu","beurk","cringe"},
    "surprise": {"surprised","wow","shocked","unbelievable","omg","whoa","surpris","incroyable","impressionnant"},
    "trust": {"trust","reliable","credible","authentic","respect","appreciate","support","fiable","respecte","apprécie","soutien"},
    "anticipation": {"excited","looking","waiting","hope","soon","expect","anticipate","impatient","hâte","attend","espère"},
}

def primary_emotion(text: str) -> str:
    s = preprocess_text(text)
    if not s:
        return "neutral"
    # On essaie d'abord NRC s'il est disponible.
    if HAVE_NRC and NRCLex is not None:
        try:
            emo = NRCLex(s)
            scores = emo.raw_emotion_scores or {}
            if scores:
                # on choisit notre max ; départage par ordre fixe
                best = None; best_val = -1
                for e in EMOTION_ORDER[:-1]:
                    v = scores.get(e, 0)
                    if v > best_val:
                        best_val = v; best = e
                return best if best is not None else "neutral"
        except Exception:
            # fall back below
            pass
    # Fallback simple lexicon
    toks = tokens_simple(s)
    scores = {k: 0 for k in BASIC_EMO_LEXICON}
    for t in toks:
        for emo, vocab in BASIC_EMO_LEXICON.items():
            if t in vocab:
                scores[emo] += 1
    emo = max(scores, key=scores.get)
    return emo if scores[emo] > 0 else "neutral"

def label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "text" not in df.columns:
        raise ValueError("Input data must have a 'text' column.")
    analyzer = SentimentIntensityAnalyzer()
    tqdm.pandas(desc="Scoring emotion/sentiment")
    df["text_clean"] = df["text"].fillna("").astype(str).map(preprocess_text)
    df["primary_emotion"] = df["text_clean"].progress_map(primary_emotion)
    df["vader_compound"] = df["text_clean"].map(lambda s: analyzer.polarity_scores(s)["compound"] if s else 0.0)
    return df

def read_inputs(patterns: List[str]) -> pd.DataFrame:
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        raise FileNotFoundError("No input files matched. Try --inputs out\\*.csv")
    frames = []
    for path in files:
        if path.lower().endswith(".csv"):
            frames.append(pd.read_csv(path))
        elif path.lower().endswith(".json"):
            frames.append(pd.read_json(path))
    if not frames:
        raise FileNotFoundError("No CSV/JSON files found among inputs.")
    df = pd.concat(frames, ignore_index=True)
    return df

def ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

# ---------- Plotting helpers ----------
def plot_emotion_distribution(df: pd.DataFrame, outdir: str):
    counts = df["primary_emotion"].value_counts().reindex(EMOTION_ORDER, fill_value=0)
    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title("Emotion distribution")
    plt.xlabel("Emotion"); plt.ylabel("Count")
    plt.tight_layout()
    path = os.path.join(outdir, "emotions_distribution.png")
    plt.savefig(path); plt.close()
    return path

def plot_emotions_by_video(df: pd.DataFrame, outdir: str, top_n: int):
    if "videoTitle" not in df.columns and "videoId" not in df.columns:
        return None
    video_key = "videoTitle" if "videoTitle" in df.columns and df["videoTitle"].notna().any() else "videoId"
    # enlève les emojis dans les titres
    df = df.copy()
    df[video_key] = df[video_key].map(strip_emoji)
    top_videos = df[video_key].value_counts().head(top_n).index.tolist()
    sub = df[df[video_key].isin(top_videos)].copy()
    pivot = pd.pivot_table(sub, index=video_key, columns="primary_emotion", values="commentId", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(columns=EMOTION_ORDER, fill_value=0)
    plt.figure(figsize=(14, 6))
    pivot.plot(kind="bar", stacked=True)
    plt.title(f"Emotions by video (top {top_n})")
    plt.xlabel("Video"); plt.ylabel("Comments")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(outdir, "emotions_by_video_topN.png")
    plt.savefig(path); plt.close()
    return path

def plot_emotions_over_time(df: pd.DataFrame, outdir: str, freq: str = "W"):
    if "publishedAt" not in df.columns or df["publishedAt"].isna().all():
        return None
    def parse_dt(x):
        if pd.isna(x): return pd.NaT
        s = str(x)
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return pd.to_datetime(s, format=fmt, utc=True)
            except Exception:
                pass
        try:
            return pd.to_datetime(s, utc=True, errors="coerce")
        except Exception:
            return pd.NaT
    ts = df.copy()
    ts["dt"] = ts["publishedAt"].map(parse_dt)
    ts = ts.dropna(subset=["dt"])
    if ts.empty: 
        return None
    grp = ts.groupby([pd.Grouper(key="dt", freq=freq), "primary_emotion"])["commentId"].count().unstack(fill_value=0)
    grp = grp.reindex(columns=EMOTION_ORDER, fill_value=0)
    plt.figure(figsize=(12, 6))
    for col in grp.columns:
        plt.plot(grp.index, grp[col], label=col)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.title(f"Emotions over time ({freq})")
    plt.xlabel("Date"); plt.ylabel("Comments")
    plt.tight_layout()
    path = os.path.join(outdir, "emotions_over_time.png")
    plt.savefig(path); plt.close()
    return path

def build_wordclouds(df: pd.DataFrame, outdir: str, min_count: int = 30):
    paths = []
    for emo in EMOTION_ORDER:
        if emo == "neutral": 
            continue
        texts = df.loc[df["primary_emotion"] == emo, "text_clean"]
        tokens = []
        for s in texts:
            tokens.extend([t for t in tokens_simple(s) if t not in DEFAULT_STOPWORDS and len(t) > 2])
        if len(tokens) < min_count:
            continue
        freqs = Counter(tokens)
        wc = WordCloud(width=1200, height=800, background_color="white")
        wc.generate_from_frequencies(freqs)
        path = os.path.join(outdir, f"top_words_{emo}.png")
        wc.to_file(path)
        paths.append(path)
    return paths

def write_excel_report(df_labeled: pd.DataFrame, outdir: str):
    out_xlsx = os.path.join(outdir, "marketing_report.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        df_labeled.to_excel(xw, index=False, sheet_name="comments_labeled")
        summary = df_labeled["primary_emotion"].value_counts().reindex(EMOTION_ORDER, fill_value=0).rename_axis("emotion").reset_index(name="count")
        total = summary["count"].sum()
        summary["share"] = (summary["count"] / total).round(4)
        summary.to_excel(xw, index=False, sheet_name="summary_emotions")
        if "videoTitle" in df_labeled.columns or "videoId" in df_labeled.columns:
            key = "videoTitle" if "videoTitle" in df_labeled.columns and df_labeled["videoTitle"].notna().any() else "videoId"
            by_vid = pd.pivot_table(df_labeled, index=key, columns="primary_emotion", values="commentId", aggfunc="count", fill_value=0)
            by_vid.to_excel(xw, sheet_name="by_video")
    return out_xlsx

# ---------- Clustering (Segmentation des auteurs) ----------
def parse_dt_any(x):
    if pd.isna(x): return pd.NaT
    s = str(x)
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s, format=fmt, utc=True)
        except Exception:
            pass
    try:
        return pd.to_datetime(s, utc=True, errors="coerce")
    except Exception:
        return pd.NaT

def author_key_row(row):
    cid = row.get("authorChannelId")
    if isinstance(cid, str) and cid.strip():
        return cid.strip()
    name = row.get("author")
    if isinstance(name, str) and name.strip():
        return "name:" + name.strip()
    return "unknown"

def build_author_features(df: pd.DataFrame, max_features: int = 1500, min_comments: int = 1):
    df = df.copy()
    df["author_key"] = df.apply(author_key_row, axis=1)
    df["text_clean"] = df["text_clean"].fillna("")
    if "isReply" not in df.columns: df["isReply"] = False
    if "likeCount" not in df.columns: df["likeCount"] = 0
    if "publishedAt" in df.columns:
        dt_col = df["publishedAt"].map(parse_dt_any)
        hours = dt_col.dt.hour
        df["_tod_morn"] = hours.between(6, 11, inclusive="both").astype(float)
        df["_tod_aft"]  = hours.between(12, 17, inclusive="both").astype(float)
        df["_tod_eve"]  = hours.between(18, 23, inclusive="both").astype(float)
        df["_tod_night"]= hours.between(0, 5, inclusive="both").astype(float)
    else:
        df["_tod_morn"]=0.0; df["_tod_aft"]=0.0; df["_tod_eve"]=0.0; df["_tod_night"]=0.0

    # agrégation par auteur (numérique + émotions)
    agg = df.groupby("author_key").agg(
        author=("author", "first"),
        authorChannelId=("authorChannelId","first"),
        n_comments=("commentId","count"),
        avg_len=("text_clean", lambda s: s.str.split().map(len).mean() if len(s)>0 else 0.0),
        reply_share=("isReply", "mean"),
        mean_likes=("likeCount", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).mean()),
        sent_mean=("vader_compound","mean"),
        sent_std=("vader_compound","std"),
        tod_morn=("_tod_morn","mean"),
        tod_aft=("_tod_aft","mean"),
        tod_eve=("_tod_eve","mean"),
        tod_night=("_tod_night","mean"),
    ).reset_index()

    emo_dummies = pd.get_dummies(df["primary_emotion"]).reindex(columns=EMOTION_ORDER, fill_value=0)
    emo_dummies["author_key"] = df["author_key"].values
    emo_share = emo_dummies.groupby("author_key").mean().reset_index()
    features = agg.merge(emo_share, on="author_key", how="left")
    features = features[features["n_comments"] >= min_comments].copy()

    texts = df.groupby("author_key")["text_clean"].apply(lambda s: " ".join(s.astype(str))).reindex(features["author_key"]).fillna("")
    tfv = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), min_df=2)
    X_text = tfv.fit_transform(texts.values)

    numeric_cols = ["n_comments","avg_len","reply_share","mean_likes","sent_mean","sent_std","tod_morn","tod_aft","tod_eve","tod_night"] + EMOTION_ORDER
    X_num = features[numeric_cols].fillna(0.0).astype(float).values
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_num_scaled = scaler.fit_transform(X_num)

    X_combined = sparse.hstack([sparse.csr_matrix(X_num_scaled), X_text], format="csr")

    return {
        "features_df": features,
        "tfidf": tfv,
        "numeric_cols": numeric_cols,
        "scaler": scaler,
        "text_offset": X_num_scaled.shape[1],
        "X": X_combined
    }

def choose_k_auto(X, k_min=3, k_max=10, sample_limit=2000, random_state=42):
    best_k, best_score = None, -1.0
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        if X.shape[0] > sample_limit:
            import numpy as np
            idx = np.random.RandomState(42).choice(X.shape[0], size=sample_limit, replace=False)
            score = silhouette_score(X[idx], labels[idx])
        else:
            score = silhouette_score(X, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k, best_score

def cluster_authors(meta, k=None, auto_k=False, random_state=42):
    X = meta["X"]
    if auto_k:
        k, _ = choose_k_auto(X)
    if not k: k = 5
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    svd = TruncatedSVD(n_components=2, random_state=random_state)
    coords = svd.fit_transform(X)
    return {"labels": labels, "cluster_model": km, "coords2d": coords, "k": k}

def cluster_top_terms(km, meta, topn=10):
    centers = km.cluster_centers_
    vocab = meta["tfidf"].get_feature_names_out()
    offset = meta["text_offset"]
    out = {}
    for c in range(centers.shape[0]):
        vec = centers[c, offset:]
        if vec.size == 0:
            out[c] = []
            continue
        import numpy as np
        top_idx = np.argsort(vec)[::-1][:topn]
        out[c] = [vocab[i] for i in top_idx if i < len(vocab)]
    return out

def save_cluster_outputs(df_labeled, meta, clustering, outdir):
    feats = meta["features_df"].copy()
    feats["cluster"] = clustering["labels"]
    feats["display_name"] = feats["author"].fillna(feats["authorChannelId"]).fillna(feats["author_key"])
    # Attach a sample comment
    sample_map = df_labeled.groupby(df_labeled.apply(lambda r: r.get("authorChannelId") or ("name:"+str(r.get("author")) if pd.notna(r.get("author")) else "unknown"), axis=1))["text"].first()
    feats["sample_comment"] = feats["author_key"].map(sample_map).fillna("")

    cols = ["author_key","display_name","authorChannelId","n_comments","cluster","avg_len","reply_share","mean_likes","sent_mean","sent_std"] + EMOTION_ORDER + ["sample_comment"]
    out_customers = feats[cols]
    out_path = os.path.join(outdir, "customers_clusters.csv")
    out_customers.to_csv(out_path, index=False)

    summary = out_customers.groupby("cluster").agg(size=("author_key","count"), avg_comments=("n_comments","mean"), avg_sentiment=("sent_mean","mean")).reset_index()
    summary_path = os.path.join(outdir, "clusters_summary.csv")
    summary.to_csv(summary_path, index=False)

    # Graphiques
    plt.figure(figsize=(8,4))
    summary.set_index("cluster")["size"].plot(kind="bar")
    plt.title("Cluster sizes")
    plt.xlabel("Cluster"); plt.ylabel("Authors")
    plt.tight_layout()
    sizes_png = os.path.join(outdir, "cluster_sizes_bar.png")
    plt.savefig(sizes_png); plt.close()

    coords = clustering["coords2d"]; labels = clustering["labels"]
    import numpy as np
    plt.figure(figsize=(8,6))
    for c in sorted(set(labels)):
        mask = labels == c
        plt.scatter(coords[mask,0], coords[mask,1], s=12, label=f"Cluster {c}")
    plt.legend()
    plt.title("Author clusters (2D)")
    plt.xlabel("SVD-1"); plt.ylabel("SVD-2")
    plt.tight_layout()
    scatter_png = os.path.join(outdir, "clusters_scatter_2d.png")
    plt.savefig(scatter_png); plt.close()

    return out_path, summary_path, sizes_png, scatter_png

def main():
    ap = argparse.ArgumentParser(description="Analyze YouTube comments for emotion, charts, and customer clustering.")
    ap.add_argument("--inputs", nargs="+", default=["out/*.csv","out/*.json"], help="Input CSV/JSON glob(s)")
    ap.add_argument("--output-dir", default="out/analysis", help="Directory to write labeled data and charts")
    ap.add_argument("--top-videos", type=int, default=10, help="Top N videos in the by-video chart")
    ap.add_argument("--no-wordclouds", dest="no_wordclouds", action="store_true", help="Disable wordcloud images")
    # clustering options
    ap.add_argument("--cluster-authors", action="store_true", help="Enable customer clustering (author segmentation)")
    ap.add_argument("--k", type=int, default=None, help="Number of clusters for KMeans")
    ap.add_argument("--auto-k", dest="auto_k", action="store_true", help="Try multiple K and pick by silhouette")
    ap.add_argument("--min-comments", type=int, default=1, help="Minimum comments per author to include")
    ap.add_argument("--max-features", type=int, default=1500, help="Max TF-IDF features (unigram+bigram)")
    args = ap.parse_args()

    outdir = ensure_out_dir(args.output_dir)
    df = read_inputs(args.inputs)
    if "commentId" not in df.columns:
        df["commentId"] = range(1, len(df)+1)

    # Label
    df_labeled = label_dataframe(df)

    # Outputs standard
    labeled_path = os.path.join(outdir, "comments_labeled.csv")
    df_labeled.to_csv(labeled_path, index=False)

    summary = df_labeled["primary_emotion"].value_counts().reindex(EMOTION_ORDER, fill_value=0).rename_axis("emotion").reset_index(name="count")
    total = summary["count"].sum()
    summary["share"] = (summary["count"] / total).round(4)
    summary_path = os.path.join(outdir, "summary_emotions.csv")
    summary.to_csv(summary_path, index=False)

    plot_emotion_distribution(df_labeled, outdir)
    plot_emotions_by_video(df_labeled, outdir, top_n=args.top_videos)
    plot_emotions_over_time(df_labeled, outdir, freq="W")

    if not args.no_wordclouds:
        build_wordclouds(df_labeled, outdir)

    # clustering
    if args.cluster_authors:
        meta = build_author_features(df_labeled, max_features=args.max_features, min_comments=args.min_comments)
        clustering = cluster_authors(meta, k=args.k, auto_k=args.auto_k)
        top_terms = cluster_top_terms(clustering["cluster_model"], meta, topn=12)
        with open(os.path.join(outdir, "clusters_top_terms.txt"), "w", encoding="utf-8") as f:
            for c, terms in top_terms.items():
                f.write(f"Cluster {c}: " + ", ".join(terms) + "\n")
        save_cluster_outputs(df_labeled, meta, clustering, outdir)

    # Rapport Excel
    write_excel_report(df_labeled, outdir)

    print(f"Saved labeled comments: {labeled_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Outputs written to: {outdir}")

if __name__ == "__main__":
    main()
