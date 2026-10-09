import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Load and clean the dataset
# --------------------------------------------------

NUM_SEED = 42
np.random.seed(NUM_SEED)

df = pd.read_parquet(
    "hf://datasets/yevgeniy03/"
    "home-telecom-callcenter-transcripts-viewer/data/train.parquet"
)

# Prefer inferred dialogue, with raw text as a fallback.
if "dialogue_text_inferred" in df.columns:
    inferred = df["dialogue_text_inferred"].fillna("").astype(str)

    if "text" in df.columns:
        raw = df["text"].fillna("").astype(str)
        df["analysis_text"] = inferred.where(
            inferred.str.strip().ne(""), raw
        )
    else:
        df["analysis_text"] = inferred
else:
    df["analysis_text"] = df["text"].fillna("").astype(str)

df["analysis_text"] = df["analysis_text"].str.strip()

df = df[
    df["analysis_text"].ne("")
].copy().reset_index(drop=True)

print("Dataset shape:", df.shape)
print("\nAvailable columns:")
print(df.columns.tolist())


# --------------------------------------------------
# 2. Prepare numerical features
# --------------------------------------------------

numeric_candidates = [
    "audio_duration_seconds",
    "word_count",
    "words_per_second",
    "turn_count",
    "agent_turn_count",
    "customer_turn_count",
    "ivr_turn_count",
    "unknown_turn_count",
    "redaction_token_count",
    "unique_redaction_token_count",
    "confidence",
    "avg_word_confidence",
]

for column in numeric_candidates:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

# Derive speaking rate if the supplied field is unavailable.
if (
    "words_per_second" not in df.columns
    or df["words_per_second"].notna().sum() == 0
):
    df["words_per_second"] = (
        df["word_count"]
        / df["audio_duration_seconds"].replace(0, np.nan)
    )

df = df.replace([np.inf, -np.inf], np.nan)

numeric_features = [
    column
    for column in numeric_candidates
    if column in df.columns and df[column].notna().any()
]

if not numeric_features:
    raise ValueError("No usable numerical features were found.")

imputer = SimpleImputer(strategy="median")

X_numeric = imputer.fit_transform(
    df[numeric_features]
)

numeric_scaler = StandardScaler()

X_numeric = numeric_scaler.fit_transform(X_numeric)


# --------------------------------------------------
# 3. Turn transcript text into numerical features
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    stop_words="english",
    sublinear_tf=True,
)

X_text_sparse = vectorizer.fit_transform(
    df["analysis_text"]
)

if min(X_text_sparse.shape) < 2:
    raise ValueError(
        "Insufficient text data for dimensionality reduction."
    )

n_components = min(
    30,
    X_text_sparse.shape[0] - 1,
    X_text_sparse.shape[1] - 1,
)

svd = TruncatedSVD(
    n_components=n_components,
    random_state=NUM_SEED,
)

X_text = svd.fit_transform(X_text_sparse)

text_scaler = StandardScaler()

X_text = text_scaler.fit_transform(X_text)


# Combine text patterns and numerical statistics.
X = np.hstack([X_numeric, X_text])

print("\nText feature matrix:", X_text.shape)
print("Combined feature matrix:", X.shape)


# --------------------------------------------------
# 4. Train the anomaly detection model
# --------------------------------------------------

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=NUM_SEED,
    n_jobs=-1,
)

model.fit(X)

# Higher raw scores mean greater unusualness.
df["Anomaly_Score"] = -model.score_samples(X)

# Flag the top 5% of scores for review.
threshold = np.percentile(
    df["Anomaly_Score"], 95
)

df["Anomaly_Flag"] = np.where(
    df["Anomaly_Score"] >= threshold,
    -1,
    1,
)

df["Status"] = df["Anomaly_Flag"].map({
    1: "Baseline-like",
    -1: "Flagged anomaly",
})

# Scale scores to the range 0–1 for the graph.
score_min = df["Anomaly_Score"].min()
score_max = df["Anomaly_Score"].max()

if score_max > score_min:
    df["Score_01"] = (
        (df["Anomaly_Score"] - score_min)
        / (score_max - score_min)
    )
else:
    df["Score_01"] = 0.0

print("\nAnomaly threshold:", round(threshold, 4))
print("Flagged records:", (df["Anomaly_Flag"] == -1).sum())


# --------------------------------------------------
# 5. Three-dimensional anomaly graph
# --------------------------------------------------

plot_3d = df.dropna(
    subset=[
        "audio_duration_seconds",
        "words_per_second",
        "Score_01",
    ]
).copy()

fig = plt.figure(figsize=(13, 9))

ax = fig.add_subplot(111, projection="3d")

blue_purple_red = LinearSegmentedColormap.from_list(
    "blue_purple_red",
    [
        "#0000FF",  # Blue: safest
        "#8000FF",  # Purple: intermediate
        "#FF0000",  # Red: most unusual
    ]
)

points = ax.scatter(
    plot_3d["audio_duration_seconds"],
    plot_3d["Score_01"],
    plot_3d["words_per_second"],
    c=plot_3d["Score_01"],
    cmap=blue_purple_red,
    vmin=0,
    vmax=1,
    s=40,
    alpha=0.8,
    edgecolors="none",
)

ax.set_title(
    "3D Call Transcript Anomaly Analysis",
    fontsize=15,
    fontweight="bold",
    pad=20,
)

ax.set_xlabel("Call Duration (seconds)", labelpad=12)
ax.set_ylabel(
    "Relative Anomaly Score (higher = more unusual)",
    labelpad=15,
)
ax.set_zlabel("Words Per Second", labelpad=12)

colorbar = fig.colorbar(
    points, ax=ax, pad=0.12, shrink=0.7
)

colorbar.set_label("Relative Anomaly Score")

colorbar.set_ticks([0, 0.5, 1])
colorbar.set_ticklabels([
    "Less unusual",
    "Intermediate",
    "More unusual",
])

ax.view_init(elev=25, azim=-55)

plt.tight_layout()
plt.savefig("call_anomaly_3d.png", dpi=180)
plt.show()


# --------------------------------------------------
# 6. Two-dimensional anomaly graph
# --------------------------------------------------

plot_2d = df.dropna(
    subset=[
        "word_count",
        "words_per_second",
    ]
).copy()

plt.figure(figsize=(11, 7))

sns.scatterplot(
    data=plot_2d,
    x="word_count",
    y="words_per_second",
    hue="Status",
    hue_order=[
        "Baseline-like",
        "Flagged anomaly",
    ],
    palette={
        "Baseline-like": "#2166ac",
        "Flagged anomaly": "#d73027",
    },
    alpha=0.8,
    s=55,
)

plt.title(
    "Transcript Anomaly Detection",
    fontsize=15,
    fontweight="bold",
)

plt.xlabel("Word Count")
plt.ylabel("Words Per Second")

plt.grid(
    True,
    linestyle="--",
    alpha=0.35,
)

plt.legend(title="Model Classification")
plt.tight_layout()

plt.savefig("call_anomaly_2d.png", dpi=180)
plt.show()


# --------------------------------------------------
# 7. Display and export flagged records
# --------------------------------------------------

display_columns = [
    column
    for column in [
        "transcript_id",
        "audio_duration_seconds",
        "word_count",
        "words_per_second",
        "Anomaly_Score",
        "Status",
        "analysis_text",
    ]
    if column in df.columns
]

flagged = (
    df[df["Anomaly_Flag"] == -1]
    .sort_values("Anomaly_Score", ascending=False)
)

print("\nTop flagged transcript records:")

print(
    flagged[display_columns]
    .head(15)
    .to_string(index=False, max_colwidth=180)
)

df.to_csv("call_anomaly_results.csv", index=False)

print("\nSaved files:")
print("call_anomaly_3d.png")
print("call_anomaly_2d.png")
print("call_anomaly_results.csv")