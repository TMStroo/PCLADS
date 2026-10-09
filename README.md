# PCLADS: Call Transcript Anomaly Detection

PCLADS is a Python research project that explores anomaly detection in call-center transcript data. It combines transcript text features with call-level statistics, then uses an Isolation Forest to flag records that differ from the dataset's learned baseline.

> **Important:** A flagged transcript is an unusual data point, not evidence of a crime. This project does not estimate guilt or provide a validated criminal-risk score. Use its results for research and human review only.

## Features

- Loads the public Parquet dataset from Hugging Face with Pandas.
- Uses transcript text and available call statistics as model inputs.
- Converts text into TF-IDF features, then reduces their dimensionality with TruncatedSVD.
- Scales numerical features and combines them with text features.
- Fits a scikit-learn Isolation Forest to identify outliers.
- Creates a 3D plot with a blue-to-purple-to-red anomaly-score gradient.
- Creates a 2D plot comparing transcript word count with speaking rate.
- Exports scores and review flags to `call_anomaly_results.csv`.

## Example outputs

The Python script saves both plots in its working directory. Run the script first, then commit the two PNG files next to this README for them to display on GitHub.

### 3D anomaly plot

- **X-axis:** Call duration in seconds
- **Y-axis:** Relative anomaly score, scaled from 0 to 1
- **Z-axis:** Words per second
- **Color:** Blue means a lower relative anomaly score, purple means an intermediate score, and red means a higher relative anomaly score.

![3D call transcript anomaly plot](call_anomaly_3d.png)

### 2D anomaly plot

- **X-axis:** Transcript word count
- **Y-axis:** Words per second
- **Blue points:** Baseline-like according to the model
- **Red points:** Flagged as anomalies according to the selected threshold

![2D call transcript anomaly plot](call_anomaly_2d.png)

These colors describe model output only. They do not indicate whether a conversation is criminal or harmless.

## Dataset

This project uses the **Home/Telecom Call Center Transcript Viewer V2** dataset:

- **Dataset page:** [Hugging Face dataset](https://huggingface.co/datasets/yevgeniy03/home-telecom-callcenter-transcripts-viewer)
- **File used by the script:** `data/train.parquet`
- **Language:** English
- **Records:** 3,239 transcript rows, according to the dataset card
- **Format:** Parquet
- **License listed by the dataset:** CC BY-NC 4.0

The dataset is a viewer copy of call-center transcripts. It includes the original `text` field and a `dialogue_text_inferred` field with inferred speaker roles. The dataset card states that inferred role labels are inspection aids, not ground truth.

**License note:** CC BY-NC 4.0 restricts commercial use. Review the dataset's license and source information before redistributing the data or using it in a commercial project. This repository should link to the dataset rather than republish its Parquet file.

## How it works

1. **Load data:** Pandas reads the Parquet file directly from Hugging Face.
2. **Choose transcript text:** The script prefers `dialogue_text_inferred` and falls back to `text` when needed.
3. **Prepare numeric data:** Available features such as duration, word count, speaking rate, turn counts, and confidence fields are converted to numeric values. Missing values are imputed with the median.
4. **Represent text:** `TfidfVectorizer` creates word and bigram features. `TruncatedSVD` reduces these features to a smaller numerical representation.
5. **Combine features:** The script scales the numeric and text-derived features and joins them into one feature matrix.
6. **Detect outliers:** `IsolationForest` assigns anomaly scores. The script flags records at or above the 95th percentile of scores, which corresponds to a 5% review threshold on this run.
7. **Visualize and export:** Matplotlib and Seaborn generate plots, and Pandas writes the results CSV.

The current script fits and scores the same dataset. This is useful for an initial exploration, but it is not a held-out evaluation of how the model performs on new calls.

## Requirements

Use a Python environment with the following packages:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- PyArrow
- fsspec
- huggingface_hub

## Installation

From the project directory, install the dependencies into the Python interpreter or virtual environment used by the project:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn pyarrow fsspec huggingface_hub
```

If PyCharm uses a project virtual environment, select that environment as the project's interpreter before installing packages.

## Run the project

Run the Python file containing the detector. If your current filename is `LLM(with a real simulated database).py`, use:

```bash
python "LLM(with a real simulated database).py"
```

The script downloads the Parquet data as needed and writes these outputs to the process's working directory:

```text
call_anomaly_3d.png
call_anomaly_2d.png
call_anomaly_results.csv
```

The database in this version is a public call-center transcript dataset, not a fully simulated database. The unusual examples and criminal-activity labels are not supplied by this dataset.

## Output CSV

`call_anomaly_results.csv` contains the available original columns plus the fields added by the script, including:

| Column | Meaning |
|---|---|
| `analysis_text` | Transcript text selected for analysis |
| `Anomaly_Score` | Raw outlier score transformed so larger values indicate greater unusualness |
| `Anomaly_Flag` | `-1` for flagged records and `1` for baseline-like records |
| `Status` | Human-readable label derived from the anomaly flag |
| `Score_01` | Min-max-scaled anomaly score between 0 and 1, used for the 3D color and Y-axis |

Exact column names and the available numeric features depend on the source dataset version.

## Limitations and next steps

- **No criminal-activity labels:** The dataset does not provide reliable labels for criminal versus non-criminal calls. The detector identifies outliers, not criminal activity.
- **No ground-truth evaluation yet:** Precision, recall, false-positive rate, and detection quality have not been established.
- **Not an LLM:** The current version uses TF-IDF, TruncatedSVD, and Isolation Forest. It does not train or fine-tune a large language model.
- **No call start-time axis:** The current dataset fields used by the graph include call duration, not a verified call start timestamp. Therefore, the 3D X-axis is duration, not time of day.
- **Feature and data bias:** Unusual call lengths, speaking rates, redaction counts, transcription errors, or inferred speaker labels can affect anomaly scores.
- **Human review is necessary:** Do not use an anomaly flag alone to accuse, identify, or take action against a person.
- **Protect transcript data:** Transcripts can contain personal or sensitive information. Avoid publishing raw transcripts or exported results containing sensitive text.

## Tech stack

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Parquet`

## Attribution

Dataset: [Home/Telecom Call Center Transcript Viewer V2](https://huggingface.co/datasets/yevgeniy03/home-telecom-callcenter-transcripts-viewer), provided under the license listed on its dataset page.
