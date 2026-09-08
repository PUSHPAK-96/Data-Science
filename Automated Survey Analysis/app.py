import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

from src.prep import load_csv
from src.sentiment import sentiment_scores, sentiment_label
from src.keywords import top_keywords


st.set_page_config(
    page_title="Automated Survey Analysis",
    page_icon="📝",
    layout="wide"
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = APP_DIR / "data" / "sample_surveys.csv"


# ---------------------------------------------------------
# TEXT COLUMN DETECTION
# ---------------------------------------------------------

POSSIBLE_TEXT_COLUMNS = [
    "free_text",
    "feedback",
    "comment",
    "comments",
    "review",
    "reviews",
    "response",
    "responses",
    "text",
    "answer",
    "answers",
    "description",
    "remarks",
    "remark",
    "suggestion",
    "suggestions",
    "opinion",
    "message",
]


def detect_text_column(df):
    """
    Automatically detect the most likely text column.
    """

    # Exact / case-insensitive match
    column_map = {str(col).strip().lower(): col for col in df.columns}

    for name in POSSIBLE_TEXT_COLUMNS:
        if name.lower() in column_map:
            return column_map[name.lower()]

    # Fallback:
    # Find object/string columns with meaningful text
    text_columns = df.select_dtypes(include=["object", "string"]).columns

    candidates = []

    for col in text_columns:
        values = df[col].dropna().astype(str)

        if len(values) == 0:
            continue

        avg_length = values.str.len().mean()
        unique_ratio = values.nunique() / max(len(values), 1)

        # Prefer columns containing reasonably long text
        if avg_length >= 15:
            candidates.append((col, avg_length, unique_ratio))

    if candidates:
        # Select column with highest average text length
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    return None


# ---------------------------------------------------------
# LOAD CSV
# ---------------------------------------------------------

def load_flexible_csv(file_or_path):

    try:
        df = pd.read_csv(file_or_path)

    except Exception as e:
        raise ValueError(
            f"Unable to read this CSV file. Please upload a valid CSV file."
        ) from e

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    # Detect text column
    text_column = detect_text_column(df)

    if text_column is None:
        raise ValueError(
            "No text/feedback column could be detected."
        )

    # Rename detected column to free_text
    if text_column != "free_text":
        df = df.rename(columns={text_column: "free_text"})

    return df


# ---------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def analyze_df(file_or_path):

    # First try the flexible CSV loader
    df = load_flexible_csv(file_or_path)

    # Remove empty text
    df["free_text"] = (
        df["free_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove rows with empty text
    df = df[df["free_text"] != ""].copy()

    if df.empty:
        raise ValueError(
            "The text column does not contain any usable survey responses."
        )

    # Sentiment
    df["sentiment_score"] = sentiment_scores(df["free_text"])
    df["sentiment"] = df["sentiment_score"].map(sentiment_label)

    # Rating
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(
            df["rating"],
            errors="coerce"
        )

    return df


# ---------------------------------------------------------
# FILTERS
# ---------------------------------------------------------

def apply_filters(df, segs, min_rating, score_range, search):

    out = df.copy()

    if segs and "segment" in out.columns:
        out = out[out["segment"].isin(segs)]

    if min_rating is not None and "rating" in out.columns:
        out = out[
            out["rating"].fillna(-1) >= min_rating
        ]

    out = out[
        (out["sentiment_score"] >= score_range[0]) &
        (out["sentiment_score"] <= score_range[1])
    ]

    if search.strip():
        s = search.strip().lower()

        out = out[
            out["free_text"]
            .str.contains(
                s,
                case=False,
                na=False
            )
        ]

    return out


# ---------------------------------------------------------
# PAGINATION
# ---------------------------------------------------------

def paginate(df, page_size, page):

    start = page * page_size
    end = start + page_size

    return df.iloc[start:end], len(df)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("📝 Automated Survey Analysis")


# ---------------------------------------------------------
# CSV FORMAT HELP
# ---------------------------------------------------------

with st.expander("📋 CSV Format Help"):

    st.markdown("""
    ### Your CSV can have any column names.

    The application automatically looks for common text columns such as:

    `free_text`, `feedback`, `comment`, `review`, `response`,
    `text`, `answer`, `remarks`, `suggestion`, etc.

    **Example 1:**

    | feedback | rating |
    |---|---:|
    | Good service | 5 |
    | Very helpful staff | 4 |

    **Example 2:**

    | customer_name | comment | rating |
    |---|---|---:|
    | Rahul | Excellent experience | 5 |
    | Amit | Service was slow | 2 |

    The detected text column will automatically be converted internally
    to `free_text`.
    """)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("Controls")

    up = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    default_use_sample = (
        DEFAULT_CSV.exists() and
        (up is None)
    )

    use_sample = st.checkbox(
        "Use sample data",
        value=default_use_sample
    )

    k = st.slider(
        "Top keywords",
        5,
        40,
        15,
        1
    )

    score_range = st.slider(
        "Sentiment score range",
        -1.0,
        1.0,
        (-1.0, 1.0),
        step=0.01
    )

    search = st.text_input(
        "Search text",
        ""
    )

    page_size = st.selectbox(
        "Rows per page",
        [25, 50, 100, 200],
        index=1
    )


# ---------------------------------------------------------
# SELECT SOURCE
# ---------------------------------------------------------

source = None

if up is not None:

    source = up

elif use_sample:

    if DEFAULT_CSV.exists():

        source = DEFAULT_CSV

    else:

        st.warning(
            "Sample file is missing. Please upload a CSV."
        )

        st.stop()

else:

    st.info(
        "Upload a CSV or tick **Use sample data**."
    )

    st.stop()


# ---------------------------------------------------------
# ANALYZE CSV
# ---------------------------------------------------------

try:

    df_base = analyze_df(source)

except ValueError as e:

    st.error("❌ Invalid CSV Format")

    st.markdown(
        "### Please upload a CSV containing a survey/feedback text column."
    )

    st.write(
        "The application can automatically detect columns such as:"
    )

    st.code(
        "free_text\n"
        "feedback\n"
        "comment\n"
        "review\n"
        "response\n"
        "text\n"
        "answer\n"
        "remarks\n"
        "suggestion"
    )

    st.info(
        "Example: If your CSV contains a column named "
        "'feedback', it will automatically be used for sentiment analysis."
    )

    st.caption(f"Details: {e}")

    st.stop()


# ---------------------------------------------------------
# SUCCESS MESSAGE
# ---------------------------------------------------------

if up is not None:

    st.success(
        "✅ CSV uploaded and processed successfully!"
    )


# ---------------------------------------------------------
# OPTIONAL FILTERS
# ---------------------------------------------------------

with st.sidebar:

    segs = []

    if "segment" in df_base.columns:

        segs = st.multiselect(
            "Segments",
            sorted(
                df_base["segment"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    min_rating = None

    if "rating" in df_base.columns:

        min_rating = st.slider(
            "Min rating",
            0,
            5,
            0,
            1
        )


# ---------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------

df = apply_filters(
    df_base,
    segs,
    min_rating,
    score_range,
    search
)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

left, right = st.columns(2)


with left:

    st.subheader("Sentiment Share")

    share = (
        df["sentiment"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .rename("proportion")
        .reset_index()
    )

    share.columns = [
        "sentiment",
        "proportion"
    ]

    st.dataframe(
        share,
        use_container_width=True
    )


with right:

    st.subheader(
        "Descriptive Stats (Sentiment Score)"
    )

    st.dataframe(
        df["sentiment_score"]
        .describe()
        .to_frame(),
        use_container_width=True
    )


# ---------------------------------------------------------
# CHARTS
# ---------------------------------------------------------

st.subheader("Charts")

c1, c2 = st.columns(2)


with c1:

    st.markdown(
        "**Distribution of Sentiment**"
    )

    chart1 = (
        alt.Chart(share)
        .mark_bar()
        .encode(
            x=alt.X(
                "sentiment:N",
                title="Sentiment"
            ),
            y=alt.Y(
                "proportion:Q",
                title="Share (%)"
            ),
            tooltip=[
                "sentiment",
                "proportion"
            ]
        )
        .interactive()
    )

    st.altair_chart(
        chart1,
        use_container_width=True
    )


with c2:

    if "segment" in df.columns:

        st.markdown(
            "**Average Sentiment by Segment**"
        )

        seg_stats = (
            df.groupby("segment")["sentiment_score"]
            .mean()
            .reset_index()
            .sort_values(
                "sentiment_score",
                ascending=False
            )
        )

        chart2 = (
            alt.Chart(seg_stats)
            .mark_bar()
            .encode(
                x=alt.X(
                    "segment:N",
                    sort="-y",
                    title="Segment"
                ),
                y=alt.Y(
                    "sentiment_score:Q",
                    title="Average sentiment"
                ),
                tooltip=[
                    "segment",
                    alt.Tooltip(
                        "sentiment_score:Q",
                        format=".3f"
                    )
                ]
            )
            .interactive()
        )

        st.altair_chart(
            chart2,
            use_container_width=True
        )


# ---------------------------------------------------------
# KEYWORDS
# ---------------------------------------------------------

st.subheader(
    "Top Keywords (Current Filters)"
)

if len(df) == 0:

    st.warning(
        "No rows after filters. Loosen your filters."
    )

else:

    kw = top_keywords(
        df["free_text"],
        k=k
    )

    st.write(kw)


# ---------------------------------------------------------
# ROWS
# ---------------------------------------------------------

st.subheader("Rows")

if "page" not in st.session_state:

    st.session_state.page = 0


st.caption(
    f"{len(df)} rows match your filters."
)


paginated, total = paginate(
    df,
    page_size,
    st.session_state.page
)


st.dataframe(
    paginated,
    use_container_width=True
)


prev_col, next_col, reset_col = st.columns(3)


with prev_col:

    if st.button(
        "◀ Previous",
        disabled=(
            st.session_state.page == 0
        )
    ):

        st.session_state.page = max(
            0,
            st.session_state.page - 1
        )


with next_col:

    if st.button(
        "Next ▶",
        disabled=(
            (st.session_state.page + 1)
            * page_size >= total
        )
    ):

        st.session_state.page += 1


with reset_col:

    if st.button("Reset page"):

        st.session_state.page = 0


# ---------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------

st.subheader("Download")


st.download_button(
    "⬇️ Download Enriched CSV",
    df.to_csv(index=False),
    file_name="survey_enriched_filtered.csv",
    mime="text/csv"
)


neg_only = df[
    df["sentiment"] == "negative"
]


st.download_button(
    "⬇️ Download Negatives Only",
    neg_only.to_csv(index=False),
    file_name="survey_negatives.csv",
    mime="text/csv"
)