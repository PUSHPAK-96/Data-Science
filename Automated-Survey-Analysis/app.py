import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

from src.prep import load_csv
from src.sentiment import sentiment_scores, sentiment_label
from src.keywords import top_keywords


st.set_page_config(page_title="Automated Survey Analysis", page_icon="📝", layout="wide")


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = APP_DIR / "data" / "sample_surveys.csv"   


@st.cache_data(show_spinner=False)
def analyze_df(file_or_path):
    df = load_csv(file_or_path)  
    df["sentiment_score"] = sentiment_scores(df["free_text"])
    df["sentiment"] = df["sentiment_score"].map(sentiment_label)
    
    if "rating" in df.columns:
        with pd.option_context("mode.use_inf_as_na", True):
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df


def apply_filters(df, segs, min_rating, score_range, search):
    out = df.copy()
    if segs and "segment" in out.columns:
        out = out[out["segment"].isin(segs)]
    if min_rating is not None and "rating" in out.columns:
        out = out[out["rating"].fillna(-1) >= min_rating]
    out = out[(out["sentiment_score"] >= score_range[0]) & (out["sentiment_score"] <= score_range[1])]
    if search.strip():
        s = search.strip().lower()
        out = out[out["free_text"].str.contains(s, case=False, na=False)]
    return out


def paginate(df, page_size, page):
    start = page * page_size
    end = start + page_size
    return df.iloc[start:end], len(df)


st.title("📝 Automated Survey Analysis")

with st.expander("CSV format help"):
    st.markdown("The CSV must include a **free_text** column. Optional columns like **rating**, **segment** are fine.")


with st.sidebar:
    st.header("Controls")
    up = st.file_uploader("Upload CSV", type=["csv"])
    
    default_use_sample = DEFAULT_CSV.exists() and (up is None)
    use_sample = st.checkbox("Use sample data", value=default_use_sample)
    k = st.slider("Top keywords", 5, 40, 15, 1)
    score_range = st.slider("Sentiment score range", -1.0, 1.0, (-1.0, 1.0), step=0.01)
    search = st.text_input("Search text", "")
    page_size = st.selectbox("Rows per page", [25, 50, 100, 200], index=1)


source = None
if up is not None:
    
    source = up
elif use_sample:
    
    if DEFAULT_CSV.exists():
        source = DEFAULT_CSV
    else:
        st.warning("Sample file is missing in the deployed app. Please upload a CSV.")
        st.stop()
else:
    
    st.info("Upload a CSV or tick **Use sample data**.")
    st.stop()


df_base = analyze_df(source)


with st.sidebar:
    segs = []
    if "segment" in df_base.columns:
        segs = st.multiselect("Segments", sorted(df_base["segment"].dropna().unique().tolist()))
    min_rating = None
    if "rating" in df_base.columns:
        min_rating = st.slider("Min rating", 0, 5, 0, 1)

df = apply_filters(df_base, segs, min_rating, score_range, search)


left, right = st.columns(2)
with left:
    st.subheader("Sentiment share")
    share = (
        df["sentiment"]
        .value_counts(normalize=True)
        .mul(100).round(1)
        .rename("proportion").reset_index()
        .rename(columns={"index": "sentiment"})
    )
    st.dataframe(share)

with right:
    st.subheader("Descriptive stats (sentiment score)")
    st.dataframe(df["sentiment_score"].describe().to_frame())


st.subheader("Charts")
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Distribution of sentiment**")
    chart1 = (
        alt.Chart(share)
        .mark_bar()
        .encode(
            x=alt.X("sentiment:N", title="Sentiment"),
            y=alt.Y("proportion:Q", title="Share (%)"),
            tooltip=["sentiment", "proportion"]
        )
        .interactive()
    )
    st.altair_chart(chart1, use_container_width=True)

with c2:
    if "segment" in df.columns:
        st.markdown("**Avg sentiment by segment**")
        seg_stats = (
            df.groupby("segment")["sentiment_score"]
              .mean().reset_index().sort_values("sentiment_score", ascending=False)
        )
        chart2 = (
            alt.Chart(seg_stats)
            .mark_bar()
            .encode(
                x=alt.X("segment:N", sort="-y", title="Segment"),
                y=alt.Y("sentiment_score:Q", title="Average sentiment"),
                tooltip=["segment", alt.Tooltip("sentiment_score:Q", format=".3f")]
            )
            .interactive()
        )
        st.altair_chart(chart2, use_container_width=True)


st.subheader("Top keywords (on current filters)")
if len(df) == 0:
    st.warning("No rows after filters. Loosen your filters.")
else:
    kw = top_keywords(df["free_text"], k=k)
    st.write(kw)


st.subheader("Rows")
if "page" not in st.session_state:
    st.session_state.page = 0

st.caption(f"{len(df)} rows match your filters.")
paginated, total = paginate(df, page_size, st.session_state.page)
st.dataframe(paginated)

prev_col, next_col, reset_col = st.columns(3)
with prev_col:
    if st.button("◀ Previous", disabled=(st.session_state.page == 0)):
        st.session_state.page = max(0, st.session_state.page - 1)
with next_col:
    if st.button("Next ▶", disabled=((st.session_state.page + 1) * page_size >= total)):
        st.session_state.page += 1
with reset_col:
    if st.button("Reset page"):
        st.session_state.page = 0


st.subheader("Download")
st.download_button(
    "⬇️ Download enriched CSV (current filters)",
    df.to_csv(index=False),
    file_name="survey_enriched_filtered.csv",
)

neg_only = df[df["sentiment"] == "negative"]
st.download_button(
    "⬇️ Download negatives only",
    neg_only.to_csv(index=False),
    file_name="survey_negatives.csv",
)