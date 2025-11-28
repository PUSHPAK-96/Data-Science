import os
import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_transactions, get_unique_stats
from src.preprocessing import to_one_hot
from src.association_rules import (
    mine_frequent_itemsets,
    mine_association_rules,
    filter_rules,
)
from src.recommender import recommend_products
from src.visualization import top_n_products, build_rules_network, graph_to_plotly_figure


st.set_page_config(
    page_title="Market Basket Intelligence Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    #bg-video {
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        width: auto;
        height: auto;
        z-index: -2;
        object-fit: cover;
        filter: blur(2px) brightness(0.4);
    }
    .bg-overlay {
        position: fixed;
        left: 0;
        top: 0;
        width: 100vw;
        height: 100vh;
        background: radial-gradient(circle at top left, rgba(15,23,42,0.75), rgba(15,23,42,0.95));
        z-index: -1;
    }
    </style>

    <video autoplay muted loop id="bg-video">
      <source src="https://videos.pexels.com/video-files/856184/856184-hd_1280_720_25fps.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="bg-overlay"></div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
    /* ---------- Layout tweaks ---------- */
    .main {
        background: transparent;  /* let the video show through */
        color: #f9fafb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .block-container {
        padding-top: 4rem;   /* space below Streamlit top bar */
        padding-bottom: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* ---------- Keyframe animations ---------- */
    @keyframes fadeInDown {
        0%   { opacity: 0; transform: translateY(-16px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        0%   { opacity: 0; transform: translateY(16px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseGlow {
        0%   { box-shadow: 0 0 0px rgba(56,189,248,0.0); }
        50%  { box-shadow: 0 0 22px rgba(56,189,248,0.35); }
        100% { box-shadow: 0 0 0px rgba(56,189,248,0.0); }
    }
    @keyframes softFade {
        0%   { opacity: 0; transform: translateY(4px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* ---------- Hero card ---------- */
    .hero-card {
        border-radius: 1.4rem;
        padding: 1.4rem 1.6rem;
        background: radial-gradient(circle at top left, rgba(56,189,248,0.16), rgba(15,23,42,0.96));
        border: 1px solid rgba(148,163,184,0.45);
        box-shadow: 0 24px 60px rgba(15,23,42,0.9);
        animation: fadeInDown 0.7s ease-out;
    }
    .hero-title {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 0.03em;
        color: #e5e7eb;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 0.85rem;
        color: rgba(148,163,184,0.96);
        max-width: 600px;
    }
    .hero-badges {
        margin-top: 0.6rem;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        padding: 0.18rem 0.7rem;
        border-radius: 999px;
        border: 1px solid rgba(94,234,212,0.55);
        font-size: 0.7rem;
        color: #a5f3fc;
        margin-right: 0.35rem;
        background: rgba(8,47,73,0.85);
        animation: softFade 0.6s ease-out;
    }
    .pill-secondary {
        border-color: rgba(129,140,248,0.7);
        color: #c7d2fe;
        background: rgba(30,64,175,0.55);
    }

    /* ---------- KPI cards ---------- */
    .metric-card {
        padding: 0.9rem 1.1rem;
        border-radius: 1.1rem;
        border: 1px solid rgba(148,163,184,0.35);
        background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(30,64,175,0.55));
        backdrop-filter: blur(10px);
        transition: transform 0.22s ease-out, box-shadow 0.22s ease-out, border-color 0.22s;
        animation: fadeInUp 0.7s ease-out;
    }
    .metric-card:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 18px 40px rgba(15,23,42,0.9);
        border-color: rgba(248,250,252,0.45);
    }

    /* Special subtle glow for Mining Snapshot card */
    .metric-card:nth-child(4) {
        animation: fadeInUp 0.7s ease-out, pulseGlow 2.4s ease-in-out 0.8s infinite;
    }

    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: rgba(148,163,184,0.95);
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #e5e7eb;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: rgba(148,163,184,0.9);
        margin-top: 0.15rem;
    }

    /* ---------- Chips / tags ---------- */
    .tag {
        display: inline-flex;
        align-items: center;
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.5);
        font-size: 0.7rem;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
        background: rgba(15,23,42,0.86);
        animation: softFade 0.4s ease-out;
    }

    /* ---------- Section titles ---------- */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }
    .section-subtitle {
        font-size: 0.78rem;
        color: rgba(148,163,184,0.95);
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



def load_data_and_params():
    with st.sidebar:
        st.markdown("### 🧱 Data & Mining Setup")

        with st.expander("Dataset", expanded=True):
            source_option = st.radio(
                "Data source",
                ["Use sample data", "Upload CSV"],
                help="Use the built-in demo dataset or upload your own transactional data.",
            )

            if source_option == "Upload CSV":
                file = st.file_uploader(
                    "Upload CSV (invoice_id, product)",
                    type=["csv"],
                    help="File should have at least invoice and product columns.",
                )
                if file is not None:
                    try:
                        df = load_transactions(file)
                    except ValueError:
                            st.markdown(
                                """
                                <div style="
                                    background: rgba(255,0,0,0.2);
                                    padding: 30px;
                                    border-radius: 14px;
                                    border: 2px solid red;
                                    text-align: center;
                                    font-size: 22px;
                                    font-weight: bold;
                                    color: white;
                                ">
                                ❌ WRONG FORMAT CSV UPLOADED!<br>
                                Please upload file with columns:<br>
                                <span style="color:yellow">invoice_id, product</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            st.stop()

                else:
                    df = None
            else:
                 base_dir = os.path.dirname(os.path.abspath(__file__))
                 sample_path = os.path.join(base_dir, "data", "sample_transactions.csv")
                 df = load_transactions(sample_path)

        st.markdown("---")
        st.markdown("#### 🎛 Mining Presets")

        preset = st.radio(
            "Choose preset",
            ["Exploration (loose)", "Balanced (default)", "Strict (high precision)"],
            index=1,
        )


        if preset == "Exploration (loose)":
            default_support = 0.02
            default_conf = 0.25
            default_lift = 1.05
            default_max_len = 4
        elif preset == "Strict (high precision)":
            default_support = 0.08
            default_conf = 0.6
            default_lift = 1.5
            default_max_len = 3
        else:  
            default_support = 0.05
            default_conf = 0.4
            default_lift = 1.2
            default_max_len = 3

        st.caption("You can still fine-tune sliders after selecting a preset.")

        col_a, col_b = st.columns(2)
        with col_a:
            min_support = st.slider(
                "Min support",
                min_value=0.01,
                max_value=0.5,
                value=default_support,
                step=0.01,
            )
            min_confidence = st.slider(
                "Min confidence",
                min_value=0.1,
                max_value=1.0,
                value=default_conf,
                step=0.05,
            )
        with col_b:
            min_lift = st.slider(
                "Min lift",
                min_value=1.0,
                max_value=10.0,
                value=default_lift,
                step=0.1,
            )
            max_len = st.slider(
                "Max items in itemset",
                min_value=2,
                max_value=5,
                value=default_max_len,
                step=1,
            )

        top_rules_to_show = st.slider(
            "Top rules to display",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
        )

        st.markdown("---")
        with st.expander("💡 Interpreting metrics", expanded=False):
            st.markdown(
                """
                - **Support** → how frequent a pattern appears  
                - **Confidence** → reliability of the rule  
                - **Lift > 1** → positive association between products  
                """
            )

    return df, min_support, min_confidence, min_lift, max_len, top_rules_to_show



def overview_visuals(df: pd.DataFrame):
    st.markdown("<div class='section-title'>📦 Product Landscape</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Understand which products dominate baskets before diving into rules.</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1.5, 1.2, 1.4])

    
    with col1:
        top_products = df["product"].value_counts().reset_index()
        top_products.columns = ["product", "count"]
        top_products = top_products.head(15)

        if not top_products.empty:
            fig = px.bar(
                top_products,
                x="product",
                y="count",
                title="Top products by frequency",
            )
            fig.update_layout(
                xaxis_title="Product",
                yaxis_title="Count",
                title_x=0.05,
                height=360,
                margin=dict(l=10, r=10, t=40, b=80),
            )
            fig.update_xaxes(tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No products to display.")

    
    with col2:
        if not df.empty:
            share = df["product"].value_counts().reset_index()
            share.columns = ["product", "count"]
            share = share.head(8)
            fig_donut = px.pie(
                share,
                names="product",
                values="count",
                hole=0.55,
                title="Share of top products",
            )
            fig_donut.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=40, b=40),
                title_x=0.1,
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No data for donut chart.")

    
    with col3:
        st.markdown("**Sample Transactions**")
        st.caption("Quick peek into raw transactional rows.")
        st.dataframe(df.head(20), use_container_width=True, height=340)



def rules_scatter_and_lift(rules_filtered: pd.DataFrame):
    st.markdown("<div class='section-title'>📌 Rule Quality Landscape</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Explore support, confidence and lift to balance recall vs precision.</div>",
        unsafe_allow_html=True,
    )

    if rules_filtered.empty:
        st.info("No rules to visualise. Try relaxing support/confidence/lift thresholds.")
        return

    col1, col2 = st.columns([1.5, 1.0])

    df_plot = rules_filtered.copy().head(400)
    df_plot["rule"] = df_plot["antecedents_str"] + " → " + df_plot["consequents_str"]

    with col1:
        fig = px.scatter(
            df_plot,
            x="support",
            y="confidence",
            size="lift",
            color="lift",
            color_continuous_scale="Turbo",
            hover_name="rule",
            hover_data=["lift"],
            title="Support vs Confidence (bubble = Lift)",
        )
        fig.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=40, b=40),
            title_x=0.1,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig_hist = px.histogram(
            df_plot,
            x="lift",
            nbins=30,
            title="Distribution of Lift",
        )
        fig_hist.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=40, b=40),
            title_x=0.2,
            xaxis_title="Lift",
            yaxis_title="Rule count",
        )
        st.plotly_chart(fig_hist, use_container_width=True)



def rules_product_filter_and_view(rules_filtered: pd.DataFrame):
    all_products = sorted(
        set(sum([a.split(", ") for a in rules_filtered["antecedents_str"]], []))
        | set(sum([c.split(", ") for c in rules_filtered["consequents_str"]], []))
    )

    col1, col2 = st.columns([1.6, 1.2])
    with col1:
        focus_product = st.selectbox(
            "Filter rules by focus product (appears in antecedent or consequent)",
            options=["(All products)"] + all_products,
        )
    with col2:
        sort_choice = st.selectbox(
            "Sort rules by",
            options=["lift", "confidence", "support"],
            index=0,
        )

    df_view = rules_filtered.copy()
    if focus_product != "(All products)":
        mask = df_view["antecedents_str"].str.contains(focus_product) | df_view[
            "consequents_str"
        ].str.contains(focus_product)
        df_view = df_view[mask]

    df_view = df_view.sort_values(sort_choice, ascending=False)

    
    with st.expander("📊 Consequent product frequency in filtered rules", expanded=False):
        if df_view.empty:
            st.info("No rules for this filter.")
        else:
            cons = (
                df_view["consequents_str"]
                .str.get_dummies(sep=", ")
                .sum()
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
            )
            cons.columns = ["product", "rule_count"]
            fig_c = px.bar(
                cons,
                x="product",
                y="rule_count",
                title="Products most often suggested as consequents",
            )
            fig_c.update_layout(
                xaxis_title="Product",
                yaxis_title="Rule count",
                height=360,
                margin=dict(l=10, r=10, t=40, b=80),
            )
            fig_c.update_xaxes(tickangle=-35)
            st.plotly_chart(fig_c, use_container_width=True)

    return df_view



def main():
    df, min_support, min_confidence, min_lift, max_len, top_rules_to_show = load_data_and_params()


    st.markdown(
        """
        <div class="hero-card">
          <div class="hero-title">🛒 Market Basket Intelligence Studio</div>
          <div class="hero-subtitle">
            Visual-first environment to mine frequent itemsets, explore association rules, and design
            upsell / cross-sell strategies that actually move revenue.
          </div>
          <div class="hero-badges">
            <span class="pill">Apriori · Association Rules</span>
            <span class="pill pill-secondary">Upsell & Cross-sell Design</span>
            <span class="pill">Support · Confidence · Lift</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    if df is None or df.empty:
        st.warning("Load a CSV from the sidebar to get started.")
        return

    
    stats = get_unique_stats(df)

    
    basket = to_one_hot(df)
    frequent_itemsets = mine_frequent_itemsets(
        basket, min_support=min_support, max_len=max_len
    )
    rules_raw = mine_association_rules(
        frequent_itemsets, metric="lift", min_threshold=min_lift
    )
    rules_filtered = filter_rules(
        rules_raw, min_confidence=min_confidence, min_lift=min_lift
    )

    n_itemsets = len(frequent_itemsets)
    n_rules_total = len(rules_raw)
    n_rules_filtered = len(rules_filtered)

    top_items = top_n_products(df, n=5)

    col1, col2, col3, col4 = st.columns([1.1, 1.1, 1.1, 2.4])
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Invoices</div>
              <div class="metric-value">{stats['n_invoices']}</div>
              <div class="metric-sub">Unique baskets in dataset</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Unique Products</div>
              <div class="metric-value">{stats['n_products']}</div>
              <div class="metric-sub">SKU-level granularity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Rows</div>
              <div class="metric-value">{stats['n_rows']}</div>
              <div class="metric-sub">Total line items</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        
        chips_html = ""
        if top_items is not None and not top_items.empty:
            for product, count in top_items.items():
                chips_html += f"<span class='tag'>{product} · {count}</span>"

        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Mining Snapshot</div>
              <div class="metric-sub">
                Itemsets: <b>{n_itemsets}</b> · Rules (all): <b>{n_rules_total}</b> · Rules (filtered): <b>{n_rules_filtered}</b>
              </div>
              <div style="margin-top:0.4rem;">
                {chips_html}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📈 Overview",
            "📜 Rules Explorer",
            "📌 Rule Metrics",
            "🕸️ Network Graph",
            "🎯 Basket Recommender",
        ]
    )


    with tab1:
        overview_visuals(df)

        st.markdown("### 🔍 Frequent Itemsets")
        if frequent_itemsets.empty:
            st.info("No frequent itemsets found with the current thresholds.")
        else:
            fi_show = frequent_itemsets.copy()
            fi_show["itemsets"] = fi_show["itemsets"].apply(
                lambda x: ", ".join(list(x))
            )
            st.dataframe(fi_show.head(40), use_container_width=True)

    
    with tab2:
        st.markdown("<div class='section-title'>📜 Association Rules Explorer</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Find the rules that can power 'Frequently bought together' widgets and checkout nudges.</div>",
            unsafe_allow_html=True,
        )

        if rules_filtered.empty:
            st.info(
                "No rules passed the filters. Try lowering the minimum support/confidence/lift."
            )
        else:
            df_view = rules_product_filter_and_view(rules_filtered)
            st.dataframe(
                df_view.head(top_rules_to_show),
                use_container_width=True,
                height=480,
            )

            csv_data = df_view.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download filtered rules as CSV",
                csv_data,
                file_name="association_rules_filtered.csv",
                mime="text/csv",
            )

    
    with tab3:
        rules_scatter_and_lift(rules_filtered)

    
    with tab4:
        st.markdown("<div class='section-title'>🕸️ Product Affinity Network</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Nodes are products; arrows show 'if bought → then likely to buy'. Great for layout & bundling ideas.</div>",
            unsafe_allow_html=True,
        )

        if rules_filtered.empty:
            st.info("No rules to visualise.")
        else:
            G = build_rules_network(rules_raw, top_k=40)
            fig_net = graph_to_plotly_figure(G)
            if fig_net.data:
                st.plotly_chart(fig_net, use_container_width=True)
            else:
                st.info("Network could not be built. Try relaxing thresholds.")

    
    with tab5:
        st.markdown("<div class='section-title'>🎯 Interactive Basket Recommender</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Select items in a live basket and get ranked add-on suggestions based on learned rules.</div>",
            unsafe_allow_html=True,
        )

        if rules_raw.empty or rules_filtered.empty:
            st.info(
                "No rules available for recommendation. Adjust the mining parameters."
            )
        else:
            products_sorted = sorted(df["product"].unique())
            preselected = products_sorted[:3] if len(products_sorted) > 0 else []

            selected_items = st.multiselect(
                "Select items currently in the customer's basket",
                options=products_sorted,
                default=preselected,
            )

            if st.button("Generate recommendations", type="primary"):
                if not selected_items:
                    st.warning("Please select at least one product.")
                else:
                    recs = recommend_products(rules_raw, selected_items, top_n=10)
                    if recs.empty:
                        st.info(
                            "No strong recommendations for this combination. "
                            "Try adding/removing items or relaxing thresholds."
                        )
                    else:
                        st.write("**Recommended add-ons for upsell / cross-sell:**")
                        st.dataframe(recs, use_container_width=True)

                        st.caption(
                            "Tip: Plug these into 'Frequently bought together', "
                            "cart add-ons, or personalised email recommendations."
                        )

    with st.expander("ℹ️ About this app"):
        st.markdown(
            """
            **Market Basket Analysis for Upselling/Cross-Selling**  
            Powered by Apriori and association rule mining. Use this studio to:
            - Discover natural product bundles  
            - Design upsell & cross-sell offers  
            - Prioritise products with high lift and confidence  
            """
        )


if __name__ == "__main__":
    main()
