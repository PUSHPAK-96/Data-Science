# Market Basket Analysis – Upsell & Cross-Sell Studio (Streamlit)

This project demonstrates Market Basket Analysis using association rule mining (Apriori)
and an interactive Streamlit dashboard for upselling and cross-selling use cases.

## Features

- Upload your own transactional CSV (invoice_id, product) or use sample data
- Mine frequent itemsets and association rules (support, confidence, lift)
- Explore rules in interactive tables
- Visualize product affinity network
- Interactive basket recommender for add-on suggestions

## Project Structure

- `app.py` – Streamlit app (main UI)
- `data/sample_transactions.csv` – Example dataset
- `src/data_loader.py` – Load & clean input data
- `src/preprocessing.py` – Transform to basket one-hot format
- `src/association_rules.py` – Frequent itemsets & association rules
- `src/recommender.py` – Simple recommendation engine based on rules
- `src/visualization.py` – Helpers for top products & network graph
- `requirements.txt` – Python dependencies

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```


Upload your own CSV or start with the sample dataset and play with the parameters
(min support, min confidence, min lift, max rule length) from the sidebar.
