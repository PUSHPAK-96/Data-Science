import pandas as pd


def recommend_products(
    rules: pd.DataFrame,
    basket_items: list[str],
    top_n: int = 5,
) -> pd.DataFrame:
    '''Given a list of items already in the basket, recommend additional products
    using association rules.

    We look for rules where antecedents are subset of basket_items.
    '''
    if rules.empty:
        return pd.DataFrame(columns=['product', 'score', 'support', 'confidence', 'lift'])

    basket_set = set(basket_items)

    def antecedent_match(row) -> bool:
        return set(row['antecedents']).issubset(basket_set)

    candidate_rules = rules[rules.apply(antecedent_match, axis=1)].copy()
    if candidate_rules.empty:
        return pd.DataFrame(columns=['product', 'score', 'support', 'confidence', 'lift'])

    
    exploded = candidate_rules.explode('consequents')

    
    exploded = exploded[~exploded['consequents'].isin(basket_set)]

    if exploded.empty:
        return pd.DataFrame(columns=['product', 'score', 'support', 'confidence', 'lift'])

    
    exploded['score'] = exploded['confidence'] * exploded['lift']

    grouped = (
        exploded.groupby('consequents')
        .agg(
            score=('score', 'sum'),
            support=('support', 'max'),
            confidence=('confidence', 'max'),
            lift=('lift', 'max'),
        )
        .reset_index()
        .rename(columns={'consequents': 'product'})
        .sort_values('score', ascending=False)
    )

    return grouped.head(top_n)
