import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def mine_frequent_itemsets(
    basket: pd.DataFrame,
    min_support: float = 0.02,
    max_len: int | None = None,
) -> pd.DataFrame:
    '''Run Apriori to get frequent itemsets.'''
    frequent = apriori(
        basket,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )
    frequent = frequent.sort_values('support', ascending=False)
    return frequent


def mine_association_rules(
    frequent_itemsets: pd.DataFrame,
    metric: str = 'lift',
    min_threshold: float = 1.0,
) -> pd.DataFrame:
    '''Derive association rules from frequent itemsets.'''
    rules = association_rules(
        frequent_itemsets,
        metric=metric,
        min_threshold=min_threshold,
    )

    # Sort for nicer display
    rules = rules.sort_values([metric, 'confidence', 'support'], ascending=False)
    return rules


def filter_rules(
    rules: pd.DataFrame,
    min_confidence: float = 0.3,
    min_lift: float = 1.0,
    sort_by: str = 'lift',
) -> pd.DataFrame:
    '''Filter rules by confidence and lift and sort for display.'''
    filtered = rules[
        (rules['confidence'] >= min_confidence) & (rules['lift'] >= min_lift)
    ].copy()

    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=False)

    
    filtered['antecedents_str'] = filtered['antecedents'].apply(
        lambda x: ', '.join(list(x))
    )
    filtered['consequents_str'] = filtered['consequents'].apply(
        lambda x: ', '.join(list(x))
    )

    
    cols = [
        'antecedents_str',
        'consequents_str',
        'support',
        'confidence',
        'lift',
        'leverage',
        'conviction',
    ]
    existing_cols = [c for c in cols if c in filtered.columns]
    rest = [c for c in filtered.columns if c not in existing_cols]
    filtered = filtered[existing_cols + rest]
    return filtered
