import pandas as pd


def to_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    '''Convert long format transactions to one-hot encoded basket matrix.

    Input:
        invoice_id, product

    Output:
        One row per invoice_id, one column per product with 0/1.
    '''
    basket = (
        df.assign(value=1)
        .pivot_table(
            index='invoice_id',
            columns='product',
            values='value',
            aggfunc='max',
            fill_value=0,
        )
    )

    basket = basket.astype(int)
    return basket
