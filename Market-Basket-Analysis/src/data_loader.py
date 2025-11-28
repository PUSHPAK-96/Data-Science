import pandas as pd


def load_transactions(path: str) -> pd.DataFrame:
    '''Load transactional data.

    Expected long format:
        invoice_id, product
    One row per product per invoice.

    Args:
        path: Path to CSV file.

    Returns:
        DataFrame with at least two columns: 'invoice_id' and 'product'.
    '''
    df = pd.read_csv(path)
    
    df.columns = [c.strip().lower() for c in df.columns]

    invoice_col = None
    product_col = None

    for c in df.columns:
        if 'invoice' in c or 'bill' in c or 'order' in c or 'basket' in c or 'transaction' in c:
            invoice_col = c
        if 'product' in c or 'item' in c or 'sku' in c:
            product_col = c

    if invoice_col is None or product_col is None:
        raise ValueError(
            "Could not automatically detect invoice/product columns. "
            "Make sure your file has columns like 'invoice_id' and 'product'."
        )

    df = df[[invoice_col, product_col]].rename(
        columns={invoice_col: 'invoice_id', product_col: 'product'}
    )

    df['invoice_id'] = df['invoice_id'].astype(str)
    df['product'] = df['product'].astype(str).str.strip()

    df = df.dropna(subset=['invoice_id', 'product'])
    return df


def get_unique_stats(df: pd.DataFrame) -> dict:
    '''Basic stats for dashboard header.'''
    n_invoices = df['invoice_id'].nunique()
    n_products = df['product'].nunique()
    n_rows = len(df)
    return {
        'n_invoices': int(n_invoices),
        'n_products': int(n_products),
        'n_rows': int(n_rows),
    }
