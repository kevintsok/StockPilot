"""
CLI script: predict next trading day and print top-K stocks.
"""
from auto_select_stock.predict_pipeline import get_top_k_stocks, get_latest_price_date


def main():
    print(f"Latest price date in DB: {get_latest_price_date()}")
    print()

    results = get_top_k_stocks(
        checkpoint="models/price_transformer-train20220101-val20230101.pt",
        strategy="confidence",
        top_k=10,
    )

    print("Top 10 stocks to buy (Confidence-Sized):")
    print(f"{'Rank':<6}{'Symbol':<10}{'Pred Return':>12}{'Weight':>10}")
    print("-" * 40)
    for rank, (sym, pred_ret, weight) in enumerate(results, 1):
        print(f"{rank:<6}{sym:<10}{pred_ret:>12.4%}{weight:>10.4f}")


if __name__ == "__main__":
    main()
