# Data preparation

The original six-asset market-data snapshot has not been recovered. This folder contains documentation only. No data license is granted by this repository's software license.

## Local CSVs

Supply `SPY.csv`, `AAPL.csv`, `MSFT.csv`, `GOOGL.csv`, `AMZN.csv`, and `BTC_USD.csv` here. Each file must contain ascending, unique daily dates and numeric columns:

```csv
Date,Open,High,Low,Close,Volume
```

The runner checks positive prices, nonnegative volume, finite values, and OHLC ordering. It respects `start` inclusive and `end` exclusive. It never fills missing market bars. Exchange trading calendars differ from Bitcoin's daily calendar.

## Adjustment and source policy

The published article describes equity data from Yahoo Finance, SPY from Nasdaq, and BTC-USD from Yahoo Finance/Binance, with no dividend adjustment. Retain source, download date, currency, calendar, timezone, adjustment policy, and SHA-256 for each input. Check historical stock splits before using local files: unadjusted split jumps can dominate the target.

`python run_reproduction.py --download` is an **optional new-data run**. It fetches all six tickers through yfinance using `auto_adjust=False`; this selects the vendor's `Close`, not its dividend-adjusted `Adj Close`. Yahoo can still apply split adjustments. This convenience source differs from the article's multi-vendor inputs and cannot recover its original snapshot. Changing a vendor or adjustment policy can change results.

Local CSVs and generated outputs are ignored by Git. `run_metadata.json` stores the input file hashes, effective date ranges, counts, configuration, and installed package versions. Obtain permission before redistributing third-party data.
