# UpdatedPMM - Custom Pure Market Making Strategy for Hummingbot

`UpdatedPMM` is a custom market-making strategy built on top of Hummingbot's PMM framework. It integrates dynamic pricing mechanisms, volatility-based spread adjustments, inventory risk management, and trend-protection logic using candle-based technical indicators.

---

## 🧠 Strategy Overview

- **Base Model:** Pure Market Making (PMM)
- **Volatility-aware:** Dynamically adjusts bid/ask spreads using the `NATR` (Normalized ATR) indicator.
- **Trend-sensitive:** Uses `RSI` and price change percentage to detect uptrends/downtrends and adapt order size and direction accordingly.
- **Inventory-aware:** Shifts prices based on base asset inventory imbalance using customizable skewing.
- **Failover logic:** Reverts to base spreads if indicator data is missing or errors occur.
- **Enhanced Logging:** Status output includes spreads, price shifts, inventory stats, and candle indicators.

---

## ⚙️ Features

- 📉 **Volatility Spread Adjustment**
  - Uses NATR to scale bid/ask spreads with recent market volatility.
- 📈 **RSI-based Trend Protection**
  - Detects uptrend/downtrend and adjusts:
    - Order amounts
    - Whether only one side (buy/sell) is placed
- 🔁 **Inventory Management**
  - Maintains healthy base/quote balance by dynamically shifting reference prices.
- 🧠 **Smart Fallbacks**
  - Defaults to base config values if indicator data isn't ready.

---

## 🛠️ Parameters Used

| Parameter                    | Description |
|-----------------------------|-------------|
| `use_volatility_spread`     | Enables spread adjustment using NATR |
| `use_trend_protection`      | Enables one-sided orders in trending markets |
| `trend_scalar`              | Scale for trend-based price shift |
| `inventory_scalar`          | Scale for price shift based on inventory imbalance |
| `max_shift_spread`          | Maximum allowed price shift in basis points |
| `order_amount`              | Base order size |
| `buy_skew_multiplier`       | Multiplier for buy orders during downtrends |
| `sell_skew_multiplier`      | Multiplier for sell orders during uptrends |
| `candle_length`             | Number of candles used for indicators |
| `candle_interval`           | Candle timeframe (e.g., 1m, 5m) |

---

## 📊 Sample Status Output

- Current spreads and adjustments
- Trend detection (direction, strength, threshold)
- Inventory skew and its impact
- Last 5 candles with RSI and NATR values

---

## 📂 File Location

Place this file in your `hummingbot/strategy/` directory.

---

## ✅ To Do

- [ ] Add backtesting support
- [ ] Add cooldown logic after large fills (optional)
- [ ] Tune strategy parameters for different pairs

---

## 📌 Notes

- This strategy is **experimental** and best used on testnet.
- Indicators rely on external candle data; ensure that your `candles_feed` is set up and working correctly.

---

## 🤝 Contributing

Feel free to fork this strategy or open a pull request with improvements or bug fixes.

---


