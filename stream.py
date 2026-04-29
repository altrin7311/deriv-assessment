import websocket
import json
import numpy as np
import csv
import os
from datetime import datetime

APP_ID = "1089"
SYMBOL = "R_100"
prices = []

def compute_signal(prices):
    if len(prices) < 20:
        needed = 20 - len(prices)
        print(f"⏳ Collecting data... need {needed} more ticks")
        return

    arr = np.array(prices)

    short_ma = np.mean(arr[-5:])
    long_ma  = np.mean(arr[-20:])

    mean  = np.mean(arr[-20:])
    std   = np.std(arr[-20:])
    z_score = (arr[-1] - mean) / std if std > 0 else 0

    momentum = arr[-1] - arr[-5]

    score = 0
    reasons = []

    if short_ma > long_ma:
        score += 1
        reasons.append("MA crossover ↑")
    elif short_ma < long_ma:
        score -= 1
        reasons.append("MA crossover ↓")

    if z_score > 1.5:
        score -= 1
        reasons.append(f"Z-score HIGH ({z_score:.2f})")
    elif z_score < -1.5:
        score += 1
        reasons.append(f"Z-score LOW ({z_score:.2f})")

    if momentum > 0:
        score += 1
        reasons.append("Momentum +")
    else:
        score -= 1
        reasons.append("Momentum -")

    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = abs(score) / 3 * 100
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[signal]
    print("-" * 50)
    print(f"  SIGNAL     : {emoji} {signal}")
    print(f"  Confidence : {confidence:.0f}%")
    print(f"  Price      : {arr[-1]:.4f}")
    print(f"  Short MA   : {short_ma:.4f}  |  Long MA: {long_ma:.4f}")
    print(f"  Z-Score    : {z_score:.2f}  |  Momentum: {momentum:.4f}")
    print(f"  Reasons    : {', '.join(reasons)}")
    print(f"  Time (UTC) : {timestamp}")
    print("-" * 50)

    log_file = "signals_log.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp","price","signal","confidence","short_ma","long_ma","z_score","momentum","reasons"])
        writer.writerow([timestamp, arr[-1], signal, f"{confidence:.0f}%", f"{short_ma:.4f}", f"{long_ma:.4f}", f"{z_score:.2f}", f"{momentum:.4f}", " | ".join(reasons)])

def on_open(ws):
    print("✅ Connected to Deriv API — streaming R_100...")
    ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))

def on_message(ws, message):
    data = json.loads(message)
    if data.get("msg_type") == "tick":
        price = data["tick"]["quote"]
        prices.append(price)
        compute_signal(prices)

def on_error(ws, error):
    print(f"❌ Error: {error}")

def on_close(ws, *args):
    print("🔌 Disconnected")

ws = websocket.WebSocketApp(
    f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}",
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
)

ws.run_forever()