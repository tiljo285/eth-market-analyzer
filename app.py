import os
import zipfile
import tempfile
import sqlite3
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & MODERN UI STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ETH Market Edge Finder",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px 20px; margin-bottom: 12px; }
    .metric-title { font-size: 0.85rem; color: #8b949e; margin-bottom: 6px; font-weight: 500; text-transform: uppercase; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #f0f6fc; }
    .metric-sub { font-size: 0.8rem; margin-top: 4px; }
    .takeaway-box { background: linear-gradient(135deg, rgba(56, 139, 253, 0.08) 0%, rgba(31, 111, 235, 0.03) 100%); border-left: 4px solid #388bfd; border-radius: 4px; padding: 14px 18px; margin: 14px 0; font-size: 0.95rem; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-EXTRACT DATABASE
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists("delta_data.db") and os.path.exists("delta_data.zip"):
    with st.spinner("Extracting database from zip..."):
        with zipfile.ZipFile("delta_data.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR & CLOUD FILE UPLOADER
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Database Upload")

uploaded_db = st.sidebar.file_uploader("Upload delta_data.db", type=["db", "sqlite"])

if uploaded_db is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
        tmp.write(uploaded_db.getvalue())
        DB_PATH = tmp.name
else:
    DB_PATH = "delta_data.db"
    if not os.path.exists(DB_PATH):
        st.sidebar.warning("👈 Please upload your delta_data.db file to begin.")
        st.title("⚡ Market Edge & Behavioral Playbook")
        st.info("Awaiting database upload. Use the sidebar to upload your Delta Exchange data.")
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# DATA CACHING & MATRIX CALCULATION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_calculate_matrix(symbol, resolution, start_ts, end_ts, tz, db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """SELECT time, open, high, low, close, volume FROM candles
           WHERE symbol=? AND resolution=? AND time>=? AND time<=?
           ORDER BY time ASC""",
        conn, params=(symbol, resolution, start_ts, end_ts))
    conn.close()
    if df.empty:
        return df, None

    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert(tz)
    df.set_index('datetime', inplace=True)
    df['trading_day'] = df.index.tz_convert('UTC').date
    df['hour'] = df.index.hour
    df['year'] = df.index.year

    bar_sec = (df.index[1] - df.index[0]).total_seconds()
    bars_per_hour = max(1, int(round(3600 / bar_sec)))
    
    horizons = {
        '15m': max(1, int(round(15 * 60 / bar_sec))),
        '1h':  bars_per_hour * 1,
        '2h':  bars_per_hour * 2,
        '4h':  bars_per_hour * 4,
        '8h':  bars_per_hour * 8,
        '12h': bars_per_hour * 12,
        '24h': bars_per_hour * 24
    }

    close_vals = df['close'].values
    n = len(df)

    for h_name, h_bars in horizons.items():
        if h_bars < n:
            ret = np.full(n, np.nan)
            ret[:-h_bars] = (close_vals[h_bars:] - close_vals[:-h_bars]) / close_vals[:-h_bars]
            df[f'ret_{h_name}'] = ret

    daily = df.groupby('trading_day').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    })
    daily['pdh'] = daily['high'].shift(1)
    daily['pdl'] = daily['low'].shift(1)
    daily['prev_bullish'] = daily['close'].shift(1) > daily['open'].shift(1)

    df = df.join(daily[['pdh', 'pdl', 'prev_bullish']], on='trading_day')
    return df, horizons

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION INPUTS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Market Selection")

conn = sqlite3.connect(DB_PATH)
try:
    symbols = pd.read_sql_query("SELECT DISTINCT symbol FROM candles", conn)['symbol'].tolist()
    resolutions = pd.read_sql_query("SELECT DISTINCT resolution FROM candles", conn)['resolution'].tolist()
except Exception:
    symbols, resolutions = ['ETHUSDT'], ['15m']
conn.close()

symbol = st.sidebar.selectbox("Symbol", symbols, index=0)
resolution = st.sidebar.selectbox("Bar Resolution", resolutions, index=0)
tz = st.sidebar.selectbox("Timezone", ['Asia/Kolkata', 'UTC', 'America/New_York'], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Period of Study")
c1, c2 = st.sidebar.columns(2)
start_date = c1.date_input("Start", value=datetime(2024, 1, 1))
end_date   = c2.date_input("End", value=datetime(2025, 1, 1))

start_ts = int(datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc).timestamp())
end_ts   = int(datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc).timestamp())

with st.spinner("Analyzing market patterns..."):
    df, horizons = load_and_calculate_matrix(symbol, resolution, start_ts, end_ts, tz, DB_PATH)

if df.empty or horizons is None:
    st.error("No data found for this selection. Adjust your date range.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER & OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
st.title("⚡ Market Edge & Behavioral Playbook")
st.caption(f"Asset: **{symbol}** · Timeframe: **{resolution}** · Daily Rollover: **05:30 AM IST (00:00 UTC)** · Samples: **{len(df):,} bars**")

base_4h_up = (df['ret_4h'].dropna() > 0).mean() * 100
base_4h_mean = df['ret_4h'].dropna().mean() * 100

events_dict = {
    "Yesterday's Low Swept (PDL Sweep)": (df['low'] < df['pdl']) & (df['low'].shift(1) >= df['pdl'].shift(1)),
    "Yesterday's High Swept (PDH Sweep)": (df['high'] > df['pdh']) & (df['high'].shift(1) <= df['pdh'].shift(1)),
    "PDL Swept + Yesterday Was Green": ((df['low'] < df['pdl']) & (df['low'].shift(1) >= df['pdl'].shift(1))) & (df['prev_bullish'] == True),
    "PDL Swept + Yesterday Was Red": ((df['low'] < df['pdl']) & (df['low'].shift(1) >= df['pdl'].shift(1))) & (df['prev_bullish'] == False),
    "PDH Swept + Yesterday Was Green": ((df['high'] > df['pdh']) & (df['high'].shift(1) <= df['pdh'].shift(1))) & (df['prev_bullish'] == True),
    "PDH Swept + Yesterday Was Red": ((df['high'] > df['pdh']) & (df['high'].shift(1) <= df['pdh'].shift(1))) & (df['prev_bullish'] == False),
    "Sudden Green Spike (+0.75% 1-bar)": (df['close'] - df['open']) / df['open'] >= 0.0075,
    "Sudden Red Dump (-0.75% 1-bar)": (df['close'] - df['open']) / df['open'] <= -0.0075,
}

tab_playbook, tab_timeline, tab_timing, tab_punchcard = st.tabs([
    "📋 The Edge Playbook",
    "⏱️ Event Timeline",
    "🗺️ The Daily Routine",
    "🎯 Top & Bottom Miner"
])

# ─── TAB 1: THE EDGE PLAYBOOK ───
with tab_playbook:
    st.subheader("Discovered Market Behaviors")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Normal Market Drift (4h)</div><div class="metric-value">{base_4h_up:.1f}% Up</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Average 4h Move</div><div class="metric-value">{base_4h_mean:+.2f}%</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Rule for an "Edge"</div><div class="metric-value">±5.0% Shift</div></div>', unsafe_allow_html=True)

    records = []
    for name, mask in events_dict.items():
        sample = df[mask & df['ret_4h'].notna()]['ret_4h']
        n = len(sample)
        if n < 10: continue
        up_rate = (sample > 0).mean() * 100
        shift = up_rate - base_4h_up
        avg_move = sample.mean() * 100
        verdict = "🟢 Strong Bullish Edge" if shift >= 5.0 else "🔴 Strong Bearish Edge" if shift <= -5.0 else "⚪ Random Noise / No Edge"
        records.append({"Market Event": name, "Occurrences": n, "Next 4h Up %": f"{up_rate:.1f}%", "Edge vs Normal": f"{shift:+.1f}%", "Avg 4h Return": f"{avg_move:+.2f}%", "Verdict": verdict})

    st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

# ─── TAB 2: TIMELINE MICROSCOPE ───
with tab_timeline:
    st.subheader("Event Timeline: What Happens Next?")
    selected_event = st.selectbox("Select a Market Event to inspect:", list(events_dict.keys()), index=0)
    event_mask = events_dict[selected_event]

    timeline_data = []
    for h_name in horizons.keys():
        col = f'ret_{h_name}'
        base_rate = (df[col].dropna() > 0).mean() * 100
        subset = df[event_mask & df[col].notna()][col]
        if len(subset) > 0:
            event_rate = (subset > 0).mean() * 100
            timeline_data.append({"Holding Time": h_name, "After Event (Up %)": round(event_rate, 1), "Normal Market (Up %)": round(base_rate, 1), "Edge Boost": round(event_rate - base_rate, 1), "Avg Move %": round(subset.mean() * 100, 2)})

    tdf = pd.DataFrame(timeline_data)
    col_chart, col_stats = st.columns([7, 5])
    with col_chart:
        fig_time = go.Figure(go.Bar(x=tdf['Holding Time'], y=tdf['Edge Boost'], marker_color=['#3fb950' if x > 0 else '#f85149' for x in tdf['Edge Boost']], text=tdf['Edge Boost'].apply(lambda v: f"{v:+.1f}%"), textposition='auto'))
        fig_time.add_hline(y=0, line_color="gray", line_width=1)
        fig_time.update_layout(title=f"Edge Shift Over Time", yaxis_title="Advantage vs Normal Market (%)", height=380)
        st.plotly_chart(fig_time, use_container_width=True)

    with col_stats:
        st.dataframe(tdf[['Holding Time', 'After Event (Up %)', 'Edge Boost', 'Avg Move %']], use_container_width=True, hide_index=True)

# ─── TAB 3: THE DAILY ROUTINE ───
with tab_timing:
    st.subheader("The Market's Daily Rhythm")
    df_reset = df.reset_index()
    high_idx = df_reset.groupby('trading_day')['high'].idxmax()
    low_idx  = df_reset.groupby('trading_day')['low'].idxmin()

    high_times = df_reset.loc[high_idx][['trading_day', 'datetime']].rename(columns={'datetime': 'high_time'})
    low_times  = df_reset.loc[low_idx][['trading_day', 'datetime']].rename(columns={'datetime': 'low_time'})
    flows = pd.merge(high_times, low_times, on='trading_day')
    flows['High_First'] = flows['high_time'] < flows['low_time']
    
    total_days = len(flows)
    high_first_pct = (flows['High_First'].sum() / total_days) * 100
    low_first_pct = 100 - high_first_pct

    c_seq1, c_seq2 = st.columns(2)
    with c_seq1: st.markdown(f'<div class="metric-card"><div class="metric-title">Days ETH Sets the High First</div><div class="metric-value" style="color: #f85149;">{high_first_pct:.1f}%</div></div>', unsafe_allow_html=True)
    with c_seq2: st.markdown(f'<div class="metric-card"><div class="metric-title">Days ETH Sets the Low First</div><div class="metric-value" style="color: #3fb950;">{low_first_pct:.1f}%</div></div>', unsafe_allow_html=True)

    df['bar_range'] = df['high'] - df['low']
    hourly_vol = df.groupby('hour')['bar_range'].mean().reset_index()

    fig_vol = go.Figure(go.Bar(x=hourly_vol['hour'], y=hourly_vol['bar_range'], marker_color='#58a6ff'))
    fig_vol.update_layout(xaxis=dict(title="Hour of Day (Local Time)", dtick=1), yaxis=dict(title="Average Bar Range"), height=350)
    st.plotly_chart(fig_vol, use_container_width=True)

# ─── TAB 4: THE PUNCHCARD ───
with tab_punchcard:
    st.subheader("Algorithmic Timing: Top & Bottom Miner")
    df_reset = df.reset_index()
    high_idx = df_reset.groupby('trading_day')['high'].idxmax()
    low_idx  = df_reset.groupby('trading_day')['low'].idxmin()

    high_times = df_reset.loc[high_idx][['trading_day', 'datetime']].rename(columns={'datetime': 'Time_of_High'})
    low_times  = df_reset.loc[low_idx][['trading_day', 'datetime']].rename(columns={'datetime': 'Time_of_Low'})
    
    extremes_df = pd.merge(high_times, low_times, on='trading_day')
    extremes_df['Hour_of_High'] = extremes_df['Time_of_High'].dt.hour
    extremes_df['Hour_of_Low'] = extremes_df['Time_of_Low'].dt.hour
    extremes_df['Day_of_Week'] = extremes_df['Time_of_High'].dt.day_name()

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    high_counts = extremes_df.groupby(['Day_of_Week', 'Hour_of_High']).size().reset_index(name='Frequency')
    high_counts['Day_of_Week'] = pd.Categorical(high_counts['Day_of_Week'], categories=days_order, ordered=True)
    
    low_counts = extremes_df.groupby(['Day_of_Week', 'Hour_of_Low']).size().reset_index(name='Frequency')
    low_counts['Day_of_Week'] = pd.Categorical(low_counts['Day_of_Week'], categories=days_order, ordered=True)

    col_h, col_l = st.columns(2)
    with col_h:
        fig_h = px.scatter(high_counts, x="Hour_of_High", y="Day_of_Week", size="Frequency", color="Frequency", color_continuous_scale="Reds", size_max=22)
        fig_h.update_layout(xaxis_title="Hour Daily High Formed (Local Time)", yaxis_title="", xaxis=dict(tickmode='linear', dtick=1, range=[-1, 24]), yaxis={'categoryorder':'array', 'categoryarray':days_order[::-1]}, height=450)
        st.plotly_chart(fig_h, use_container_width=True)

    with col_l:
        fig_l = px.scatter(low_counts, x="Hour_of_Low", y="Day_of_Week", size="Frequency", color="Frequency", color_continuous_scale="Greens", size_max=22)
        fig_l.update_layout(xaxis_title="Hour Daily Low Formed (Local Time)", yaxis_title="", xaxis=dict(tickmode='linear', dtick=1, range=[-1, 24]), yaxis={'categoryorder':'array', 'categoryarray':days_order[::-1]}, height=450)
        st.plotly_chart(fig_l, use_container_width=True)