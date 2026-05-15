#!/usr/bin/env python3
"""아파트 매매가격지수 예측 노트북 생성 스크립트
실행: python create_notebook.py
"""
try:
    import nbformat as nbf
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nbformat', '-q'])
    import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {"name": "python", "version": "3.12.7"}
}

c = []
md   = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# =============================================================================
# SECTION 0: TITLE & SETUP
# =============================================================================
c.append(md(
    "# 서울 아파트 매매가격지수 예측 모델\n\n"
    "**과제 2** — KB Land 주간 데이터 (2021-05-10 ~ 2026-05-04) 기반 3주 예측\n\n"
    "---\n"
    "- 대상 구: 노원구 / 은평구 / 서대문구 / 서초구 / 강남구 / 송파구\n"
    "- 예측 목표일: 2026-05-25\n"
    "- 모델 비교 기준: **3-step ahead RMSE (전 모델 동일 조건)** — ARIMA 포함 공정 비교\n"
    "- 하이브리드 채택: Ljung-Box 잔차 검정 결과에 따라 조건부 적용\n\n"
    "---\n"
    "**분석 순서** : "
    "① 데이터 로드 → ② EDA → ③ 특성 공학 → "
    "④ ARIMA + 잔차 검정 → ⑤ 딥러닝 모델 → "
    "⑥ (조건부) 하이브리드 → ⑦ 공정 비교 → ⑧ 최종 예측"
))

c.append(code("""\
import subprocess, sys
for pkg in ['statsmodels', 'seaborn']:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
print('설치 완료')\
"""))

c.append(code("""\
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.fft import fft, fftfreq
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
torch.manual_seed(42)
np.random.seed(42)
print('임포트 완료')\
"""))

c.append(code("""\
DISTRICTS        = ['노원구', '은평구', '서대문구', '서초구', '강남구', '송파구']
FORECAST_HORIZON = 3
SEQ_LEN          = 12      # ACF 결과 보고 조정 가능
TEST_RATIO       = 0.20
VAL_RATIO        = 0.10
HIDDEN_SIZE      = 32      # 데이터 규모(~260주) 고려 축소
N_LAYERS         = 2
DROPOUT          = 0.2
BATCH_SIZE       = 16
N_EPOCHS         = 200
PATIENCE         = 20
DATA_PATH        = 'data/kb_apartment_index.csv'   # CSV 우선, xlsx 자동 폴백

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}  |  HIDDEN: {HIDDEN_SIZE}  |  SEQ_LEN: {SEQ_LEN}')\
"""))

# =============================================================================
# SECTION 1: DATA LOAD
# =============================================================================
c.append(md(
    "## 1. 데이터 로드 및 전처리\n\n"
    "- 소스: KB Land 주간 아파트 매매가격지수\n"
    "- 형식 자동 감지: CSV 우선, Excel(xlsx/xls) 폴백\n"
    "- 결측치 처리: ffill → bfill 순차 적용"
))

c.append(code("""\
def load_data(filepath):
    \"\"\"CSV / Excel 자동 감지 로드.\"\"\"\
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.csv':
        df_raw = pd.read_csv(filepath)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        date_col = df_raw.columns[0]
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors='coerce')
        df_raw = df_raw.dropna(subset=[date_col]).set_index(date_col).sort_index()
        available = [d for d in DISTRICTS if d in df_raw.columns]
        if not available:
            raise ValueError(f'구 열 없음 — 전체 컬럼: {list(df_raw.columns)}')
        return df_raw[available].astype(float).loc['2021-01-01':]

    # Excel 폴백
    if not os.path.exists(filepath):
        for alt_ext in ['.xlsx', '.xls']:
            alt = os.path.splitext(filepath)[0] + alt_ext
            if os.path.exists(alt):
                filepath, ext = alt, alt_ext
                break
        else:
            raise FileNotFoundError(f'파일 없음: {filepath}')

    engine = 'xlrd' if ext == '.xls' else 'openpyxl'
    probe  = pd.read_excel(filepath, header=None, nrows=20, engine=engine)
    header_row = next(
        (i for i in probe.index
         if sum(d in ' '.join(str(v) for v in probe.iloc[i].values) for d in DISTRICTS) >= 3),
        3
    )
    df_raw  = pd.read_excel(filepath, header=header_row, engine=engine)
    col_map = {}
    for col in df_raw.columns:
        for d in DISTRICTS:
            if d in str(col) and d not in col_map.values():
                col_map[col] = d; break
    date_col = df_raw.columns[0]
    df = df_raw[[date_col] + list(col_map.keys())].copy()
    df.rename(columns={date_col: 'date', **col_map}, inplace=True)
    for fmt in ['%Y.%m.%d', '%Y-%m-%d', None]:
        try:
            df['date'] = (pd.to_datetime(df['date'].astype(str), format=fmt, errors='coerce')
                          if fmt else pd.to_datetime(df['date'], errors='coerce'))
            if df['date'].notna().sum() > 10: break
        except Exception: pass
    df = df.dropna(subset=['date']).set_index('date').sort_index()
    available = [d for d in DISTRICTS if d in df.columns]
    return df[available].astype(float).loc['2021-01-01':]


def handle_missing(df):
    cnt = df.isnull().sum().sum()
    if cnt: print(f'결측치 {cnt}개 → ffill/bfill 처리')
    return df.ffill().bfill()


df = load_data(DATA_PATH)
df = handle_missing(df)
print(f'형태   : {df.shape}')
print(f'기간   : {df.index.min().date()} ~ {df.index.max().date()}  ({len(df)}주)')
print(f'결측치 : {df.isnull().sum().sum()}개')
print()
display(df.describe().round(2))\
"""))

# =============================================================================
# SECTION 2: EDA
# =============================================================================
c.append(md(
    "## 2. 탐색적 데이터 분석 (EDA)\n\n"
    "| 분석 항목 | 목적 |\n"
    "|-----------|------|\n"
    "| 2-1 시계열 시각화 | 전체 추이·이벤트 파악 |\n"
    "| 2-2 ADF 정상성 검정 | 차분 차수(d) 결정 |\n"
    "| 2-3 ACF / PACF | 최적 SEQ_LEN 및 ARIMA(p,q) 참고 |\n"
    "| 2-4 FFT 주기성 분석 | 주요 사이클 파악 |\n"
    "| 2-5 계절 분해 | 트렌드·계절·잔차 분리 |\n"
    "| 2-6 상관관계 히트맵 | 구별 공동 움직임 확인 |"
))

c.append(code("""\
# 2-1 시계열 시각화
fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True)
axes = axes.flatten()
event_bands = [
    ('2021-05-01', '2022-06-30', '#FFA500', 0.15, 'COVID 급등'),
    ('2022-07-01', '2023-12-31', '#FF4444', 0.15, '금리 인상·하락'),
    ('2024-01-01', '2026-06-01', '#44BB44', 0.12, '회복세'),
]
for i, d in enumerate(DISTRICTS):
    ax = axes[i]
    ax.plot(df.index, df[d], linewidth=1.5, color='steelblue')
    for s, e, col, a, lbl in event_bands:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=a, color=col,
                   label=lbl if i == 0 else '')
    ax.set_title(d, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=8)
    ax.set_ylabel('가격지수', fontsize=9)
axes[0].legend(loc='upper right', fontsize=8)
plt.suptitle('서울 주요 구 아파트 매매가격지수 주간 추이 (2021~2026)',
             fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()\
"""))

c.append(code("""\
# 2-2 ADF 정상성 검정
rows = []
for d in DISTRICTS:
    for d_ord, lbl in [(0, '원시'), (1, '1차 차분')]:
        s = df[d].copy()
        for _ in range(d_ord): s = s.diff().dropna()
        stat, pval, *_, crit, _ = adfuller(s.dropna(), autolag='AIC')
        rows.append({'구': d, '차분': lbl,
                     'ADF 통계량': round(stat, 3), 'p-value': round(pval, 4),
                     '정상성': '✓ 정상' if pval < 0.05 else '✗ 비정상'})
adf_df = pd.DataFrame(rows)
print('=== ADF 정상성 검정 ===')
display(adf_df)\
"""))

c.append(code("""\
# 2-3 ACF / PACF + SEQ_LEN 자동 권장
fig, axes = plt.subplots(len(DISTRICTS), 2, figsize=(14, 3 * len(DISTRICTS)))
suggested = []
for i, d in enumerate(DISTRICTS):
    s = df[d].diff().dropna()
    sm.graphics.tsa.plot_acf( s, lags=30, ax=axes[i,0], zero=False,
                               title=f'{d} — ACF (1차 차분)')
    sm.graphics.tsa.plot_pacf(s, lags=30, ax=axes[i,1], method='ywm', zero=False,
                               title=f'{d} — PACF (1차 차분)')
    acf_vals, conf = acf(s, nlags=30, alpha=0.05)
    sig = [lag for lag in range(1, 31)
           if abs(acf_vals[lag]) > abs(conf[lag, 1] - acf_vals[lag])]
    suggested.append(max(sig) + 2 if sig else SEQ_LEN)
plt.tight_layout(); plt.show()

auto_seq = int(np.median(suggested))
print(f'구별 ACF 유의 Lag: {suggested}')
print(f'SEQ_LEN 권장값 (중앙값 기준): {auto_seq}')
print(f'현재 SEQ_LEN = {SEQ_LEN}  (필요시 상수 셀에서 조정)')\
"""))

c.append(code("""\
# 2-4 FFT 주기성 분석
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, d in enumerate(DISTRICTS):
    vals = df[d].values.astype(float)
    n = len(vals); x = np.arange(n)
    detrended = vals - np.polyval(np.polyfit(x, vals, 1), x)
    freqs = fftfreq(n, d=1); power = np.abs(fft(detrended))
    pos = freqs > 0; periods = 1.0 / freqs[pos]; pw = power[pos]
    mask = (periods >= 2) & (periods <= 65)
    axes[i].plot(periods[mask], pw[mask], linewidth=1, color='steelblue')
    axes[i].set_xlabel('주기 (주)', fontsize=9)
    axes[i].set_ylabel('진폭', fontsize=9)
    axes[i].set_title(d, fontsize=11, fontweight='bold')
    for idx in np.argsort(pw[mask])[-3:][::-1]:
        p_v, v_v = periods[mask][idx], pw[mask][idx]
        axes[i].annotate(f'{p_v:.0f}w', xy=(p_v, v_v),
                         xytext=(3, 5), textcoords='offset points',
                         fontsize=8, color='red', fontweight='bold')
    for ref in [4, 13, 26, 52]:
        axes[i].axvline(ref, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
plt.suptitle('FFT 주기성 분석 — 선형 트렌드 제거 후 (2~65주)',
             fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()\
"""))

c.append(code("""\
# 2-5 계절 분해 (period=52)
fig, axes = plt.subplots(len(DISTRICTS), 4, figsize=(20, 3 * len(DISTRICTS)))
for i, d in enumerate(DISTRICTS):
    res = seasonal_decompose(df[d], model='additive', period=52, extrapolate_trend='freq')
    for j, (comp, lbl) in enumerate([(df[d], 'Original'), (res.trend, 'Trend'),
                                      (res.seasonal, 'Seasonal'), (res.resid, 'Residual')]):
        axes[i,j].plot(comp.index, comp.values, linewidth=0.8, color='steelblue')
        if i == 0: axes[i,j].set_title(lbl, fontsize=11, fontweight='bold')
        if j == 0: axes[i,j].set_ylabel(d, fontsize=9, fontweight='bold')
        axes[i,j].tick_params(labelsize=7); axes[i,j].grid(True, alpha=0.2)
plt.suptitle('계절 분해 (Additive, period=52주)', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()\
"""))

c.append(code("""\
# 2-6 구별 상관관계 히트맵
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(df[DISTRICTS].corr(), annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0.9, vmax=1.0, ax=axes[0], linewidths=0.5)
axes[0].set_title('수준(Level) 상관관계', fontsize=12, fontweight='bold')
sns.heatmap(df[DISTRICTS].diff().dropna().corr(), annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=-0.2, vmax=1.0, ax=axes[1], linewidths=0.5)
axes[1].set_title('1차 차분 상관관계', fontsize=12, fontweight='bold')
plt.suptitle('구별 아파트 가격지수 상관관계', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()\
"""))

# =============================================================================
# SECTION 3: FEATURE ENGINEERING
# =============================================================================
c.append(md(
    "## 3. 특성 공학 (Feature Engineering)\n\n"
    "| 종류 | 특성 | 의도 |\n"
    "|------|------|------|\n"
    "| Lag | `{구}_lag_1` ~ `{구}_lag_N` | 직전 N주 가격 직접 참조 |\n"
    "| Rolling | `rmean_{4,8,12}`, `rstd_{4,8,12}` | 단기·중기 추세 평활 |\n"
    "| Diff | `diff1`, `diff2` | 차분으로 비정상성 제거 |\n"
    "| YoY | `yoy` (lag 52) | 전년동기비 변화율 |\n"
    "| Temporal | `week/month sin·cos` | 계절성 순환 인코딩 |\n"
    "| Cross | 강남-노원 스프레드, 서울 평균 | 구간 공동 요인 |\n\n"
    "- 스케일러 fit: 훈련 데이터에만 적용 → 데이터 누수 방지\n"
    "- 예측 전략: Direct multi-output — t+1, t+2, t+3 동시 예측 → 오차 누적 없음"
))

c.append(code("""\
def engineer_features(df, seq_len=SEQ_LEN):
    feat = df.copy()
    for d in DISTRICTS:
        for lag in range(1, seq_len + 1):
            feat[f'{d}_lag_{lag}'] = df[d].shift(lag)
        for w in [4, 8, 12]:
            feat[f'{d}_rmean_{w}'] = df[d].rolling(w).mean()
            feat[f'{d}_rstd_{w}']  = df[d].rolling(w).std()
        feat[f'{d}_diff1'] = df[d].diff(1)
        feat[f'{d}_diff2'] = df[d].diff(2)
        if len(df) > 52:
            feat[f'{d}_yoy'] = df[d].pct_change(52) * 100
    week  = df.index.isocalendar().week.astype(float)
    month = df.index.month.astype(float)
    feat['week_sin']  = np.sin(2 * np.pi * week  / 52)
    feat['week_cos']  = np.cos(2 * np.pi * week  / 52)
    feat['month_sin'] = np.sin(2 * np.pi * month / 12)
    feat['month_cos'] = np.cos(2 * np.pi * month / 12)
    feat['trend_idx'] = np.linspace(0, 1, len(df))
    feat['gangnam_nowon_spread'] = df['강남구'] - df['노원구']
    feat['seoul_avg']            = df[DISTRICTS].mean(axis=1)
    feat['seoul_avg_diff1']      = feat['seoul_avg'].diff(1)
    feat = feat.dropna()
    print(f'특성 행렬: {feat.shape}  ({feat.shape[1]}개 특성, {feat.shape[0]}주)')
    return feat


def inverse_transform_targets(scaled, scaler, tgt_idx):
    result = np.zeros_like(scaled)
    for j, ci in enumerate(tgt_idx):
        result[..., j] = scaled[..., j] * scaler.data_range_[ci] + scaler.data_min_[ci]
    return result


def create_sequences(data_scaled, seq_len, horizon, tgt_idx):
    X, y = [], []
    for i in range(len(data_scaled) - seq_len - horizon + 1):
        X.append(data_scaled[i:i+seq_len, :])
        y.append(data_scaled[i+seq_len:i+seq_len+horizon, tgt_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_splits(feat_df, seq_len, horizon, test_r=TEST_RATIO, val_r=VAL_RATIO):
    feat_names = list(feat_df.columns)
    tgt_idx    = [feat_names.index(d) for d in DISTRICTS if d in feat_names]
    n          = len(feat_df)
    train_end  = int(n * (1 - test_r - val_r))
    val_end    = int(n * (1 - test_r))
    scaler     = MinMaxScaler((0, 1))
    arr        = feat_df.values.astype(np.float32)
    scaler.fit(arr[:train_end])
    scaled     = scaler.transform(arr)
    X, y       = create_sequences(scaled, seq_len, horizon, tgt_idx)
    ns         = len(X)
    tr_e       = int(ns * (1 - test_r - val_r))
    vl_e       = int(ns * (1 - test_r))
    return {
        'X_train': X[:tr_e],   'y_train': y[:tr_e],
        'X_val':   X[tr_e:vl_e], 'y_val': y[tr_e:vl_e],
        'X_test':  X[vl_e:],   'y_test': y[vl_e:],
        'scaler': scaler, 'tgt_idx': tgt_idx,
        'feat_df': feat_df, 'feat_names': feat_names,
        'train_end': train_end, 'val_end': val_end,
    }


feat_df = engineer_features(df, SEQ_LEN)
splits  = prepare_splits(feat_df, SEQ_LEN, FORECAST_HORIZON)
X_tr, y_tr = splits['X_train'], splits['y_train']
X_vl, y_vl = splits['X_val'],   splits['y_val']
X_te, y_te = splits['X_test'],  splits['y_test']
print(f'훈련: {X_tr.shape}  검증: {X_vl.shape}  테스트: {X_te.shape}')\
"""))

# =============================================================================
# SECTION 4: ARIMA + LJUNG-BOX
# =============================================================================
c.append(md(
    "## 4. 기준 모델: ARIMA + 잔차 검정\n\n"
    "- AIC 그리드 서치: p ∈ [0,3], d=1, q ∈ [0,3]\n"
    "- 평가: **3-step walk-forward** — 딥러닝과 동일 조건으로 공정 비교\n"
    "- Ljung-Box 잔차 검정 (lag=10)\n"
    "  - p < 0.05: 잔차에 비선형 패턴 존재 → 하이브리드 채택\n"
    "  - p ≥ 0.05: 잔차 = White Noise → ARIMA 단독으로 충분"
))

c.append(code("""\
def run_arima(df, test_r=TEST_RATIO, horizon=FORECAST_HORIZON):
    n, split = len(df), int(len(df) * (1 - test_r))
    results  = {}

    for district in DISTRICTS:
        series       = df[district].values
        train, test  = series[:split], series[split:]

        # AIC 그리드 서치
        best_aic, best_order = np.inf, (1, 1, 1)
        for p in range(4):
            for q in range(4):
                try:
                    m = ARIMA(train, order=(p, 1, q)).fit()
                    if m.aic < best_aic:
                        best_aic, best_order = m.aic, (p, 1, q)
                except Exception:
                    pass

        # 3-step walk-forward 평가 (딥러닝과 동일 조건)
        hist        = list(train)
        preds_3     = []
        actuals_3   = []
        for t in range(len(test) - horizon + 1):
            m  = ARIMA(hist, order=best_order).fit()
            fc = m.forecast(steps=horizon)
            preds_3.append(fc)
            actuals_3.append(test[t:t + horizon])
            hist.append(test[t])   # 1주씩 전진
        preds_3   = np.array(preds_3)
        actuals_3 = np.array(actuals_3)

        rmse = np.sqrt(np.mean((preds_3 - actuals_3) ** 2))
        mae  = np.mean(np.abs(preds_3 - actuals_3))
        mape = np.mean(np.abs((actuals_3 - preds_3) / (actuals_3 + 1e-8))) * 100

        # 잔차 검정 — 전체 데이터 재학습
        m_full    = ARIMA(series, order=best_order).fit()
        resid     = m_full.resid.dropna()
        lb        = acorr_ljungbox(resid, lags=[10], return_df=True)
        lb_pval   = lb['lb_pvalue'].values[0]
        future_3  = m_full.forecast(steps=3)

        results[district] = {
            'order': best_order,
            'preds_3': preds_3, 'actuals_3': actuals_3,
            'rmse': rmse, 'mae': mae, 'mape': mape,
            'lb_pval': lb_pval, 'residuals': resid,
            'future_3': future_3, 'model': m_full,
        }
        print(f'{district}: ARIMA{best_order}  '
              f'RMSE={rmse:.4f}  MAE={mae:.4f}  LB_p={lb_pval:.4f}')

    return results


print('ARIMA 3-step walk-forward 평가 중... (5~10분 소요)')
arima_results = run_arima(df)

last_date      = df.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                               periods=FORECAST_HORIZON, freq='W-MON')
arima_forecast = pd.DataFrame(
    {d: arima_results[d]['future_3'] for d in DISTRICTS},
    index=forecast_dates
)
arima_forecast.index.name = '예측일'
print('\\nARIMA 3주 예측:')
display(arima_forecast.round(2))\
"""))

c.append(code("""\
# Ljung-Box 결과 정리 + 하이브리드 채택 결정
lb_rows = []
for d in DISTRICTS:
    r = arima_results[d]
    lb_rows.append({
        '구': d,
        'ARIMA 차수': str(r['order']),
        'LB p-value': round(r['lb_pval'], 4),
        '판정': '비선형 패턴 존재 → 하이브리드 권장' if r['lb_pval'] < 0.05
                else 'White Noise → ARIMA 충분'
    })
display(pd.DataFrame(lb_rows))

need_hybrid = any(r['lb_pval'] < 0.05 for r in arima_results.values())
n_sig       = sum(r['lb_pval'] < 0.05 for r in arima_results.values())
print(f'\\n비선형 패턴 구: {n_sig}/{len(DISTRICTS)}개')
print(f'하이브리드 채택: {"YES" if need_hybrid else "NO — ARIMA 잔차 White Noise"}')\
"""))

# =============================================================================
# SECTION 5: DL MODELS
# =============================================================================
c.append(md(
    "## 5. 딥러닝 모델 (PyTorch)\n\n"
    "- 평가 조건: 3-step Direct Multi-output → ARIMA와 동일 기준\n"
    f"- 모델 규모: Hidden={'{HIDDEN_SIZE}'}  Layers={'{N_LAYERS}'}  SEQ_LEN={'{SEQ_LEN}'}\n\n"
    "| 모델 | 특징 |\n"
    "|------|------|\n"
    "| SimpleRNN | 단순 구조 기준선 |\n"
    "| LSTM | 장기 의존성, 게이트 구조 |\n"
    "| GRU | LSTM 경량화, 파라미터 절감 |\n"
    "| BiLSTM | 양방향, FC 입력 hidden×2 |"
))

c.append(code("""\
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class SimpleRNN(nn.Module):
    def __init__(self, n_feat, hidden, n_tgt, horizon, n_layers=1, dropout=0.2):
        super().__init__()
        self.horizon, self.n_tgt = horizon, n_tgt
        self.rnn  = nn.RNN(n_feat, hidden, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, horizon * n_tgt)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(self.drop(out[:, -1, :])).view(-1, self.horizon, self.n_tgt)


class LSTMModel(nn.Module):
    def __init__(self, n_feat, hidden, n_tgt, horizon, n_layers=2, dropout=0.2):
        super().__init__()
        self.horizon, self.n_tgt = horizon, n_tgt
        self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, horizon * n_tgt)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :])).view(-1, self.horizon, self.n_tgt)


class GRUModel(nn.Module):
    def __init__(self, n_feat, hidden, n_tgt, horizon, n_layers=2, dropout=0.2):
        super().__init__()
        self.horizon, self.n_tgt = horizon, n_tgt
        self.gru  = nn.GRU(n_feat, hidden, n_layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, horizon * n_tgt)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.drop(out[:, -1, :])).view(-1, self.horizon, self.n_tgt)


class BiLSTMModel(nn.Module):
    def __init__(self, n_feat, hidden, n_tgt, horizon, n_layers=2, dropout=0.2):
        super().__init__()
        self.horizon, self.n_tgt = horizon, n_tgt
        self.bilstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True,
                              dropout=dropout, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden * 2, horizon * n_tgt)
    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.fc(self.drop(out[:, -1, :])).view(-1, self.horizon, self.n_tgt)\
"""))

c.append(code("""\
def train_model(model, tr_ld, vl_ld, n_epochs=N_EPOCHS, lr=1e-3, patience=PATIENCE):
    model = model.to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, verbose=False)
    crit  = nn.MSELoss()
    best_val, best_w, no_imp, best_ep = np.inf, None, 0, 0
    hist = {'train': [], 'val': []}

    for ep in range(n_epochs):
        model.train()
        tl = sum(
            (lambda loss: (opt.zero_grad(), loss.backward(),
                           nn.utils.clip_grad_norm_(model.parameters(), 1.0),
                           opt.step(), loss.item() * len(Xb))[-1])(crit(model(Xb.to(DEVICE)), yb.to(DEVICE)))
            for Xb, yb in tr_ld
        ) / len(tr_ld.dataset)
        model.eval()
        with torch.no_grad():
            vl = sum(crit(model(Xb.to(DEVICE)), yb.to(DEVICE)).item() * len(Xb)
                     for Xb, yb in vl_ld) / len(vl_ld.dataset)
        hist['train'].append(tl); hist['val'].append(vl)
        sched.step(vl)
        if vl < best_val:
            best_val = vl
            best_w   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp   = 0; best_ep = ep + 1
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f'  Early stop @ ep{ep+1}  best={best_ep}  val={best_val:.6f}')
                break
        if (ep + 1) % 50 == 0:
            print(f'  ep {ep+1:3d} | train {tl:.6f} | val {vl:.6f}')
    model.load_state_dict(best_w)
    return hist, model


def evaluate_model(model, X_test, y_test, scaler, tgt_idx):
    model.eval()
    with torch.no_grad():
        ps = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    p_o = inverse_transform_targets(ps,     scaler, tgt_idx)
    y_o = inverse_transform_targets(y_test, scaler, tgt_idx)
    return {d: {
        'RMSE': round(np.sqrt(mean_squared_error(y_o[:,:,i].flatten(), p_o[:,:,i].flatten())), 4),
        'MAE':  round(mean_absolute_error(y_o[:,:,i].flatten(), p_o[:,:,i].flatten()), 4),
        'MAPE': round(np.mean(np.abs((y_o[:,:,i] - p_o[:,:,i]) / (y_o[:,:,i] + 1e-8))) * 100, 2),
    } for i, d in enumerate(DISTRICTS)}\
"""))

c.append(code("""\
N_FEAT = splits['X_train'].shape[2]
N_TGT  = len(DISTRICTS)
tr_ld  = DataLoader(TimeSeriesDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=False)
vl_ld  = DataLoader(TimeSeriesDataset(X_vl, y_vl), batch_size=BATCH_SIZE, shuffle=False)

model_cfgs = {
    'SimpleRNN': SimpleRNN(  N_FEAT, HIDDEN_SIZE, N_TGT, FORECAST_HORIZON, 1,        DROPOUT),
    'LSTM':      LSTMModel(  N_FEAT, HIDDEN_SIZE, N_TGT, FORECAST_HORIZON, N_LAYERS, DROPOUT),
    'GRU':       GRUModel(   N_FEAT, HIDDEN_SIZE, N_TGT, FORECAST_HORIZON, N_LAYERS, DROPOUT),
    'BiLSTM':    BiLSTMModel(N_FEAT, HIDDEN_SIZE, N_TGT, FORECAST_HORIZON, N_LAYERS, DROPOUT),
}

trained_models, histories, all_metrics = {}, {}, {}
for name, model in model_cfgs.items():
    print(f'\\n▶ {name} 학습 중...')
    hist, trained          = train_model(model, tr_ld, vl_ld)
    trained_models[name]   = trained
    histories[name]        = hist
    all_metrics[name]      = evaluate_model(trained, X_te, y_te,
                                             splits['scaler'], splits['tgt_idx'])
    avg_r = np.mean([m['RMSE'] for m in all_metrics[name].values()])
    print(f'  평균 RMSE: {avg_r:.4f}')
print('\\n모든 DL 모델 학습 완료')\
"""))

c.append(code("""\
# 학습 손실 곡선
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
for i, (name, hist) in enumerate(histories.items()):
    axes[i].plot(hist['train'], label='Train', linewidth=1.2, color='steelblue')
    axes[i].plot(hist['val'],   label='Val',   linewidth=1.2, linestyle='--', color='darkorange')
    axes[i].set_title(name, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Epoch'); axes[i].set_ylabel('MSE Loss')
    axes[i].legend(); axes[i].grid(True, alpha=0.3); axes[i].set_yscale('log')
plt.suptitle('모델별 학습/검증 손실 곡선', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()\
"""))

# =============================================================================
# SECTION 5-b: CONDITIONAL HYBRID
# =============================================================================
c.append(md(
    "## 5-b. 하이브리드 모델 (조건부 적용)\n\n"
    "- 적용 조건: Ljung-Box p < 0.05 구가 1개 이상 존재\n"
    "- 구조\n"
    "  - Step 1: ARIMA로 선형 성분 포착 → 잔차 추출\n"
    "  - Step 2: 경량 LSTM으로 잔차의 비선형 패턴 학습\n"
    "  - Step 3: 최종 예측 = ARIMA 예측 + LSTM 잔차 예측\n"
    "- 잔차 White Noise 구: ARIMA 예측값 그대로 사용"
))

c.append(code("""\
class ResidualLSTM(nn.Module):
    \"\"\"ARIMA 잔차 전용 경량 단변량 LSTM.\"\"\"\
    def __init__(self, seq_len, hidden=16, n_layers=1, dropout=0.1, horizon=3):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(1, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden, horizon)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def make_resid_sequences(resid, seq_len, horizon):
    X, y = [], []
    for i in range(len(resid) - seq_len - horizon + 1):
        X.append(resid[i:i+seq_len, np.newaxis])
        y.append(resid[i+seq_len:i+seq_len+horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_hybrid(arima_results, df, test_r=TEST_RATIO,
                 seq_len=SEQ_LEN, horizon=FORECAST_HORIZON):
    split = int(len(df) * (1 - test_r))
    hybrid_results = {}

    for district in DISTRICTS:
        r = arima_results[district]

        if r['lb_pval'] >= 0.05:
            # White Noise → ARIMA 그대로
            hybrid_results[district] = {
                'rmse': r['rmse'], 'mae': r['mae'], 'mape': r['mape'],
                'future_3': r['future_3'], 'mode': 'ARIMA only'
            }
            print(f'{district}: White Noise → ARIMA 사용  RMSE={r["rmse"]:.4f}')
            continue

        # ARIMA 훈련 잔차 추출
        train  = df[district].values[:split]
        series = df[district].values
        arima_tr = ARIMA(train, order=r['order']).fit()
        resid_tr = arima_tr.resid.values

        # 잔차 스케일링
        rs = MinMaxScaler((-1, 1))
        rs.fit(resid_tr.reshape(-1, 1))
        resid_sc = rs.transform(resid_tr.reshape(-1, 1)).flatten()

        # 잔차 시퀀스 분할
        Xr, yr   = make_resid_sequences(resid_sc, seq_len, horizon)
        sp       = int(len(Xr) * 0.85)
        tr_r = DataLoader(TimeSeriesDataset(Xr[:sp], yr[:sp]), batch_size=16, shuffle=False)
        vl_r = DataLoader(TimeSeriesDataset(Xr[sp:], yr[sp:]), batch_size=16, shuffle=False)

        # 잔차 LSTM 학습
        rlstm = ResidualLSTM(seq_len, hidden=16, horizon=horizon).to(DEVICE)
        _, rlstm = train_model(rlstm, tr_r, vl_r, n_epochs=100, patience=15)

        # 3-step walk-forward 하이브리드 평가
        hist        = list(train)
        test_data   = series[split:]
        hp, ha      = [], []
        for t in range(len(test_data) - horizon + 1):
            am   = ARIMA(hist, order=r['order']).fit()
            afc  = am.forecast(steps=horizon)
            recent_sc = rs.transform(am.resid.values[-seq_len:].reshape(-1,1)).flatten()
            Xt = torch.tensor(recent_sc[np.newaxis, :, np.newaxis], dtype=torch.float32).to(DEVICE)
            rlstm.eval()
            with torch.no_grad():
                rp_sc = rlstm(Xt).cpu().numpy().flatten()
            rp = rs.inverse_transform(rp_sc.reshape(-1,1)).flatten()
            hp.append(afc + rp)
            ha.append(test_data[t:t+horizon])
            hist.append(test_data[t])

        hp, ha = np.array(hp), np.array(ha)
        rmse = np.sqrt(np.mean((hp - ha) ** 2))
        mae  = np.mean(np.abs(hp - ha))
        mape = np.mean(np.abs((ha - hp) / (ha + 1e-8))) * 100

        # 미래 예측
        af  = ARIMA(series, order=r['order']).fit()
        afc = af.forecast(steps=horizon)
        rc  = rs.transform(af.resid.values[-seq_len:].reshape(-1,1)).flatten()
        Xt  = torch.tensor(rc[np.newaxis, :, np.newaxis], dtype=torch.float32).to(DEVICE)
        rlstm.eval()
        with torch.no_grad():
            rp = rs.inverse_transform(rlstm(Xt).cpu().numpy().flatten().reshape(-1,1)).flatten()
        future_hybrid = afc + rp

        hybrid_results[district] = {
            'rmse': rmse, 'mae': mae, 'mape': mape,
            'future_3': future_hybrid, 'mode': 'ARIMA+LSTM'
        }
        print(f'{district} Hybrid: RMSE={rmse:.4f}  (ARIMA: {r["rmse"]:.4f})')

    return hybrid_results


if need_hybrid:
    print('▶ 하이브리드 모델 학습 중...')
    hybrid_results = train_hybrid(arima_results, df)
else:
    print('▶ 하이브리드 불필요 — ARIMA 잔차 White Noise 확인')
    hybrid_results = None\
"""))

# =============================================================================
# SECTION 6: FAIR COMPARISON
# =============================================================================
c.append(md(
    "## 6. 공정 비교 — 전 모델 3-step RMSE 기준\n\n"
    "- ARIMA: 3-step walk-forward (Section 4)\n"
    "- 딥러닝: 3-step direct multi-output (Section 5)\n"
    "- 하이브리드: 3-step walk-forward (Section 5-b, 조건부)\n"
    "- 동일 평가 조건 → 직접 수치 비교 가능\n"
    "- `best_name`: **전체 모델 포함** 평균 RMSE 최솟값으로 선택"
))

c.append(code("""\
# 전 모델 RMSE 집계
all_rmse = {}
for name in trained_models:
    all_rmse[name] = np.mean([m['RMSE'] for m in all_metrics[name].values()])
all_rmse['ARIMA'] = np.mean([r['rmse'] for r in arima_results.values()])
if need_hybrid and hybrid_results:
    all_rmse['Hybrid'] = np.mean([v['rmse'] for v in hybrid_results.values()])

# 요약 표
summary = pd.DataFrame([
    {'모델': k, '평균 RMSE': round(v, 4),
     '비고': '★ 최우수' if k == min(all_rmse, key=all_rmse.get) else ''}
    for k, v in sorted(all_rmse.items(), key=lambda x: x[1])
])
print('=== 전 모델 공정 비교 (3-step RMSE, 오름차순) ===')
display(summary)

# 구별 상세 표
detail = []
for model_name in all_rmse:
    if model_name in all_metrics:
        src = all_metrics[model_name]
    elif model_name == 'ARIMA':
        src = {d: {'RMSE': arima_results[d]['rmse'],
                   'MAE':  arima_results[d]['mae'],
                   'MAPE': arima_results[d]['mape']} for d in DISTRICTS}
    elif model_name == 'Hybrid' and hybrid_results:
        src = {d: {'RMSE': hybrid_results[d]['rmse'],
                   'MAE':  hybrid_results[d]['mae'],
                   'MAPE': hybrid_results[d]['mape']} for d in DISTRICTS}
    else:
        continue
    for d, m in src.items():
        detail.append({'모델': model_name, '구': d,
                       'RMSE': m['RMSE'], 'MAE': m['MAE'], 'MAPE(%)': m['MAPE']})
print('\\n=== 구별 상세 성능 ===')
display(pd.DataFrame(detail).sort_values(['구', 'RMSE']).reset_index(drop=True))

best_name = min(all_rmse, key=all_rmse.get)
print(f'\\n▶ 최적 모델: {best_name}  (평균 RMSE: {all_rmse[best_name]:.4f})')\
"""))

# =============================================================================
# SECTION 7: FINAL PREDICTION
# =============================================================================
c.append(md(
    "## 7. 최종 예측 (2026-05-25)\n\n"
    "- 최적 모델로 t+1, t+2, t+3 예측\n"
    "- ARIMA / Hybrid / DL 모델 유형에 따라 예측 방식 자동 분기\n"
    "- 목표일: 2026-05-25 (데이터 마지막 일자 기준 +3주)"
))

c.append(code("""\
def predict_future_nn(model, feat_df, scaler, tgt_idx, seq_len=SEQ_LEN):
    last_win = feat_df.values[-seq_len:].astype(np.float32)
    scaled   = scaler.transform(last_win)
    Xt       = torch.tensor(scaled[np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        ps = model(Xt).cpu().numpy()
    return inverse_transform_targets(ps, scaler, tgt_idx).squeeze(0)  # (horizon, 6)


last_date      = feat_df.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                               periods=FORECAST_HORIZON, freq='W-MON')

if best_name == 'ARIMA':
    pred_arr = np.stack([arima_results[d]['future_3'] for d in DISTRICTS], axis=1)
elif best_name == 'Hybrid' and hybrid_results:
    pred_arr = np.stack([hybrid_results[d]['future_3'] for d in DISTRICTS], axis=1)
else:
    pred_arr = predict_future_nn(trained_models[best_name], feat_df,
                                 splits['scaler'], splits['tgt_idx'])

forecast_df = pd.DataFrame(pred_arr, index=forecast_dates, columns=DISTRICTS)
forecast_df.index.name = '예측일'
print(f'=== 최종 예측 결과 [{best_name}] ===')
display(forecast_df.round(2))
print('\\n▶ 2026-05-25 예측값 (과제 제출):')
display(forecast_df.iloc[2].round(2).to_frame('예측 지수'))\
"""))

c.append(code("""\
# 최종 시각화
split_idx = int(len(df) * (1 - TEST_RATIO))

# 테스트 구간 t+1 예측값 확보
if best_name in trained_models:
    bm = trained_models[best_name]; bm.eval()
    with torch.no_grad():
        tp = bm(torch.tensor(X_te, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    tp_o = inverse_transform_targets(tp, splits['scaler'], splits['tgt_idx'])[:, 0, :]
    te_dates = df.index[split_idx:split_idx + len(tp_o)]
elif best_name == 'ARIMA':
    tp_o = np.stack([arima_results[d]['preds_3'][:, 0] for d in DISTRICTS], axis=1)
    te_dates = df.index[split_idx:split_idx + len(tp_o)]
else:
    tp_o, te_dates = None, df.index[split_idx:]

fig, axes = plt.subplots(3, 2, figsize=(16, 13))
axes = axes.flatten()
for i, district in enumerate(DISTRICTS):
    ax = axes[i]
    ax.plot(df.index[:split_idx], df[district].values[:split_idx],
            color='steelblue', linewidth=1.2, label='훈련 데이터')
    ax.plot(df.index[split_idx:], df[district].values[split_idx:],
            color='royalblue', linewidth=1.2, alpha=0.7, label='테스트 실제값')
    if tp_o is not None:
        n_tp = min(len(tp_o), len(te_dates))
        ax.plot(te_dates[:n_tp], tp_o[:n_tp, i],
                color='darkorange', linewidth=1.2, linestyle='--',
                label=f'{best_name} 예측(t+1)')
    ax.plot(forecast_df.index, forecast_df[district].values,
            'r*', markersize=13, zorder=5, label='미래 예측 3주')
    for dt, val in zip(forecast_df.index, forecast_df[district]):
        ax.annotate(f'{val:.1f}', xy=(dt, val),
                    xytext=(5, 8), textcoords='offset points',
                    fontsize=8.5, color='red', fontweight='bold')
    ax.axvline(df.index[split_idx], color='gray', linestyle=':', alpha=0.6)
    ax.set_title(district, fontsize=12, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=8)
    ax.set_ylabel('가격지수', fontsize=9)
plt.suptitle(f'{best_name} — 훈련 / 테스트 / 3주 미래 예측 (목표: 2026-05-25)',
             fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()\
"""))

c.append(code("""\
# 최종 결과 요약
sep = '=' * 62
n_sig = sum(r['lb_pval'] < 0.05 for r in arima_results.values())
print(sep)
print('과제 2 — 서울 아파트 매매가격지수 3주 예측 최종 결과')
print(sep)
print(f'최적 모델    : {best_name}')
print(f'선택 근거    : 전 모델 포함 3-step RMSE 최솟값')
print(f'입력 윈도우  : {SEQ_LEN}주  |  예측 horizon: {FORECAST_HORIZON}주')
print(f'Hidden Size  : {HIDDEN_SIZE}  |  Layers: {N_LAYERS}')
print()
print(f'하이브리드 채택 여부')
print(f'  LB p<0.05 구: {n_sig}/{len(DISTRICTS)}개')
print(f'  결론: {"채택" if need_hybrid else "기각 — 잔차 White Noise"}')
print()
print('2026-05-25 예측 지수:')
for d in DISTRICTS:
    print(f'  {d:<6}: {forecast_df.iloc[2][d]:.2f}')
print()
print('전 모델 3-step RMSE 비교 (공정):')
for name, rmse in sorted(all_rmse.items(), key=lambda x: x[1]):
    marker = '  ◀ 최우수' if name == best_name else ''
    print(f'  {name:<12}: {rmse:.4f}{marker}')
print(sep)\
"""))

# =============================================================================
# BUILD
# =============================================================================
nb.cells = c

import json, os
out_path = 'apartment_price_prediction.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nbf.writes(nb, version=4), f, ensure_ascii=False)

# nbformat 직접 쓰기
nbf.write(nb, out_path)
print(f'노트북 생성 완료: {out_path}')
print(f'총 셀 수: {len(nb.cells)}개')
