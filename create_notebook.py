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

import os

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.12.7"
    }
}

c = []
md   = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# =============================================================================
# SECTION 0: SETUP
# =============================================================================

c.append(md(
    "# 서울 아파트 매매가격지수 예측 모델\n\n"
    "**과제 2** — KB Land 주간 데이터 (2021.05.10 ~ 2026.05.04) 기반 3주 예측  \n"
    "**대상 구**: 노원구 / 은평구 / 서대문구 / 서초구 / 강남구 / 송파구  \n"
    "**예측 목표일**: 2026-05-25\n\n"
    "---\n"
    "**분석 순서**: ① 데이터 로드 → ② EDA (정상성/주기성/분해) → ③ 특성 공학 "
    "→ ④ ARIMA 기준 모델 → ⑤ RNN/LSTM/GRU/BiLSTM 학습 → ⑥ 모델 비교 → ⑦ 최종 예측"
))

c.append(code("""\
# 필요 패키지 설치 (최초 1회)
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
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
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
# ── 전역 상수 ──────────────────────────────────────────────────────────────
DISTRICTS        = ['노원구', '은평구', '서대문구', '서초구', '강남구', '송파구']
FORECAST_HORIZON = 3       # 예측 주수 (t+1, t+2, t+3)
SEQ_LEN          = 12      # 입력 시퀀스 길이 → ACF 결과 보고 조정
TEST_RATIO       = 0.20
VAL_RATIO        = 0.10
HIDDEN_SIZE      = 64
N_LAYERS         = 2
DROPOUT          = 0.2
BATCH_SIZE       = 16
N_EPOCHS         = 150
PATIENCE         = 20
DATA_PATH        = 'data/kb_apartment_index.xlsx'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')\
"""))

# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================

c.append(md(
    "## 1. 데이터 로드 및 전처리\n\n"
    "> **준비**: KB Land "
    "(https://data.kbland.kr/kbstats/wmh?tIdx=HT01&tsIdx=weekAptSalePriceInx) "
    "에서 Excel 파일 다운로드 → `data/kb_apartment_index.xlsx` 로 저장 후 실행"
))

c.append(code("""\
def load_kb_excel(filepath):
    \"\"\"KB Land 주간 아파트 매매가격지수 Excel 자동 로드.\"\"\"
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f'파일 없음: {filepath}\\n'
            'KB Land에서 Excel을 다운로드한 뒤 data/ 폴더에 저장하세요.'
        )

    ext    = os.path.splitext(filepath)[1].lower()
    engine = 'xlrd' if ext == '.xls' else 'openpyxl'

    # 첫 20행 프로빙 → 헤더 행 위치 탐지
    probe = pd.read_excel(filepath, header=None, nrows=20, engine=engine)
    header_row = None
    for i in probe.index:
        row_str = ' '.join(str(v) for v in probe.iloc[i].values)
        if sum(d in row_str for d in DISTRICTS) >= 3:
            header_row = i
            break
    if header_row is None:
        print('경고: 헤더 자동 감지 실패 → header_row=3 사용')
        header_row = 3

    df_raw = pd.read_excel(filepath, header=header_row, engine=engine)

    # 날짜 열 탐지 (첫 열부터 순서대로 확인)
    date_col = df_raw.columns[0]
    for col in df_raw.columns:
        sample = df_raw[col].dropna().head(5)
        try:
            parsed = pd.to_datetime(sample.astype(str), format='%Y.%m.%d', errors='coerce')
            if parsed.notna().all():
                date_col = col
                break
        except Exception:
            pass
        try:
            parsed = pd.to_datetime(sample, errors='coerce')
            if parsed.notna().all() and parsed.dt.year.between(2000, 2030).all():
                date_col = col
                break
        except Exception:
            pass

    # 구 열 매핑 (부분 문자열 매칭)
    col_map = {}
    for col in df_raw.columns:
        for d in DISTRICTS:
            if d in str(col) and d not in col_map.values():
                col_map[col] = d
                break

    missing = [d for d in DISTRICTS if d not in col_map.values()]
    if missing:
        print(f'경고: 찾지 못한 구 → {missing}')
        print(f'전체 컬럼: {list(df_raw.columns[:40])}')

    df = df_raw[[date_col] + list(col_map.keys())].copy()
    df.rename(columns={date_col: 'date', **col_map}, inplace=True)

    # 날짜 파싱 (형식 순차 시도)
    for fmt in ['%Y.%m.%d', '%Y-%m-%d', None]:
        try:
            if fmt:
                df['date'] = pd.to_datetime(df['date'].astype(str), format=fmt, errors='coerce')
            else:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            if df['date'].notna().sum() > 10:
                break
        except Exception:
            pass

    # Excel 직렬 번호 폴백
    if df['date'].isna().mean() > 0.5:
        df['date'] = pd.to_datetime(
            df_raw[date_col].astype(float), unit='D', origin='1899-12-30', errors='coerce'
        )

    df = df.dropna(subset=['date'])
    df = df.set_index('date').sort_index()
    available = [d for d in DISTRICTS if d in df.columns]
    df = df[available].astype(float)
    df = df.loc['2021-01-01':]
    return df


def handle_missing(df):
    cnt = df.isnull().sum().sum()
    if cnt > 0:
        print(f'결측치 {cnt}개 → ffill/bfill 처리')
    return df.ffill().bfill()


def inspect_data(df):
    print(f'형태     : {df.shape}')
    print(f'기간     : {df.index.min().date()} ~ {df.index.max().date()}  ({len(df)}주)')
    print(f'결측치   :\\n{df.isnull().sum().to_string()}')
    print(f'\\n기술통계 :\\n{df.describe().round(2).to_string()}')\
"""))

c.append(code("""\
os.makedirs('data', exist_ok=True)
df = load_kb_excel(DATA_PATH)
df = handle_missing(df)
inspect_data(df)
df.head()\
"""))

# =============================================================================
# SECTION 2: EDA
# =============================================================================

c.append(md(
    "## 2. 탐색적 데이터 분석 (EDA)\n\n"
    "| 분석 | 목적 |\n"
    "|------|------|\n"
    "| 2.1 시계열 시각화 | 전체 추이·이벤트 파악 |\n"
    "| 2.2 ADF 정상성 검정 | 차분 차수(d) 결정 |\n"
    "| 2.3 ACF / PACF | 최적 SEQ_LEN 및 ARIMA(p,q) 결정 |\n"
    "| 2.4 FFT 주기성 분석 | 주요 사이클 파악 |\n"
    "| 2.5 계절 분해 | 트렌드·계절·잔차 분리 |\n"
    "| 2.6 상관관계 히트맵 | 구별 공동 움직임 확인 |"
))

# 2.1 Time series
c.append(code("""\
# ── 2.1 시계열 시각화 (이벤트 밴드 포함) ────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True)
axes = axes.flatten()

event_bands = [
    ('2021-05-01', '2022-06-30', '#FFA500', 0.15, 'COVID 급등 (21-22)'),
    ('2022-07-01', '2023-12-31', '#FF4444', 0.15, '금리 인상·하락 (22-23)'),
    ('2024-01-01', '2026-06-01', '#44BB44', 0.12, '회복세 (24-26)'),
]

for i, district in enumerate(DISTRICTS):
    ax = axes[i]
    ax.plot(df.index, df[district], linewidth=1.5, color='steelblue')
    for start, end, color, alpha, label in event_bands:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=alpha, color=color,
                   label=label if i == 0 else '')
    ax.set_title(district, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=8)
    ax.set_ylabel('가격지수', fontsize=9)

axes[0].legend(loc='upper right', fontsize=8)
plt.suptitle('서울 주요 구 아파트 매매가격지수 주간 추이 (2021~2026)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_01_price_series.png', dpi=150, bbox_inches='tight')
plt.show()\
"""))

# 2.2 ADF
c.append(code("""\
# ── 2.2 ADF 정상성 검정 ─────────────────────────────────────────────────────
adf_rows = []
for district in DISTRICTS:
    for d_order, label in [(0, '원시'), (1, '1차 차분'), (2, '2차 차분')]:
        series = df[district].copy()
        for _ in range(d_order):
            series = series.diff().dropna()
        stat, pval, _, _, crit, _ = adfuller(series.dropna(), autolag='AIC')
        adf_rows.append({
            '구': district, '차분': label,
            'ADF 통계량': round(stat, 3),
            'p-value': round(pval, 4),
            '정상성': '✓ 정상' if pval < 0.05 else '✗ 비정상',
            '1% 임계값': round(crit['1%'], 3)
        })

adf_df = pd.DataFrame(adf_rows)
print('=== ADF 정상성 검정 (p < 0.05 → 정상) ===')
display(adf_df)
print('\\n▶ 예상: 원시 데이터 비정상 → 1차 차분 후 정상 (d=1 확인)')\
"""))

# 2.3 ACF/PACF
c.append(code("""\
# ── 2.3 ACF / PACF 분석 ─────────────────────────────────────────────────────
# ACF 마지막 유의 lag + 2 ≈ SEQ_LEN 권장값
fig, axes = plt.subplots(len(DISTRICTS), 2, figsize=(14, 3 * len(DISTRICTS)))

for i, district in enumerate(DISTRICTS):
    series = df[district].diff().dropna()
    sm.graphics.tsa.plot_acf(
        series, lags=30, ax=axes[i, 0], zero=False,
        title=f'{district} — ACF (1차 차분)'
    )
    sm.graphics.tsa.plot_pacf(
        series, lags=30, ax=axes[i, 1], method='ywm', zero=False,
        title=f'{district} — PACF (1차 차분)'
    )

plt.tight_layout()
plt.savefig('fig_02_acf_pacf.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'\\n▶ 현재 SEQ_LEN = {SEQ_LEN}  (ACF 확인 후 필요시 상수 셀에서 조정)')
print('▶ PACF 절단점 → ARIMA p 파라미터,  ACF 절단점 → ARIMA q 파라미터')\
"""))

# 2.4 FFT
c.append(code("""\
# ── 2.4 FFT 주기성 분석 ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()

ref_periods = [(4, '월간(4w)'), (13, '분기(13w)'), (26, '반년(26w)'), (52, '연간(52w)')]

for i, district in enumerate(DISTRICTS):
    vals = df[district].values.astype(float)
    n    = len(vals)
    x    = np.arange(n)
    detrended = vals - np.polyval(np.polyfit(x, vals, 1), x)

    freqs = fftfreq(n, d=1)
    power = np.abs(fft(detrended))

    pos     = freqs > 0
    periods = 1.0 / freqs[pos]
    pw      = power[pos]
    mask    = (periods >= 2) & (periods <= 65)

    axes[i].plot(periods[mask], pw[mask], linewidth=1, color='steelblue')
    axes[i].set_xlabel('주기 (주)', fontsize=9)
    axes[i].set_ylabel('진폭', fontsize=9)
    axes[i].set_title(district, fontsize=11, fontweight='bold')

    # 상위 3개 주기 표시
    top_idx = np.argsort(pw[mask])[-3:][::-1]
    for idx in top_idx:
        p_val = periods[mask][idx]
        v_val = pw[mask][idx]
        axes[i].annotate(f'{p_val:.0f}w', xy=(p_val, v_val),
                         xytext=(3, 5), textcoords='offset points',
                         fontsize=8, color='red', fontweight='bold')

    for ref_p, ref_lbl in ref_periods:
        if ref_p <= 65:
            axes[i].axvline(ref_p, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

plt.suptitle('FFT 주기성 분석 — 선형 트렌드 제거 후 (2~65주)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_03_fft.png', dpi=150, bbox_inches='tight')
plt.show()
print('\\n▶ 강한 연간(52주) 계절성이 없으면 SEQ_LEN을 짧게 유지해도 됨')\
"""))

# 2.5 Decomposition
c.append(code("""\
# ── 2.5 계절 분해 (period=52, 연간 계절성 가정) ─────────────────────────────
fig, axes = plt.subplots(len(DISTRICTS), 4, figsize=(20, 3 * len(DISTRICTS)))

for i, district in enumerate(DISTRICTS):
    result = seasonal_decompose(df[district], model='additive',
                                period=52, extrapolate_trend='freq')
    parts = [
        (df[district],    'Original'),
        (result.trend,    'Trend'),
        (result.seasonal, 'Seasonal'),
        (result.resid,    'Residual'),
    ]
    for j, (comp, lbl) in enumerate(parts):
        axes[i, j].plot(comp.index, comp.values, linewidth=0.8, color='steelblue')
        if i == 0:
            axes[i, j].set_title(lbl, fontsize=11, fontweight='bold')
        if j == 0:
            axes[i, j].set_ylabel(district, fontsize=9, fontweight='bold')
        axes[i, j].tick_params(labelsize=7)
        axes[i, j].grid(True, alpha=0.2)

plt.suptitle('계절 분해 (Additive, period=52주)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_04_decomposition.png', dpi=150, bbox_inches='tight')
plt.show()\
"""))

# 2.6 Correlation
c.append(code("""\
# ── 2.6 구별 상관관계 히트맵 ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

corr_level = df[DISTRICTS].corr()
sns.heatmap(corr_level, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0.9, vmax=1.0, ax=axes[0], linewidths=0.5)
axes[0].set_title('수준(Level) 상관관계', fontsize=12, fontweight='bold')

corr_diff = df[DISTRICTS].diff().dropna().corr()
sns.heatmap(corr_diff, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=-0.2, vmax=1.0, ax=axes[1], linewidths=0.5)
axes[1].set_title('1차 차분 상관관계', fontsize=12, fontweight='bold')

plt.suptitle('구별 아파트 가격지수 상관관계', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_05_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
print('\\n▶ 수준 상관관계: 모든 구가 0.9 이상 → 공통 시장 요인 지배')
print('▶ 차분 상관관계: 강남3구(강남/서초/송파) vs 외곽(노원/은평/서대문) 구분 확인')\
"""))

# =============================================================================
# SECTION 3: FEATURE ENGINEERING
# =============================================================================

c.append(md(
    "## 3. 특성 공학 (Feature Engineering)\n\n"
    "ACF/PACF 분석 결과를 바탕으로 **SEQ_LEN을 확정**하고, 신경망 입력을 위한 파생 변수를 생성합니다.\n\n"
    "| 종류 | 특성 | 의도 |\n"
    "|------|------|------|\n"
    "| Lag | `{구}_lag_1` ~ `{구}_lag_N` | 직전 N주 가격 직접 참조 |\n"
    "| Rolling | `rmean_{4,8,12}`, `rstd_{4,8,12}` | 단기·중기 추세 평활 |\n"
    "| Diff | `diff1`, `diff2` | 차분으로 비정상성 제거 |\n"
    "| YoY | `yoy` (lag 52) | 전년동기비 변화율 |\n"
    "| Temporal | `week/month sin·cos` | 계절성 순환 인코딩 |\n"
    "| Cross | 강남-노원 스프레드, 서울 평균 | 구간 공동 요인 |\n\n"
    "**Direct multi-output 전략**: X=(n, SEQ_LEN, n_feat), y=(n, 3, 6) — t+1, t+2, t+3 동시 예측 → 오차 누적 없음"
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

    week = df.index.isocalendar().week.astype(float)
    feat['week_sin']  = np.sin(2 * np.pi * week / 52)
    feat['week_cos']  = np.cos(2 * np.pi * week / 52)
    month = df.index.month.astype(float)
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
    \"\"\"타겟 열만 역변환 — scaler 파라미터 직접 사용 (전체 특성 재구성 불필요).\"\"\"
    result = np.zeros_like(scaled)
    for j, ci in enumerate(tgt_idx):
        result[..., j] = scaled[..., j] * scaler.data_range_[ci] + scaler.data_min_[ci]
    return result


def create_sequences(data_scaled, seq_len, horizon, tgt_idx):
    \"\"\"X: (samples, seq_len, n_feat)  y: (samples, horizon, n_tgt) — Direct multi-output.\"\"\"
    X, y = [], []
    n = len(data_scaled)
    for i in range(n - seq_len - horizon + 1):
        X.append(data_scaled[i:i+seq_len, :])
        y.append(data_scaled[i+seq_len:i+seq_len+horizon, tgt_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_splits(feat_df, seq_len, horizon, test_r=TEST_RATIO, val_r=VAL_RATIO):
    \"\"\"시간 순서 유지 분할 + MinMaxScaler (훈련 데이터에만 fit → 데이터 누수 방지).\"\"\"
    feat_names = list(feat_df.columns)
    tgt_idx    = [feat_names.index(d) for d in DISTRICTS if d in feat_names]

    n         = len(feat_df)
    train_end = int(n * (1 - test_r - val_r))
    val_end   = int(n * (1 - test_r))

    scaler = MinMaxScaler((0, 1))
    arr    = feat_df.values.astype(np.float32)
    scaler.fit(arr[:train_end])       # 훈련 데이터에만 fit
    scaled = scaler.transform(arr)

    X, y  = create_sequences(scaled, seq_len, horizon, tgt_idx)
    ns    = len(X)
    tr_e  = int(ns * (1 - test_r - val_r))
    vl_e  = int(ns * (1 - test_r))

    return {
        'X_train': X[:tr_e],   'y_train': y[:tr_e],
        'X_val':   X[tr_e:vl_e], 'y_val': y[tr_e:vl_e],
        'X_test':  X[vl_e:],   'y_test': y[vl_e:],
        'scaler': scaler, 'tgt_idx': tgt_idx,
        'feat_df': feat_df, 'feat_names': feat_names,
        'train_end': train_end, 'val_end': val_end,
    }\
"""))

c.append(code("""\
feat_df = engineer_features(df, SEQ_LEN)
splits  = prepare_splits(feat_df, SEQ_LEN, FORECAST_HORIZON)

X_tr, y_tr = splits['X_train'], splits['y_train']
X_vl, y_vl = splits['X_val'],   splits['y_val']
X_te, y_te = splits['X_test'],  splits['y_test']

print(f'훈련: {X_tr.shape}  검증: {X_vl.shape}  테스트: {X_te.shape}')
tgt_names = [splits['feat_names'][i] for i in splits['tgt_idx']]
print(f'타겟 열: {tgt_names}')\
"""))

# =============================================================================
# SECTION 4: ARIMA BASELINE
# =============================================================================

c.append(md(
    "## 4. 기준 모델: ARIMA\n\n"
    "신경망 모델과 비교할 통계적 기준선.  \n"
    "AIC 기준 그리드 서치(p∈[0..3], d=1, q∈[0..3])로 최적 ARIMA(p,1,q) 탐색,  \n"
    "롤링 예측(walk-forward)으로 테스트셋 평가 → 3주 미래 예측."
))

c.append(code("""\
def run_arima_baseline(df, test_r=TEST_RATIO, horizon=FORECAST_HORIZON):
    n      = len(df)
    split  = int(n * (1 - test_r))
    results = {}

    for district in DISTRICTS:
        series        = df[district].values
        train, test   = series[:split], series[split:]

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

        # 롤링 예측 (walk-forward)
        hist  = list(train)
        preds = []
        for t in range(len(test)):
            m = ARIMA(hist, order=best_order).fit()
            preds.append(m.forecast(steps=1)[0])
            hist.append(test[t])
        preds = np.array(preds)

        rmse = np.sqrt(mean_squared_error(test, preds))
        mae  = mean_absolute_error(test, preds)
        mape = np.mean(np.abs((test - preds) / (test + 1e-8))) * 100

        # 전체 데이터로 재학습 → 3주 미래 예측
        m_full   = ARIMA(series, order=best_order).fit()
        future_3 = m_full.forecast(steps=3)

        results[district] = {
            'order': best_order, 'test_preds': preds, 'test_actuals': test,
            'rmse': rmse, 'mae': mae, 'mape': mape, 'future_3': future_3
        }
        print(f'{district}: ARIMA{best_order}  RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mape:.2f}%')

    return results\
"""))

c.append(code("""\
print('ARIMA 학습 중... (3~7분 소요)')
arima_results = run_arima_baseline(df)

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

# =============================================================================
# SECTION 5: PYTORCH MODELS
# =============================================================================

c.append(md(
    "## 5. 딥러닝 모델 정의 (PyTorch)\n\n"
    "공통 구조: `RNN계층 → Dropout → FC(horizon × n_targets)` → reshape `(B, horizon, 6)`\n\n"
    "| 모델 | 특이사항 |\n"
    "|------|----------|\n"
    "| **SimpleRNN** | `nn.RNN`, 1레이어, 빠른 수렴 |\n"
    "| **LSTM** | `nn.LSTM`, 2레이어, 장기 의존성 |\n"
    "| **GRU** | `nn.GRU`, 2레이어, LSTM 경량화 |\n"
    "| **BiLSTM** | `bidirectional=True`, FC 입력 hidden×2 |"
))

c.append(code("""\
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):           return len(self.X)
    def __getitem__(self, i):    return self.X[i], self.y[i]


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

# =============================================================================
# SECTION 6: TRAINING
# =============================================================================

c.append(md(
    "## 6. 모델 학습\n\n"
    "- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)  \n"
    "- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)  \n"
    "- **Gradient Clipping**: max_norm=1.0  \n"
    "- **Early Stopping**: patience=20 (val loss 기준, 150 에포크 한도)"
))

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
        tl = 0
        for Xb, yb in tr_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(Xb)
        tl /= len(tr_ld.dataset)

        model.eval()
        vl = 0
        with torch.no_grad():
            for Xb, yb in vl_ld:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                vl += crit(model(Xb), yb).item() * len(Xb)
        vl /= len(vl_ld.dataset)

        hist['train'].append(tl)
        hist['val'].append(vl)
        sched.step(vl)

        if vl < best_val:
            best_val = vl
            best_w   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp   = 0
            best_ep  = ep + 1
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f'  Early stop @ epoch {ep+1}  best={best_ep}  val={best_val:.6f}')
                break

        if (ep + 1) % 30 == 0:
            print(f'  ep {ep+1:3d} | train {tl:.6f} | val {vl:.6f}')

    model.load_state_dict(best_w)
    return hist, model


def evaluate_model(model, X_test, y_test, scaler, tgt_idx):
    model.eval()
    with torch.no_grad():
        ps = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    p_orig = inverse_transform_targets(ps,     scaler, tgt_idx)
    y_orig = inverse_transform_targets(y_test, scaler, tgt_idx)
    metrics = {}
    for i, d in enumerate(DISTRICTS):
        pf = p_orig[:, :, i].flatten()
        yf = y_orig[:, :, i].flatten()
        metrics[d] = {
            'RMSE': round(np.sqrt(mean_squared_error(yf, pf)), 4),
            'MAE':  round(mean_absolute_error(yf, pf), 4),
            'MAPE': round(np.mean(np.abs((yf - pf) / (yf + 1e-8))) * 100, 2),
        }
    return metrics\
"""))

c.append(code("""\
N_FEAT = splits['X_train'].shape[2]
N_TGT  = len(DISTRICTS)

tr_ld = DataLoader(TimeSeriesDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=False)
vl_ld = DataLoader(TimeSeriesDataset(X_vl, y_vl), batch_size=BATCH_SIZE, shuffle=False)

print(f'입력 특성 수: {N_FEAT}')
print(f'FC 출력 크기: {FORECAST_HORIZON} × {N_TGT} = {FORECAST_HORIZON * N_TGT}')\
"""))

c.append(code("""\
model_cfgs = {
    'SimpleRNN': SimpleRNN(  N_FEAT, HIDDEN_SIZE, N_TGT, FORECAST_HORIZON, 1,        DROPOUT),
    'LSTM':      LSTMModel(  N_FEAT, HIDDEN_SIZE, N_TGT, FORECAST_HORIZON, N_LAYERS, DROPOUT),
    'GRU':       GRUModel(   N_FEAT, HIDDEN_SIZE, N_TGT, FORECAST_HORIZON, N_LAYERS, DROPOUT),
    'BiLSTM':    BiLSTMModel(N_FEAT, HIDDEN_SIZE, N_TGT, FORECAST_HORIZON, N_LAYERS, DROPOUT),
}

trained_models, histories, all_metrics = {}, {}, {}

for name, model in model_cfgs.items():
    print('\\n' + '='*50)
    print(f'▶ {name} 학습 중...')
    hist, trained = train_model(model, tr_ld, vl_ld)
    trained_models[name] = trained
    histories[name]      = hist
    all_metrics[name]    = evaluate_model(
        trained, X_te, y_te, splits['scaler'], splits['tgt_idx']
    )
    avg_r = np.mean([m['RMSE'] for m in all_metrics[name].values()])
    print(f'  → 평균 RMSE: {avg_r:.4f}')

print('\\n모든 모델 학습 완료!')\
"""))

c.append(code("""\
# 학습 손실 곡선
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
for i, (name, hist) in enumerate(histories.items()):
    axes[i].plot(hist['train'], label='Train', linewidth=1.2, color='steelblue')
    axes[i].plot(hist['val'],   label='Val',   linewidth=1.2, linestyle='--', color='darkorange')
    axes[i].set_title(f'{name} — 손실 곡선', fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel('MSE Loss')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    axes[i].set_yscale('log')
plt.suptitle('모델별 학습/검증 손실 곡선', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_06_loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()\
"""))

# =============================================================================
# SECTION 7: EVALUATION & PREDICTION
# =============================================================================

c.append(md(
    "## 7. 모델 평가 및 최종 예측\n\n"
    "### 7.1 모델 비교표 (RMSE / MAE / MAPE)\n"
    "### 7.2 최적 모델로 2026-05-25 예측\n"
    "### 7.3 최종 시각화"
))

c.append(code("""\
# ── 7.1 모델 비교표 ──────────────────────────────────────────────────────────
arima_metrics = {
    d: {'RMSE': arima_results[d]['rmse'],
        'MAE':  arima_results[d]['mae'],
        'MAPE': arima_results[d]['mape']}
    for d in DISTRICTS
}
all_metrics_full = {**all_metrics, 'ARIMA': arima_metrics}

rows = []
for model_name, dm in all_metrics_full.items():
    for district, m in dm.items():
        rows.append({'모델': model_name, '구': district,
                     'RMSE': m['RMSE'], 'MAE': m['MAE'], 'MAPE(%)': m['MAPE']})

cmp_df = pd.DataFrame(rows)
avg_df = cmp_df.groupby('모델')[['RMSE','MAE','MAPE(%)']].mean().round(4).reset_index()
avg_df['구'] = '★ AVERAGE'
cmp_all = pd.concat([cmp_df, avg_df], ignore_index=True)

print('=== 모델 성능 비교 (테스트셋) ===')
display(cmp_all[cmp_all['구'] == '★ AVERAGE'].sort_values('RMSE').reset_index(drop=True))

# 최적 딥러닝 모델 선택
nn_avg     = {k: np.mean([m['RMSE'] for m in v.values()])
              for k, v in all_metrics.items()}
best_name  = min(nn_avg, key=nn_avg.get)
best_model = trained_models[best_name]
print(f'\\n▶ 최적 딥러닝 모델: {best_name}  (평균 RMSE: {nn_avg[best_name]:.4f})')\
"""))

c.append(code("""\
# ── 7.2 최종 3주 예측 ────────────────────────────────────────────────────────
def predict_future(model, feat_df, scaler, tgt_idx, seq_len=SEQ_LEN):
    last_win = feat_df.values[-seq_len:].astype(np.float32)
    scaled   = scaler.transform(last_win)
    Xt       = torch.tensor(scaled[np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        ps = model(Xt).cpu().numpy()     # (1, horizon, 6)
    return inverse_transform_targets(ps, scaler, tgt_idx).squeeze(0)  # (horizon, 6)


pred_vals      = predict_future(best_model, feat_df, splits['scaler'], splits['tgt_idx'])
last_date      = feat_df.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                               periods=FORECAST_HORIZON, freq='W-MON')

forecast_df = pd.DataFrame(pred_vals, index=forecast_dates, columns=DISTRICTS)
forecast_df.index.name = '예측일'

print(f'=== 최종 예측 결과 [{best_name}] ===')
display(forecast_df.round(2))

print('\\n▶ 2026-05-25 예측값 (과제 제출):')
display(forecast_df.iloc[2].round(2).to_frame('예측 지수'))\
"""))

c.append(code("""\
# ── 7.3 최종 시각화 ──────────────────────────────────────────────────────────
best_model.eval()
with torch.no_grad():
    tp = best_model(torch.tensor(X_te, dtype=torch.float32).to(DEVICE)).cpu().numpy()
tp_orig = inverse_transform_targets(tp,   splits['scaler'], splits['tgt_idx'])

n_df      = len(df)
split_idx = int(n_df * (1 - TEST_RATIO))
test_dates = df.index[split_idx:]

fig, axes = plt.subplots(3, 2, figsize=(16, 13))
axes = axes.flatten()

for i, district in enumerate(DISTRICTS):
    ax = axes[i]
    ax.plot(df.index[:split_idx], df[district].values[:split_idx],
            color='steelblue', linewidth=1.2, label='훈련 데이터')
    ax.plot(test_dates, df[district].values[split_idx:],
            color='royalblue', linewidth=1.2, alpha=0.7, label='테스트 실제값')
    n_tp = min(len(tp_orig), len(test_dates))
    ax.plot(test_dates[:n_tp], tp_orig[:n_tp, 0, i],
            color='darkorange', linewidth=1.2, linestyle='--',
            label=f'{best_name} 예측')
    ax.plot(forecast_df.index, forecast_df[district].values,
            'r*', markersize=13, zorder=5, label='미래 예측 (3주)')
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
plt.tight_layout()
plt.savefig(f'fig_07_final_{best_name}.png', dpi=150, bbox_inches='tight')
plt.show()\
"""))

c.append(code("""\
# ── 최종 결과 요약 ───────────────────────────────────────────────────────────
sep = '=' * 60
print(sep)
print('과제 2 — 서울 아파트 매매가격지수 3주 예측 최종 결과')
print(sep)
print(f'최적 모델  : {best_name}  (평균 RMSE 기준)')
print(f'입력 윈도우: {SEQ_LEN}주  |  예측 horizon: {FORECAST_HORIZON}주')
print(f'테스트 기간: {df.index[split_idx].date()} ~ {df.index[-1].date()}')
print()
print('▶ 2026-05-25 예측 지수:')
for d in DISTRICTS:
    print(f'  {d:<7s} : {forecast_df.iloc[2][d]:.2f}')
print()
print('▶ 딥러닝 모델 평균 성능 (RMSE):')
arima_avg = np.mean([v['rmse'] for v in arima_results.values()])
for name in ['SimpleRNN', 'LSTM', 'GRU', 'BiLSTM']:
    marker = '  ◀ 최우수' if name == best_name else ''
    print(f'  {name:<12s}: {nn_avg[name]:.4f}{marker}')
print(f'  {"ARIMA":<12s}: {arima_avg:.4f}  (기준 모델)')
print(sep)\
"""))

# =============================================================================
# ASSEMBLE AND SAVE
# =============================================================================
nb.cells = c

out_dir  = r'C:\Users\KiKi\Desktop\ai'
out_path = os.path.join(out_dir, 'apartment_price_prediction.ipynb')
data_dir = os.path.join(out_dir, 'data')

os.makedirs(data_dir, exist_ok=True)

with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'노트북 생성 완료: {out_path}')
print(f'총 {len(c)}개 셀')
print(f'data/ 폴더: {data_dir}')
print()
print('다음 단계:')
print('  1. KB Land에서 Excel 다운로드 → data/kb_apartment_index.xlsx 저장')
print('  2. jupyter lab 실행 후 apartment_price_prediction.ipynb 열기')
print('  3. Kernel → Restart & Run All')
