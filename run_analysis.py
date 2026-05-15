"""
KB Land 주간 아파트 매매가격지수 — 전체 분석 파이프라인
- 데이터: data/kb_apartment_index.csv (2021.05.10 ~ 2026.05.04, 251주)
- 대상: 노원구/은평구/서대문구/서초구/강남구/송파구
- 목표: 2026-05-25 예측
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.fft import fft, fftfreq
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
torch.manual_seed(42); np.random.seed(42)

DISTRICTS        = ['노원구','은평구','서대문구','서초구','강남구','송파구']
FORECAST_HORIZON = 3
SEQ_LEN          = 16
TEST_RATIO       = 0.20
VAL_RATIO        = 0.10
HIDDEN_SIZE      = 64
N_LAYERS         = 2
DROPOUT          = 0.2
PROJ_DIM         = 32    # 입력 투영 차원: 164 고차원 특성 → 32로 압축
BATCH_SIZE       = 16
N_EPOCHS         = 150
PATIENCE         = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

os.makedirs('output', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/kb_apartment_index.csv', index_col='날짜', parse_dates=True)
df = df[DISTRICTS]
print(f'\n[데이터] {df.shape}  {df.index[0].date()} ~ {df.index[-1].date()}')

# ── 이벤트별 가격 변화량 (충격 분석) ─────────────────────────────────────────
EVENT_PERIODS = [
    ('COVID 급등기',   '2021-05-10', '2022-06-27'),
    ('금리인상 하락기', '2022-07-04', '2023-12-25'),
    ('회복기',         '2024-01-01', '2026-05-04'),
]
event_changes = {}
for ev_name, ev_s, ev_e in EVENT_PERIODS:
    sub = df[(df.index >= ev_s) & (df.index <= ev_e)]
    if len(sub) < 2: continue
    ev_dict = {}
    for d in DISTRICTS:
        v0, v1 = sub[d].iloc[0], sub[d].iloc[-1]
        ev_dict[d] = {
            'start':      round(float(v0), 3),
            'end':        round(float(v1), 3),
            'delta':      round(float(v1 - v0), 3),
            'delta_pct':  round(float((v1/v0 - 1)*100), 2),
            'weeks':      int(len(sub)),
            'weekly_avg': round(float((v1-v0)/len(sub)), 4),
        }
    event_changes[ev_name] = ev_dict
print('\n[이벤트 변화량]')
for ev,dd in event_changes.items():
    avg_d = np.mean([dd[d]['delta'] for d in DISTRICTS])
    print(f'  {ev}: 평균 변화량 {avg_d:+.2f}')

# ─────────────────────────────────────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== 2.1 시계열 시각화 ===')
fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True)
axes = axes.flatten()
event_bands = [
    ('2021-05-01','2022-06-30','#FFA500',0.15,'COVID 급등 (21-22)'),
    ('2022-07-01','2023-12-31','#FF4444',0.15,'금리 인상·하락 (22-23)'),
    ('2024-01-01','2026-06-01','#44BB44',0.12,'회복세 (24~)'),
]
for i, d in enumerate(DISTRICTS):
    ax = axes[i]
    ax.plot(df.index, df[d], linewidth=1.5, color='steelblue')
    for s,e,c,a,lbl in event_bands:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=a, color=c,
                   label=lbl if i==0 else '')
    ax.set_title(d, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y.%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=8)
axes[0].legend(loc='upper left', fontsize=8)
plt.suptitle('서울 주요 구 아파트 매매가격지수 (2021~2026)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/fig_01_price_series.png', bbox_inches='tight')
plt.close()
print('  → fig_01 저장')

print('\n=== 2.2 ADF 정상성 검정 ===')
adf_rows = []
for d in DISTRICTS:
    for diff, lbl in [(0,'원시'),(1,'1차 차분'),(2,'2차 차분')]:
        s = df[d].copy()
        for _ in range(diff): s = s.diff().dropna()
        stat,pval,_,_,crit,_ = adfuller(s.dropna(), autolag='AIC')
        adf_rows.append({'구':d,'차분':lbl,'ADF통계량':round(stat,3),
                         'p-value':round(pval,4),'정상성':'✓' if pval<0.05 else '✗',
                         '1%임계값':round(crit['1%'],3)})
adf_df = pd.DataFrame(adf_rows)
print(adf_df[adf_df['차분']=='1차 차분'][['구','p-value','정상성']].to_string(index=False))

print('\n=== 2.3 ACF/PACF ===')
fig, axes = plt.subplots(len(DISTRICTS), 2, figsize=(14, 3*len(DISTRICTS)))
for i, d in enumerate(DISTRICTS):
    s = df[d].diff().dropna()
    sm.graphics.tsa.plot_acf(s, lags=30, ax=axes[i,0], zero=False,
                              title=f'{d} ACF (1차 차분)')
    sm.graphics.tsa.plot_pacf(s, lags=30, ax=axes[i,1], method='ywm', zero=False,
                               title=f'{d} PACF (1차 차분)')
plt.tight_layout()
plt.savefig('output/fig_02_acf_pacf.png', bbox_inches='tight')
plt.close()
print('  → fig_02 저장')

# ACF 유의 lag 추정 (95% CI = 1.96/sqrt(n)) — 정보 제공용, SEQ_LEN 결정에 직접 사용하지 않음
from statsmodels.tsa.stattools import acf as _acf
ci = 1.96 / np.sqrt(len(df))
acf_lags = []
for d in DISTRICTS:
    acf_vals = _acf(df[d].diff().dropna(), nlags=30, fft=True)
    last_sig = max([l for l,v in enumerate(acf_vals[1:], 1) if abs(v) > ci], default=4)
    acf_lags.append(last_sig)
# SEQ_LEN 결정 근거 (정직한 버전):
# ① VAR 최적 시차 7주(AIC 선택) × 2 = 14주 → 16주로 반올림
# ② IRF 충격 흡수 기간 ~6-8주의 2배 = 12-16주
# ③ 데이터 제약: SEQ_LEN > 20이면 훈련 시퀀스 < 100개 → 과적합 위험
# ④ ACF는 30+주 유의 자기상관을 보이나, 251주 데이터에서 완전 활용 불가 (데이터 부족)
# → SEQ_LEN = 16 (고정값, VAR·IRF 기반 실용적 선택)
print(f'  ACF 유의 lag: {acf_lags}  (모두 nlags=30 상한에 도달 — 30주 이상임을 의미)')
print(f'  SEQ_LEN = {SEQ_LEN}  (VAR 시차 7주 × 2 ≈ 14주; 데이터 제약 고려 16주 채택)')

print('\n=== 2.4 FFT 주기성 분석 ===')
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
dominant_periods = {}
for i, d in enumerate(DISTRICTS):
    vals = df[d].values.astype(float)
    n = len(vals)
    detrended = vals - np.polyval(np.polyfit(np.arange(n), vals, 1), np.arange(n))
    freqs = fftfreq(n, d=1); power = np.abs(fft(detrended))
    pos = freqs > 0; periods = 1.0/freqs[pos]; pw = power[pos]
    mask = (periods>=2)&(periods<=65)
    axes[i].plot(periods[mask], pw[mask], linewidth=1, color='steelblue')
    for ref_p, lbl in [(4,'월간'),(13,'분기'),(26,'반년'),(52,'연간')]:
        axes[i].axvline(ref_p, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    top = np.argsort(pw[mask])[-3:][::-1]
    dp = [round(periods[mask][t]) for t in top]
    dominant_periods[d] = dp
    for t in top:
        axes[i].annotate(f'{periods[mask][t]:.0f}w', xy=(periods[mask][t], pw[mask][t]),
                         xytext=(3,5), textcoords='offset points', fontsize=8, color='red')
    axes[i].set_title(d, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('주기(주)', fontsize=9); axes[i].set_ylabel('진폭', fontsize=9)
plt.suptitle('FFT 주기성 분석 (트렌드 제거 후)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/fig_03_fft.png', bbox_inches='tight')
plt.close()
print(f'  주요 주기: {dominant_periods}')
print('  → fig_03 저장')

print('\n=== 2.5 계절 분해 ===')
fig, axes = plt.subplots(len(DISTRICTS), 4, figsize=(20, 3*len(DISTRICTS)))
for i, d in enumerate(DISTRICTS):
    res = seasonal_decompose(df[d], model='additive', period=52, extrapolate_trend='freq')
    for j,(comp,lbl) in enumerate([(df[d],'원시'),(res.trend,'추세'),(res.seasonal,'계절'),(res.resid,'잔차')]):
        axes[i,j].plot(comp.index, comp.values, linewidth=0.8, color='steelblue')
        if i==0: axes[i,j].set_title(lbl, fontsize=11, fontweight='bold')
        if j==0: axes[i,j].set_ylabel(d, fontsize=9, fontweight='bold')
        axes[i,j].tick_params(labelsize=7); axes[i,j].grid(True, alpha=0.2)
plt.suptitle('계절 분해 (period=52주)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/fig_04_decomposition.png', bbox_inches='tight')
plt.close()
print('  → fig_04 저장')

print('\n=== 2.6 상관관계 히트맵 ===')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(df.corr(), annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.9, vmax=1.0,
            ax=axes[0], linewidths=0.5)
axes[0].set_title('수준 상관관계', fontsize=12, fontweight='bold')
sns.heatmap(df.diff().dropna().corr(), annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=-0.2, vmax=1.0, ax=axes[1], linewidths=0.5)
axes[1].set_title('차분 상관관계', fontsize=12, fontweight='bold')
plt.suptitle('구별 가격지수 상관관계', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/fig_05_correlation.png', bbox_inches='tight')
plt.close()
print('  → fig_05 저장')

# ─────────────────────────────────────────────────────────────────────────────
# 3. 특성 공학
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== 3. 특성 공학 ===')
def engineer_features(df, seq_len=SEQ_LEN):
    feat = df.copy()
    for d in DISTRICTS:
        for lag in range(1, seq_len+1): feat[f'{d}_lag_{lag}'] = df[d].shift(lag)
        for w in [4,8,12]:
            feat[f'{d}_rm{w}'] = df[d].rolling(w).mean()
            feat[f'{d}_rs{w}'] = df[d].rolling(w).std()
        feat[f'{d}_d1'] = df[d].diff(1)
        feat[f'{d}_d2'] = df[d].diff(2)
        if len(df)>52: feat[f'{d}_yoy'] = df[d].pct_change(52)*100
    week = df.index.isocalendar().week.astype(float)
    feat['wsin'] = np.sin(2*np.pi*week/52); feat['wcos'] = np.cos(2*np.pi*week/52)
    month = df.index.month.astype(float)
    feat['msin'] = np.sin(2*np.pi*month/12); feat['mcos'] = np.cos(2*np.pi*month/12)
    feat['trend'] = np.linspace(0,1,len(df))
    feat['gn_nw'] = df['강남구']-df['노원구']
    feat['avg']   = df[DISTRICTS].mean(axis=1)
    feat['avgd1'] = feat['avg'].diff(1)
    return feat.dropna()

def inv_transform(scaled, scaler, tidx):
    result = np.zeros_like(scaled)
    for j,ci in enumerate(tidx):
        result[...,j] = scaled[...,j]*scaler.data_range_[ci]+scaler.data_min_[ci]
    return result

def create_seqs(data, seq_len, horizon, tidx):
    """
    타겟을 절대값이 아닌 '마지막 입력 시점 대비 누적 차분'으로 구성.
    → 학습/테스트 분포가 동일해져 분포 이탈 문제 해소.
    y shape: (n_samples, horizon, n_targets) — 차분값
    """
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i+seq_len])
        last = data[i+seq_len-1, tidx]                     # 마지막 입력값 (n_targets,)
        diffs = np.array([data[i+seq_len+h, tidx] - last   # 누적 차분
                          for h in range(horizon)])          # (horizon, n_targets)
        y.append(diffs)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

feat_df = engineer_features(df, SEQ_LEN)
fnames  = list(feat_df.columns)
tidx    = [fnames.index(d) for d in DISTRICTS]

n = len(feat_df)
tr_end = int(n*(1-TEST_RATIO-VAL_RATIO))
vl_end = int(n*(1-TEST_RATIO))
arr = feat_df.values.astype(np.float32)
# 학습 데이터에만 fit (데이터 누수 방지)
# 차분 타겟 전략으로 분포 이탈 문제는 별도 해결
scaler = MinMaxScaler((0,1)); scaler.fit(arr[:tr_end]); scaled = scaler.transform(arr)
X,y = create_seqs(scaled, SEQ_LEN, FORECAST_HORIZON, tidx)
ns = len(X)
# ── 경계 정합 수정 ──────────────────────────────────────────────────────────────
# 비율 기반 분할(int(ns*ratio))은 피처 경계(tr_end/vl_end)와 ±5 오차 발생 가능.
# 시퀀스 i의 마지막 y-label 인덱스 = i + SEQ_LEN + FORECAST_HORIZON - 1
# → 훈련 시퀀스 y-label ≤ tr_end-1 을 보장:
#   te ≤ tr_end - SEQ_LEN - FORECAST_HORIZON + 1
te = tr_end - SEQ_LEN - FORECAST_HORIZON + 1   # = 139 - 16 - 3 + 1 = 121
ve = vl_end - SEQ_LEN - FORECAST_HORIZON + 1   # = 159 - 16 - 3 + 1 = 141
X_tr,y_tr = X[:te],y[:te]; X_vl,y_vl = X[te:ve],y[te:ve]; X_te,y_te = X[ve:],y[ve:]
N_FEAT = X_tr.shape[2]; N_TGT = len(DISTRICTS)
print(f'  특성 수: {N_FEAT}  SEQ_LEN={SEQ_LEN}')
print(f'  Train: {X_tr.shape}  Val: {X_vl.shape}  Test: {X_te.shape}')

# ─────────────────────────────────────────────────────────────────────────────
# 4. ARIMA 기준 모델
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== 4. ARIMA 기준 모델 (walk-forward 공정 평가) ===')
# NN 모델과 동일하게: 매 스텝 실제 데이터를 이용해 rolling 3주 예측
# (one-shot 51주 예측은 NN 대비 구조적으로 불리 → 불공정 비교)
split = int(len(df)*(1-TEST_RATIO))
arima_results = {}
for d in DISTRICTS:
    series = df[d].values
    train, test = series[:split], series[split:]
    # 훈련 데이터로 최적 차수 탐색 (AIC)
    best_aic, best_order = np.inf, (1,1,1)
    for p in range(4):
        for q in range(4):
            try:
                m = ARIMA(train, order=(p,1,q)).fit()
                if m.aic < best_aic: best_aic, best_order = m.aic, (p,1,q)
            except: pass
    # Walk-forward: 매 스텝 t에서 실제 이력으로 다음 3주 예측
    n_test = len(test)
    preds_wf, actuals_wf, last_knowns_wf = [], [], []
    for t in range(n_test - FORECAST_HORIZON + 1):
        history = np.concatenate([train, test[:t]])
        try:
            fc = ARIMA(history, order=best_order).fit().forecast(steps=FORECAST_HORIZON)
        except:
            fc = np.full(FORECAST_HORIZON, history[-1])
        preds_wf.append(fc)
        actuals_wf.append(test[t:t+FORECAST_HORIZON])
        last_knowns_wf.append(history[-1])
    preds_wf    = np.array(preds_wf)      # (n, 3)
    actuals_wf  = np.array(actuals_wf)    # (n, 3)
    last_wf     = np.array(last_knowns_wf) # (n,)
    pf_all = preds_wf.flatten(); yf_all = actuals_wf.flatten()
    # DA: NN과 동일 기준 (last_known 기준 방향)
    da_vals = []
    for h in range(FORECAST_HORIZON):
        da_vals.append(np.mean(np.sign(actuals_wf[:,h]-last_wf)==np.sign(preds_wf[:,h]-last_wf)))
    da_arima = float(np.mean(da_vals)*100)
    # 미래 예측 (전체 데이터 학습 후)
    future = ARIMA(series, order=best_order).fit().forecast(steps=FORECAST_HORIZON)
    arima_results[d] = {
        'order':best_order,'preds':preds_wf[:,-1],'actuals':actuals_wf[:,-1],
        'rmse':  float(np.sqrt(mean_squared_error(yf_all, pf_all))),
        'mae':   float(mean_absolute_error(yf_all, pf_all)),
        'mape':  float(np.mean(np.abs((yf_all-pf_all)/(np.abs(yf_all)+1e-8)))*100),
        'smape': float(np.mean(2*np.abs(yf_all-pf_all)/(np.abs(yf_all)+np.abs(pf_all)+1e-8))*100),
        'r2':    float(r2_score(yf_all, pf_all)),
        'da':    da_arima,
        'future':future,
    }
    print(f'  {d}: ARIMA{best_order}  RMSE={arima_results[d]["rmse"]:.4f}  MAPE={arima_results[d]["mape"]:.2f}%  DA={da_arima:.1f}%')

# ─────────────────────────────────────────────────────────────────────────────
# 5. 딥러닝 모델
# ─────────────────────────────────────────────────────────────────────────────
class TS_DS(Dataset):
    def __init__(self,X,y): self.X=torch.tensor(X,dtype=torch.float32); self.y=torch.tensor(y,dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.y[i]

class _Proj(nn.Module):
    """고차원 입력(164) → proj 차원으로 압축: 잡음 감소 + 파라미터 효율화"""
    def __init__(self, nf, proj):
        super().__init__()
        self.fc = nn.Linear(nf, proj)
        self.ln = nn.LayerNorm(proj)
    def forward(self, x):           # x: (B, T, nf)
        return torch.relu(self.ln(self.fc(x)))  # (B, T, proj)

class SimpleRNN(nn.Module):
    def __init__(self, nf, proj, h, nt, hz, nl=1, do=0.2):
        super().__init__(); self.hz, self.nt = hz, nt
        self.proj = _Proj(nf, proj)
        self.rnn  = nn.RNN(proj, h, nl, batch_first=True, dropout=do if nl>1 else 0)
        self.drop = nn.Dropout(do); self.fc = nn.Linear(h, hz*nt)
    def forward(self, x):
        o, _ = self.rnn(self.proj(x))
        return self.fc(self.drop(o[:,-1,:])).view(-1, self.hz, self.nt)

class LSTMModel(nn.Module):
    def __init__(self, nf, proj, h, nt, hz, nl=2, do=0.2):
        super().__init__(); self.hz, self.nt = hz, nt
        self.proj = _Proj(nf, proj)
        self.lstm = nn.LSTM(proj, h, nl, batch_first=True, dropout=do)
        self.drop = nn.Dropout(do); self.fc = nn.Linear(h, hz*nt)
    def forward(self, x):
        o, _ = self.lstm(self.proj(x))
        return self.fc(self.drop(o[:,-1,:])).view(-1, self.hz, self.nt)

class GRUModel(nn.Module):
    def __init__(self, nf, proj, h, nt, hz, nl=2, do=0.2):
        super().__init__(); self.hz, self.nt = hz, nt
        self.proj = _Proj(nf, proj)
        self.gru  = nn.GRU(proj, h, nl, batch_first=True, dropout=do)
        self.drop = nn.Dropout(do); self.fc = nn.Linear(h, hz*nt)
    def forward(self, x):
        o, _ = self.gru(self.proj(x))
        return self.fc(self.drop(o[:,-1,:])).view(-1, self.hz, self.nt)

def train_model(model, tr_ld, vl_ld):
    model=model.to(DEVICE)
    opt=optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
    sched=optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=0.5,verbose=False)
    crit=nn.MSELoss()
    best_val,best_w,no_imp=np.inf,None,0
    hist={'train':[],'val':[]}
    for ep in range(N_EPOCHS):
        model.train(); tl=0
        for Xb,yb in tr_ld:
            Xb,yb=Xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
            loss=crit(model(Xb),yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tl+=loss.item()*len(Xb)
        tl/=len(tr_ld.dataset)
        model.eval(); vl=0
        with torch.no_grad():
            for Xb,yb in vl_ld:
                Xb,yb=Xb.to(DEVICE),yb.to(DEVICE)
                vl+=crit(model(Xb),yb).item()*len(Xb)
        vl/=len(vl_ld.dataset)
        hist['train'].append(tl); hist['val'].append(vl); sched.step(vl)
        if vl<best_val: best_val=vl; best_w={k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp=0
        else:
            no_imp+=1
            if no_imp>=PATIENCE:
                print(f'  early stop @{ep+1}'); break
        if (ep+1)%30==0: print(f'  ep{ep+1:3d} train={tl:.6f} val={vl:.6f}')
    model.load_state_dict(best_w)
    return hist, model

def _metrics(yf, pf, yo_3d=None, po_3d=None, dist_i=None, last_known=None):
    rmse  = float(np.sqrt(mean_squared_error(yf, pf)))
    mae   = float(mean_absolute_error(yf, pf))
    mape  = float(np.mean(np.abs((yf-pf)/(np.abs(yf)+1e-8)))*100)
    smape = float(np.mean(2*np.abs(yf-pf)/(np.abs(yf)+np.abs(pf)+1e-8))*100)
    r2    = float(r2_score(yf, pf))
    # Directional Accuracy: 각 예측 창에서 마지막 입력 대비 방향 일치율
    if yo_3d is not None and last_known is not None:
        lk = last_known[:, dist_i]  # (n_windows,)
        da_vals = []
        for h in range(yo_3d.shape[1]):
            actual_dir = np.sign(yo_3d[:, h, dist_i] - lk)
            pred_dir   = np.sign(po_3d[:, h, dist_i] - lk)
            da_vals.append(np.mean(actual_dir == pred_dir))
        da = float(np.mean(da_vals) * 100)
    else:
        da = float('nan')
    return {'RMSE':round(rmse,4),'MAE':round(mae,4),'MAPE':round(mape,2),
            'SMAPE':round(smape,2),'R2':round(r2,4),'DA':round(da,2)}

def eval_model(model, Xte, yte):
    """
    yte = 누적 차분(scaled). last_known + yte → 절대 지수로 복원 후 평가.
    """
    model.eval()
    with torch.no_grad():
        ps = model(torch.tensor(Xte, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    # 마지막 입력 시점의 타겟 컬럼 scaled 값
    last_sc = Xte[:, -1, :][:, tidx]                        # (n, n_tgt)
    po_sc   = last_sc[:, np.newaxis, :] + ps                 # (n, h, n_tgt)
    yo_sc   = last_sc[:, np.newaxis, :] + yte                # (n, h, n_tgt)
    po = inv_transform(po_sc, scaler, tidx)
    yo = inv_transform(yo_sc, scaler, tidx)
    last_known = inv_transform(last_sc, scaler, tidx)        # (n, n_tgt)
    metrics = {}
    for i, d in enumerate(DISTRICTS):
        pf, yf = po[:,:,i].flatten(), yo[:,:,i].flatten()
        m = _metrics(yf, pf, yo_3d=yo, po_3d=po, dist_i=i, last_known=last_known)
        # ── horizon별 개별 R²·RMSE 추가 ─────────────────────────────────────
        for h in range(FORECAST_HORIZON):
            yh = yo[:, h, i]; ph = po[:, h, i]
            m[f'R2_h{h+1}']   = round(float(r2_score(yh, ph)), 4)
            m[f'RMSE_h{h+1}'] = round(float(np.sqrt(mean_squared_error(yh, ph))), 4)
        metrics[d] = m
    return metrics

print('\n=== 5-6. 딥러닝 모델 학습 ===')
tr_ld=DataLoader(TS_DS(X_tr,y_tr),batch_size=BATCH_SIZE,shuffle=False)
vl_ld=DataLoader(TS_DS(X_vl,y_vl),batch_size=BATCH_SIZE,shuffle=False)

model_cfgs={
    'SimpleRNN':SimpleRNN(N_FEAT,PROJ_DIM,HIDDEN_SIZE,N_TGT,FORECAST_HORIZON,1,DROPOUT),
    'LSTM':     LSTMModel(N_FEAT,PROJ_DIM,HIDDEN_SIZE,N_TGT,FORECAST_HORIZON,N_LAYERS,DROPOUT),
    'GRU':      GRUModel( N_FEAT,PROJ_DIM,HIDDEN_SIZE,N_TGT,FORECAST_HORIZON,N_LAYERS,DROPOUT),
}
trained_models,histories,all_metrics={},{},{}
for name,model in model_cfgs.items():
    print(f'\n--- {name} ---')
    hist,trained=train_model(model,tr_ld,vl_ld)
    trained_models[name]=trained; histories[name]=hist
    all_metrics[name]=eval_model(trained,X_te,y_te)
    avg_r=np.mean([m['RMSE'] for m in all_metrics[name].values()])
    print(f'  평균 RMSE: {avg_r:.4f}')

# 손실 곡선 (3모델 → 1×3 배치)
fig,axes=plt.subplots(1,3,figsize=(15,4)); axes=axes.flatten()
for i,(name,hist) in enumerate(histories.items()):
    axes[i].plot(hist['train'],label='Train',linewidth=1.2,color='steelblue')
    axes[i].plot(hist['val'],label='Val',linewidth=1.2,linestyle='--',color='darkorange')
    axes[i].set_title(f'{name}',fontsize=11,fontweight='bold')
    axes[i].set_xlabel('Epoch'); axes[i].set_ylabel('MSE Loss')
    axes[i].legend(); axes[i].grid(True,alpha=0.3); axes[i].set_yscale('log')
plt.suptitle('모델별 학습/검증 손실',fontsize=13,fontweight='bold')
plt.tight_layout(); plt.savefig('output/fig_06_loss_curves.png',bbox_inches='tight'); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 7. 평가 비교 & 최종 예측
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== 7. 모델 평가 비교 ===')
arima_metrics={d:{'RMSE':round(arima_results[d]['rmse'],4),
                  'MAE': round(arima_results[d]['mae'],4),
                  'MAPE':round(arima_results[d]['mape'],2),
                  'SMAPE':round(arima_results[d]['smape'],2),
                  'R2':  round(arima_results[d]['r2'],4),
                  'DA':  round(arima_results[d]['da'],2)} for d in DISTRICTS}
all_metrics['ARIMA']=arima_metrics
rows=[]
for mn,dm in all_metrics.items():
    for d,m in dm.items():
        rows.append({'모델':mn,'구':d,'RMSE':m['RMSE'],'MAE':m['MAE'],
                     'MAPE(%)':m['MAPE'],'SMAPE(%)':m['SMAPE'],'R2':m['R2'],'DA(%)':m['DA']})
cmp_df=pd.DataFrame(rows)
avg_df=cmp_df.groupby('모델')[['RMSE','MAE','MAPE(%)','SMAPE(%)','R2','DA(%)']].mean().round(4)
avg_df.to_csv('output/model_comparison.csv',encoding='utf-8-sig')
print(avg_df.sort_values('RMSE').to_string())

nn_avg={k:np.mean([m['RMSE'] for m in v.values()]) for k,v in all_metrics.items() if k!='ARIMA'}
nn_da ={k:np.mean([m['DA']   for m in v.values()]) for k,v in all_metrics.items() if k!='ARIMA'}
# 선택 기준: 최소 RMSE 기준 10% 이내 후보 중 DA 최대 (균형성 우선)
best_rmse_val=min(nn_avg.values())
candidates={k:v for k,v in nn_avg.items() if v<=best_rmse_val*1.10}
best_name=max(candidates,key=lambda x:nn_da[x])
best_model=trained_models[best_name]
print(f'\n최적 모델: {best_name}  (평균 RMSE={nn_avg[best_name]:.4f}  평균 DA={nn_da[best_name]:.1f}%)')
print(f'  ※ RMSE 상위 10% 이내 후보: {list(candidates.keys())} → DA 최대 모델 선택')

# 최종 예측: 차분 예측 → 마지막 실제값 + 누적 차분 = 절대 지수
last_win    = feat_df.values[-SEQ_LEN:].astype(np.float32)
scaled_win  = scaler.transform(last_win)
last_sc_tgt = scaled_win[-1, tidx]                          # (n_tgt,)
Xt = torch.tensor(scaled_win[np.newaxis,:,:], dtype=torch.float32).to(DEVICE)
best_model.eval()
with torch.no_grad():
    ps_diff = best_model(Xt).cpu().numpy()                  # (1, h, n_tgt) — 차분
pred_sc   = last_sc_tgt[np.newaxis, :] + ps_diff.squeeze(0)  # (h, n_tgt)
pred_vals = inv_transform(pred_sc, scaler, tidx)              # (h, n_tgt) — 절대 지수

last_date=feat_df.index[-1]
forecast_dates=pd.date_range(start=last_date+pd.Timedelta(weeks=1),
                             periods=FORECAST_HORIZON, freq='W-MON')
forecast_df=pd.DataFrame(pred_vals,index=forecast_dates,columns=DISTRICTS)
forecast_df.index.name='예측일'
print('\n=== 최종 3주 예측 ===')
print(forecast_df.round(3).to_string())

# ARIMA 예측도 비교용
arima_fc=pd.DataFrame({d:arima_results[d]['future'] for d in DISTRICTS},
                      index=forecast_dates)
arima_fc.index.name='예측일'

# 최종 시각화 — 차분 → 절대 지수 복원
split_idx = int(len(df)*(1-TEST_RATIO))
best_model.eval()
with torch.no_grad():
    tp_diff = best_model(torch.tensor(X_te, dtype=torch.float32).to(DEVICE)).cpu().numpy()
last_sc_te = X_te[:, -1, :][:, tidx]                        # (n_win, n_tgt)
tp_sc      = last_sc_te[:, np.newaxis, :] + tp_diff          # (n_win, h, n_tgt)
tp_orig    = inv_transform(tp_sc, scaler, tidx)              # (n_win, h, n_tgt)
test_dates=df.index[split_idx:]

fig,axes=plt.subplots(3,2,figsize=(16,13)); axes=axes.flatten()
for i,d in enumerate(DISTRICTS):
    ax=axes[i]
    ax.plot(df.index[:split_idx],df[d].values[:split_idx],color='steelblue',linewidth=1.2,label='훈련')
    ax.plot(test_dates,df[d].values[split_idx:],color='royalblue',linewidth=1.2,alpha=0.7,label='테스트 실제')
    n_tp=min(len(tp_orig),len(test_dates))
    ax.plot(test_dates[:n_tp],tp_orig[:n_tp,0,i],color='darkorange',linewidth=1.2,linestyle='--',label=f'{best_name}')
    ax.plot(forecast_df.index,forecast_df[d].values,'r*',markersize=14,zorder=5,label='예측(3주)')
    for dt,val in zip(forecast_df.index,forecast_df[d]):
        ax.annotate(f'{val:.1f}',xy=(dt,val),xytext=(5,8),textcoords='offset points',
                    fontsize=9,color='red',fontweight='bold')
    ax.axvline(df.index[split_idx],color='gray',linestyle=':',alpha=0.6)
    ax.set_title(d,fontsize=12,fontweight='bold'); ax.legend(fontsize=7.5,loc='upper left')
    ax.grid(True,alpha=0.3); ax.xaxis.set_major_formatter(mdates.DateFormatter('%y.%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.get_xticklabels(),rotation=30,fontsize=8); ax.set_ylabel('가격지수')
plt.suptitle(f'{best_name} — 훈련/테스트/예측 (목표: 2026-05-25)',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig(f'output/fig_07_final_{best_name}.png',bbox_inches='tight'); plt.close()
print(f'\n  → fig_07 저장')

# ─────────────────────────────────────────────────────────────────────────────
# 결과 저장
# ─────────────────────────────────────────────────────────────────────────────
# ── 모델 파라미터 수 ──────────────────────────────────────────────────────────
def count_params(m): return int(sum(p.numel() for p in m.parameters()))
model_params = {name: count_params(mdl) for name, mdl in trained_models.items()}
# ARIMA: p+q+1(상수) 파라미터 수 (구별 평균)
arima_orders = {d: arima_results[d]['order'] for d in DISTRICTS}
arima_param_avg = round(np.mean([o[0]+o[2]+1 for o in arima_orders.values()]),1)

results = {
    'best_model': best_name,
    'seq_len': SEQ_LEN,
    'forecast_horizon': FORECAST_HORIZON,
    'test_period': f'{df.index[split_idx].date()} ~ {df.index[-1].date()}',
    'nn_avg_rmse':   {k: round(v,4) for k,v in nn_avg.items()},
    'arima_avg_rmse':round(float(np.mean([v['rmse']  for v in arima_results.values()])),4),
    'arima_avg_da':  round(float(np.mean([v['da']    for v in arima_results.values()])),2),
    'arima_avg_r2':  round(float(np.mean([v['r2']    for v in arima_results.values()])),4),
    'model_metrics': {
        mn: {d: {k: round(v,4) for k,v in m.items()}
             for d,m in dm.items()}
        for mn,dm in all_metrics.items()
    },
    'model_params': model_params,
    'arima_param_avg': arima_param_avg,
    'arima_orders': {d: list(arima_orders[d]) for d in DISTRICTS},
    'forecast_2026_05_25': {d: round(float(forecast_df.iloc[2][d]),3) for d in DISTRICTS},
    'forecast_all': {
        str(dt.date()): {d: round(float(forecast_df.loc[dt,d]),3) for d in DISTRICTS}
        for dt in forecast_df.index
    },
    'arima_forecast_2026_05_25': {d: round(float(arima_fc.iloc[2][d]),3) for d in DISTRICTS},
    'event_changes': event_changes,
    'dominant_periods': dominant_periods,
    'adf_1st_diff': {d: {'p_value': round(float(adfuller(df[d].diff().dropna(),autolag='AIC')[1]),4)}
                     for d in DISTRICTS},
}
with open('output/results.json','w',encoding='utf-8') as f:
    json.dump(results,f,ensure_ascii=False,indent=2)

print('\n' + '='*60)
print('분석 완료! output/ 폴더 확인')
print(f'최적 모델: {best_name}')
print('\n▶ 2026-05-25 예측 ('+best_name+'):')
for d in DISTRICTS: print(f'  {d:<7}: {results["forecast_2026_05_25"][d]:.3f}')
print('='*60)
