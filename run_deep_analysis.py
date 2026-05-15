"""
KB Land 주간 아파트 매매가격지수 — 심화 데이터 특성 분석
- 정상성(ADF+KPSS+ZA), 자기상관(Ljung-Box), 구조 변화, 변동성 군집(ARCH),
  장기기억(Hurst), 정규성, Granger 인과, 구간별 체제, 주기성 원인 구분
"""
import warnings; warnings.filterwarnings('ignore')
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import detrend as sp_detrend
import statsmodels.api as sm
from statsmodels.tsa.stattools import (adfuller, kpss, zivot_andrews,
                                        acf, pacf, grangercausalitytests,
                                        coint)
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.seasonal import STL

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

import os; os.makedirs('output', exist_ok=True)

DISTRICTS = ['노원구','은평구','서대문구','서초구','강남구','송파구']
GANGNAM   = ['강남구','서초구','송파구']
OUTLYING  = ['노원구','은평구','서대문구']

# 이벤트 (날짜, 라벨, 색)
EVENTS = [
    ('2022-01-14', '기준금리 인상 시작\n(0.75→1.0%)',  'red'),
    ('2022-07-13', '금리 2.5%',                        'red'),
    ('2023-01-13', '금리 3.5%(최고)',                   'darkred'),
    ('2024-01-25', '금리 동결→인하 기대',               'green'),
    ('2021-07-01', 'DSR 규제 강화',                    'orange'),
    ('2022-06-21', '6.21 부동산 대책',                  'purple'),
    ('2023-09-26', '특례보금자리론 종료',               'brown'),
]

df = pd.read_csv('data/kb_apartment_index.csv', index_col='날짜', parse_dates=True)
df = df[DISTRICTS]
n = len(df)
print(f'[데이터] {df.shape}  {df.index[0].date()} ~ {df.index[-1].date()}')

# 구간 정의 (체제)
REGIMES = {
    '급등기(21.05~22.06)': (pd.Timestamp('2021-05-01'), pd.Timestamp('2022-06-30')),
    '급락기(22.07~23.12)': (pd.Timestamp('2022-07-01'), pd.Timestamp('2023-12-31')),
    '회복기(24.01~26.05)': (pd.Timestamp('2024-01-01'), pd.Timestamp('2026-06-01')),
}
REGIME_COLORS = {'급등기(21.05~22.06)':'#FFA500',
                 '급락기(22.07~23.12)':'#FF4444',
                 '회복기(24.01~26.05)':'#44BB44'}

summary = {}  # 결과 저장용

# ══════════════════════════════════════════════════════════════════════════════
# A. 기술 통계 & 분포 특성
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== A. 기술 통계 및 분포 특성 ===')
ret = df.pct_change().dropna() * 100  # 주간 변화율 (%)

desc_rows = []
for d in DISTRICTS:
    s = df[d]
    r = ret[d]
    jb_stat, jb_p = stats.jarque_bera(r.dropna())
    sw_stat, sw_p = stats.shapiro(r.dropna()[:50])   # Shapiro는 n≤5000
    desc_rows.append({
        '구': d,
        '평균': round(float(s.mean()), 3),
        '표준편차': round(float(s.std()), 3),
        '최소': round(float(s.min()), 3),
        '최대': round(float(s.max()), 3),
        '왜도': round(float(stats.skew(s)), 3),
        '첨도': round(float(stats.kurtosis(s)), 3),
        '주간변화율평균(%)': round(float(r.mean()), 4),
        '주간변화율std(%)': round(float(r.std()), 4),
        'JB_p': round(jb_p, 4),
        '정규성(α=0.05)': '✓' if jb_p > 0.05 else '✗ 비정규',
    })
desc_df = pd.DataFrame(desc_rows)
print(desc_df.to_string(index=False))
summary['descriptive'] = desc_rows

# 그림 A1: 분포 히스토그램 + QQ plot
fig, axes = plt.subplots(2, 6, figsize=(20, 8))
for i, d in enumerate(DISTRICTS):
    r = ret[d].dropna()
    ax1 = axes[0, i]
    ax1.hist(r, bins=30, density=True, color='steelblue', alpha=0.7, edgecolor='white')
    xr = np.linspace(r.min(), r.max(), 100)
    ax1.plot(xr, stats.norm.pdf(xr, r.mean(), r.std()), 'r-', linewidth=1.5, label='정규')
    ax1.set_title(d, fontsize=10, fontweight='bold')
    ax1.set_xlabel('주간 변화율(%)', fontsize=8)
    if i == 0: ax1.set_ylabel('밀도', fontsize=8)
    ax1.legend(fontsize=7)
    ax2 = axes[1, i]
    stats.probplot(r, dist='norm', plot=ax2)
    ax2.set_title(f'{d} Q-Q', fontsize=9)
    ax2.set_xlabel('이론 분위수', fontsize=8); ax2.set_ylabel('표본 분위수', fontsize=8)
plt.suptitle('주간 변화율 분포 분석 (히스토그램 + Q-Q plot)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/deep_A1_distribution.png', bbox_inches='tight')
plt.close()
print('  → deep_A1 저장')

# ══════════════════════════════════════════════════════════════════════════════
# B. 정상성 심화 (ADF + KPSS + Zivot-Andrews)
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== B. 정상성 심화 검정 ===')
stationarity_rows = []
for d in DISTRICTS:
    s = df[d].dropna().values
    sd = np.diff(s)           # 1차 차분
    # ADF
    adf_s, adf_p = adfuller(s, autolag='AIC')[:2]
    adf_d, adf_dp = adfuller(sd, autolag='AIC')[:2]
    # KPSS (귀무가설: 정상)
    try:
        kpss_s, kpss_p = kpss(s, regression='ct', nlags='auto')[:2]
        kpss_d, kpss_dp = kpss(sd, regression='c', nlags='auto')[:2]
    except: kpss_s=kpss_p=kpss_d=kpss_dp=float('nan')
    # Zivot-Andrews (구조 단절 허용 단위근)
    try:
        za = zivot_andrews(s, maxlag=8, regression='ct', autolag=None)
        za_stat, za_p, za_bp = za[0], za[1], za[4]
        za_date = df.index[za_bp].date() if za_bp < len(df.index) else 'N/A'
    except: za_stat=za_p=float('nan'); za_date='N/A'
    # ADF 결론: ADF p>0.05 → 단위근 존재 (비정상)
    # KPSS 결론: KPSS p<0.05 → 비정상
    # 최종: ADF reject × KPSS 비정상 → "강한 비정상"
    adf_conclusion  = '비정상' if adf_p  > 0.05 else '정상'
    kpss_conclusion = '비정상' if kpss_p < 0.05 else '정상'
    za_conclusion   = '단위근없음(구조변화허용)' if (not np.isnan(za_p) and za_p < 0.05) else '단위근있음'
    stationarity_rows.append({
        '구': d, 'ADF_p(수준)': round(adf_p,4), 'KPSS_p(수준)': round(kpss_p,4) if not np.isnan(kpss_p) else 'N/A',
        'ADF': adf_conclusion, 'KPSS': kpss_conclusion,
        'ZA_p': round(za_p,4) if not np.isnan(za_p) else 'N/A', 'ZA_판정': za_conclusion, 'ZA_단절일': str(za_date),
        'ADF_1차차분_p': round(adf_dp,4), 'KPSS_1차차분_p': round(kpss_dp,4) if not np.isnan(kpss_dp) else 'N/A',
    })
stat_df = pd.DataFrame(stationarity_rows)
print(stat_df[['구','ADF_p(수준)','KPSS_p(수준)','ADF','KPSS','ZA_p','ZA_판정','ZA_단절일']].to_string(index=False))
summary['stationarity'] = stationarity_rows

# ══════════════════════════════════════════════════════════════════════════════
# C. 자기상관 심화 — Ljung-Box 검정
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== C. Ljung-Box 자기상관 검정 ===')
lb_rows = []
for d in DISTRICTS:
    r = ret[d].dropna()
    for lag in [4, 12, 26, 52]:
        res = acorr_ljungbox(r, lags=[lag], return_df=True)
        lb_rows.append({'구': d, 'lag': lag,
                        'LB_통계량': round(float(res['lb_stat'].iloc[-1]), 3),
                        'p_value': round(float(res['lb_pvalue'].iloc[-1]), 4),
                        '자기상관유의': '✓' if float(res['lb_pvalue'].iloc[-1]) < 0.05 else '✗'})
lb_df = pd.DataFrame(lb_rows)
print(lb_df.pivot(index='구', columns='lag', values='p_value').round(4).to_string())
summary['ljung_box'] = lb_rows

# ══════════════════════════════════════════════════════════════════════════════
# D. 변동성 군집 — ARCH 효과 검정
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== D. ARCH 효과(변동성 군집) 검정 ===')
arch_rows = []
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, d in enumerate(DISTRICTS):
    r = ret[d].dropna().values
    # ARCH-LM test
    arch_stat, arch_p, _, _ = het_arch(r, nlags=4)
    arch_rows.append({'구': d, 'ARCH_통계량': round(arch_stat, 3),
                      'p_value': round(arch_p, 4), 'ARCH효과': '있음' if arch_p < 0.05 else '없음'})
    # 롤링 변동성 시각화
    roll_std = pd.Series(r, index=ret.index).rolling(12).std()
    axes[i].plot(ret.index[:-1] if len(roll_std) != len(ret.index) else ret.index,
                 roll_std.values, color='steelblue', linewidth=1.2)
    for event_date, label, color in EVENTS:
        ed = pd.Timestamp(event_date)
        if df.index[0] <= ed <= df.index[-1]:
            axes[i].axvline(ed, color=color, linestyle='--', alpha=0.6, linewidth=1)
    for rname, (rs, re) in REGIMES.items():
        axes[i].axvspan(rs, re, alpha=0.07, color=REGIME_COLORS[rname])
    axes[i].set_title(f'{d}  (ARCH p={arch_p:.3f}{"★" if arch_p<0.05 else ""})',
                      fontsize=10, fontweight='bold')
    axes[i].set_ylabel('12주 롤링 표준편차', fontsize=8)
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%y.%m'))
    axes[i].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[i].get_xticklabels(), rotation=30, fontsize=7)
    axes[i].grid(True, alpha=0.3)
plt.suptitle('롤링 변동성 (12주) — ARCH 효과 검정', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/deep_D1_arch_volatility.png', bbox_inches='tight')
plt.close()
arch_df = pd.DataFrame(arch_rows)
print(arch_df.to_string(index=False))
summary['arch'] = arch_rows
print('  → deep_D1 저장')

# ══════════════════════════════════════════════════════════════════════════════
# E. 장기기억 — Hurst 지수
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== E. Hurst 지수 (장기기억) ===')

def hurst_rs(ts, max_lag=None):
    """Rescaled Range (R/S) 분석 — 1차 차분(변화율)에 적용"""
    ts = np.diff(np.array(ts, dtype=float))   # 수준이 아닌 변화량으로 계산
    if max_lag is None: max_lag = len(ts) // 2
    lags = range(10, max_lag)
    rs_vals = []
    for lag in lags:
        sub_ts = ts[:lag]
        mean_adj = sub_ts - np.mean(sub_ts)
        cum = np.cumsum(mean_adj)
        r_val = np.max(cum) - np.min(cum)
        s_val = np.std(sub_ts, ddof=1)
        if s_val > 0: rs_vals.append(r_val / s_val)
        else: rs_vals.append(np.nan)
    lags_arr  = np.array(list(lags))
    rs_arr    = np.array(rs_vals)
    valid     = ~np.isnan(rs_arr) & (rs_arr > 0)
    if valid.sum() < 5: return float('nan'), float('nan')
    slope, _, _, _, _ = stats.linregress(np.log(lags_arr[valid]), np.log(rs_arr[valid]))
    return slope, rs_arr

hurst_rows = []
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, d in enumerate(DISTRICTS):
    h, rs_arr = hurst_rs(df[d].values)
    interpretation = ('강한 추세 지속(H>0.7)' if h > 0.7 else
                      '약한 추세 지속(0.5<H≤0.7)' if h > 0.5 else
                      '랜덤워크(H≈0.5)' if h > 0.45 else
                      '평균회귀(H<0.5)')
    hurst_rows.append({'구': d, 'Hurst': round(h, 4), '해석': interpretation})
    # R/S plot
    lags_plot = np.arange(10, len(df) // 2)
    valid = ~np.isnan(rs_arr) & (rs_arr > 0)
    ax = axes[i]
    ax.scatter(np.log(lags_plot[valid[:len(lags_plot)]]),
               np.log(rs_arr[:len(lags_plot)][valid[:len(lags_plot)]]),
               s=5, alpha=0.5, color='steelblue')
    x_fit = np.array([np.log(lags_plot[valid].min()), np.log(lags_plot[valid].max())])
    intercept = np.mean(np.log(rs_arr[:len(lags_plot)][valid[:len(lags_plot)]]) -
                         h * np.log(lags_plot[valid[:len(lags_plot)]]))
    ax.plot(x_fit, h * x_fit + intercept, 'r-', linewidth=1.5, label=f'H={h:.3f}')
    ax.axline((0, 0), slope=0.5, color='gray', linestyle=':', linewidth=1, label='H=0.5(랜덤워크)')
    ax.set_title(f'{d}', fontsize=10, fontweight='bold')
    ax.set_xlabel('log(lag)', fontsize=8); ax.set_ylabel('log(R/S)', fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle('Hurst 지수 — R/S 분석 (장기기억)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/deep_E1_hurst.png', bbox_inches='tight')
plt.close()
hurst_df = pd.DataFrame(hurst_rows)
print(hurst_df.to_string(index=False))
summary['hurst'] = hurst_rows
print('  → deep_E1 저장')

# ══════════════════════════════════════════════════════════════════════════════
# F. 구조적 단절 — 이벤트 전후 분석 + 통계적 체제 비교
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== F. 구조적 체제 비교 ===')

# F1: 구간별 기술 통계
regime_stats = []
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, d in enumerate(DISTRICTS):
    ax = axes[i]
    colors = ['#FFA500','#FF4444','#44BB44']
    for j, (rname, (rs, re)) in enumerate(REGIMES.items()):
        sub = df[d].loc[(df.index >= rs) & (df.index < re)]
        sub_r = ret[d].loc[(ret.index >= rs) & (ret.index < re)]
        if len(sub) < 4: continue
        mu = float(sub_r.mean()); sigma = float(sub_r.std())
        trend_slope = float(np.polyfit(np.arange(len(sub)), sub.values, 1)[0])
        regime_stats.append({
            '구': d, '체제': rname,
            '주간변화율평균(%)': round(mu, 4), '주간변화율std(%)': round(sigma, 4),
            '주간추세(포인트/주)': round(trend_slope, 4),
            '기간중앙값': round(float(sub.median()), 3),
        })
        ax.plot(sub.index, sub.values, color=colors[j], linewidth=1.5,
                label=f'{rname.split("(")[0]}\n슬로프:{trend_slope:+.3f}/w')
    # 이벤트 표시
    for event_date, label, color in EVENTS:
        ed = pd.Timestamp(event_date)
        if df.index[0] <= ed <= df.index[-1]:
            ax.axvline(ed, color=color, linestyle='--', alpha=0.7, linewidth=0.8)
    ax.set_title(d, fontsize=11, fontweight='bold')
    ax.legend(fontsize=6.5, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y.%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=7)
    ax.set_ylabel('가격지수', fontsize=8)
plt.suptitle('구간별 체제 분석 (급등/급락/회복)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/deep_F1_regime.png', bbox_inches='tight')
plt.close()
reg_df = pd.DataFrame(regime_stats)
print(reg_df.to_string(index=False))
summary['regime_stats'] = regime_stats
print('  → deep_F1 저장')

# F2: 이벤트 전후 t-검정 (금리 인상 2022-01-14)
print('\n  [이벤트 전후 t-검정]')
event_tests = []
for d in DISTRICTS:
    for evt_date, evt_label in [
            ('2022-01-14', '금리인상'),
            ('2022-07-13', '금리급등'),
            ('2024-01-25', '회복전환')]:
        ed = pd.Timestamp(evt_date)
        win = pd.Timedelta(weeks=26)
        before = ret[d].loc[(ret.index >= ed - win) & (ret.index < ed)].dropna()
        after  = ret[d].loc[(ret.index >= ed) & (ret.index <  ed + win)].dropna()
        if len(before) < 5 or len(after) < 5: continue
        t_stat, t_p = stats.ttest_ind(before, after)
        event_tests.append({'구': d, '이벤트': evt_label,
                             '이전평균': round(float(before.mean()),4),
                             '이후평균': round(float(after.mean()),4),
                             't_p': round(t_p,4), '유의': '★' if t_p<0.05 else '-'})
ev_df = pd.DataFrame(event_tests)
print(ev_df[ev_df['유의']=='★'][['구','이벤트','이전평균','이후평균','t_p']].to_string(index=False))
summary['event_tests'] = event_tests

# ══════════════════════════════════════════════════════════════════════════════
# G. 주기성 심화 — 이벤트 vs 내재 주기 구분
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== G. 주기성 심화 분석 ===')

# G1: STL 분해 (robust, 계절 주기 4, 13, 26, 52주 각각 비교)
stl_periods = [4, 13, 26, 52]
seasonal_strengths = {d: [] for d in DISTRICTS}
residual_ratios    = {d: [] for d in DISTRICTS}

for period in stl_periods:
    for d in DISTRICTS:
        try:
            stl = STL(df[d], period=period, robust=True)
            res = stl.fit()
            var_seasonal = float(np.var(res.seasonal))
            var_resid    = float(np.var(res.resid))
            var_total    = float(np.var(df[d].values))
            # 계절성 강도 = 1 - Var(residual) / Var(seasonal + residual)
            fs = max(0, 1 - var_resid / (var_seasonal + var_resid + 1e-12))
            seasonal_strengths[d].append(round(fs, 4))
            residual_ratios[d].append(round(var_resid / var_total, 4))
        except: seasonal_strengths[d].append(float('nan'))

print('  [계절성 강도 (0=없음, 1=완전)] 주기별:')
ss_df = pd.DataFrame(seasonal_strengths, index=[f'{p}주' for p in stl_periods])
print(ss_df.to_string())
summary['seasonal_strength'] = {'periods': stl_periods,
                                 'data': {d: seasonal_strengths[d] for d in DISTRICTS}}

# G2: 이벤트 제거 후 FFT — 이벤트 제거 시 주기성이 사라지는가?
# 이벤트 기간(급락기)을 선형보간으로 채운 후 FFT
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
fft_comparison = {}

for i, d in enumerate(DISTRICTS):
    vals = df[d].values.astype(float)
    n_vals = len(vals)
    # 트렌드 제거
    detrended = sp_detrend(vals)
    # 원본 FFT
    freqs = fftfreq(n_vals, d=1)
    power = np.abs(fft(detrended))
    pos = freqs > 0; periods = 1.0 / freqs[pos]; pw = power[pos]
    mask = (periods >= 2) & (periods <= 65)
    # 이벤트 충격 제거: 급락기(22.07~23.12) 선형보간
    vals_noevent = vals.copy()
    rs_idx = np.searchsorted(df.index, pd.Timestamp('2022-07-01'))
    re_idx = np.searchsorted(df.index, pd.Timestamp('2024-01-01'))
    if re_idx > rs_idx:
        slope = (vals[re_idx] - vals[rs_idx]) / (re_idx - rs_idx)
        for k in range(rs_idx, min(re_idx, n_vals)):
            vals_noevent[k] = vals[rs_idx] + slope * (k - rs_idx)
    detrended_ne = sp_detrend(vals_noevent)
    power_ne = np.abs(fft(detrended_ne))
    pw_ne = power_ne[pos]

    # 상위 3 주기
    top_orig = np.argsort(pw[mask])[-1]
    top_ne   = np.argsort(pw_ne[mask])[-1]
    dom_orig = round(periods[mask][top_orig])
    dom_ne   = round(periods[mask][top_ne])
    fft_comparison[d] = {'dominant_original': dom_orig, 'dominant_no_event': dom_ne,
                         'same_peak': abs(dom_orig - dom_ne) < 5}

    ax = axes[i]
    ax.plot(periods[mask], pw[mask], color='steelblue', linewidth=1.2, label='원본')
    ax.plot(periods[mask], pw_ne[mask], color='darkorange', linewidth=1.2, linestyle='--', label='이벤트 제거')
    for ref_p, lbl in [(4,'4w'),(13,'13w'),(26,'26w'),(52,'52w')]:
        ax.axvline(ref_p, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax.set_title(f'{d}  (원본:{dom_orig}w / 이벤트제거:{dom_ne}w)', fontsize=9, fontweight='bold')
    ax.set_xlabel('주기(주)', fontsize=8); ax.set_ylabel('진폭', fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle('FFT 주기성 — 원본 vs 이벤트 충격 제거 후', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/deep_G1_fft_comparison.png', bbox_inches='tight')
plt.close()
print(f'  주기 비교: {fft_comparison}')
summary['fft_comparison'] = fft_comparison
print('  → deep_G1 저장')

# G3: 구간별 ACF (체제별 자기상관 변화)
fig, axes = plt.subplots(len(DISTRICTS), 3, figsize=(18, 3*len(DISTRICTS)))
acf_by_regime = {}
for i, d in enumerate(DISTRICTS):
    acf_by_regime[d] = {}
    for j, (rname, (rs, re)) in enumerate(REGIMES.items()):
        sub = df[d].loc[(df.index >= rs) & (df.index < re)]
        if len(sub) > 15:
            sub_d = np.diff(sub.values)
            acf_vals = acf(sub_d, nlags=min(12, len(sub_d)-2), fft=True)
            ci = 1.96 / np.sqrt(len(sub_d))
            ax = axes[i, j]
            ax.bar(range(1, len(acf_vals)), acf_vals[1:],
                   color=['steelblue' if abs(v)>ci else 'lightblue' for v in acf_vals[1:]])
            ax.axhline(ci, color='r', linestyle='--', linewidth=0.8)
            ax.axhline(-ci, color='r', linestyle='--', linewidth=0.8)
            ax.set_title(f'{d} — {rname.split("(")[0]}', fontsize=8)
            ax.set_ylim(-0.5, 0.8)
            n_sig = sum(1 for v in acf_vals[1:] if abs(v) > ci)
            acf_by_regime[d][rname] = {'n_significant_lags': n_sig}
plt.suptitle('체제별 ACF 비교 (구조 변화 전·후 자기상관 패턴)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/deep_G2_regime_acf.png', bbox_inches='tight')
plt.close()
summary['acf_by_regime'] = acf_by_regime
print('  → deep_G2 저장')

# ══════════════════════════════════════════════════════════════════════════════
# H. 지역 간 관계 — Granger 인과성 & 공적분
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== H. 지역 간 Granger 인과성 ===')
granger_results = []
ref_districts = ['강남구', '서초구']  # 선행 후보
target_districts = ['노원구','은평구','서대문구']

for cause in ref_districts:
    for effect in target_districts:
        data_g = pd.DataFrame({'cause': df[cause].diff().dropna(),
                               'effect': df[effect].diff().dropna()}).dropna()
        try:
            gc = grangercausalitytests(data_g[['effect','cause']], maxlag=4, verbose=False)
            best_p = min([gc[lag][0]['ssr_ftest'][1] for lag in range(1,5)])
            best_lag = min(range(1,5), key=lambda lag: gc[lag][0]['ssr_ftest'][1])
            granger_results.append({'원인': cause, '결과': effect,
                                    'F_p': round(best_p, 4), '최적_lag': best_lag,
                                    '선행관계': '★ 유의' if best_p < 0.05 else '-'})
        except: pass

# 공적분 (Engle-Granger)
print('\n  [공적분 검정 — 강남↔외곽]')
coint_results = []
for d1 in GANGNAM:
    for d2 in OUTLYING:
        try:
            t, p, crit = coint(df[d1], df[d2])
            coint_results.append({'구1': d1, '구2': d2, 'p_value': round(p,4),
                                  '공적분': '★ 있음' if p < 0.05 else '없음'})
        except: pass
gc_df = pd.DataFrame(granger_results)
co_df = pd.DataFrame(coint_results)
if len(gc_df): print(gc_df.to_string(index=False))
if len(co_df): print(co_df.to_string(index=False))
summary['granger'] = granger_results
summary['cointegration'] = coint_results

# H2: 교차상관 히트맵 (리드-래그)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# 수준 상관
corr_level = df.corr()
im1 = axes[0].imshow(corr_level.values, cmap='RdYlGn', vmin=0.9, vmax=1.0)
axes[0].set_xticks(range(len(DISTRICTS))); axes[0].set_xticklabels(DISTRICTS, rotation=45, fontsize=9)
axes[0].set_yticks(range(len(DISTRICTS))); axes[0].set_yticklabels(DISTRICTS, fontsize=9)
for ii in range(len(DISTRICTS)):
    for jj in range(len(DISTRICTS)):
        axes[0].text(jj, ii, f'{corr_level.values[ii,jj]:.3f}', ha='center', va='center', fontsize=8)
plt.colorbar(im1, ax=axes[0]); axes[0].set_title('수준 상관', fontsize=11, fontweight='bold')

# 2주 lag 상관 (강남이 선행하면 lag>0 상관이 높아야)
lag_corrs = pd.DataFrame(index=DISTRICTS, columns=DISTRICTS, dtype=float)
for d1 in DISTRICTS:
    for d2 in DISTRICTS:
        s1 = df[d1].diff().dropna().values
        s2 = df[d2].diff().dropna().values
        if d1 == d2: lag_corrs.loc[d1,d2] = 1.0; continue
        # d1 (t-2) vs d2 (t)의 상관: d1이 2주 선행?
        lag_corrs.loc[d1,d2] = float(np.corrcoef(s1[:-2], s2[2:])[0,1])
im2 = axes[1].imshow(lag_corrs.values.astype(float), cmap='RdYlGn', vmin=-0.1, vmax=0.5)
axes[1].set_xticks(range(len(DISTRICTS))); axes[1].set_xticklabels(DISTRICTS, rotation=45, fontsize=9)
axes[1].set_yticks(range(len(DISTRICTS))); axes[1].set_yticklabels(DISTRICTS, fontsize=9)
for ii in range(len(DISTRICTS)):
    for jj in range(len(DISTRICTS)):
        axes[1].text(jj, ii, f'{float(lag_corrs.values[ii,jj]):.3f}', ha='center', va='center', fontsize=8)
plt.colorbar(im2, ax=axes[1])
axes[1].set_title('2주 리드-래그 상관\n(행 구가 2주 선행)', fontsize=11, fontweight='bold')
plt.suptitle('지역 간 상관관계 분석', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/deep_H1_cross_corr.png', bbox_inches='tight')
plt.close()
print('  → deep_H1 저장')

# ══════════════════════════════════════════════════════════════════════════════
# I. 롤링 통계 — 체제 전환 시각화
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== I. 롤링 통계 시각화 ===')
fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4)

for i, d in enumerate(DISTRICTS):
    ax = fig.add_subplot(gs[i//2, i%2])
    s = df[d]
    roll_mean = s.rolling(12).mean()
    roll_std  = s.rolling(12).std()
    ax.plot(s.index, s.values, color='lightblue', linewidth=0.8, alpha=0.6, label='원시')
    ax.plot(roll_mean.index, roll_mean.values, color='steelblue', linewidth=1.8, label='12주 이동평균')
    ax2 = ax.twinx()
    ax2.fill_between(roll_std.index, roll_std.values, alpha=0.2, color='orange', label='12주 변동성')
    ax2.set_ylabel('변동성(표준편차)', fontsize=7, color='orange')
    ax2.tick_params(colors='orange', labelsize=7)
    for event_date, label, color in EVENTS[:4]:
        ed = pd.Timestamp(event_date)
        ax.axvline(ed, color=color, linestyle='--', alpha=0.7, linewidth=1.0)
    for rname, (rs, re) in REGIMES.items():
        ax.axvspan(rs, re, alpha=0.07, color=REGIME_COLORS[rname])
    ax.set_title(d, fontsize=11, fontweight='bold')
    ax.set_ylabel('가격지수', fontsize=8)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y.%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=7)
plt.suptitle('롤링 평균·변동성 — 체제 전환 및 이벤트', fontsize=14, fontweight='bold')
plt.savefig('output/deep_I1_rolling.png', bbox_inches='tight')
plt.close()
print('  → deep_I1 저장')

# ══════════════════════════════════════════════════════════════════════════════
# J. 종합 요약 출력
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('심화 분석 종합 결론')
print('='*70)

print('\n[정상성]')
for r in stationarity_rows:
    print(f'  {r["구"]}: ADF={r["ADF"]}(p={r["ADF_p(수준)"]}), '
          f'KPSS={r["KPSS"]}(p={r["KPSS_p(수준)"]}), '
          f'ZA={r["ZA_판정"]}(단절일={r["ZA_단절일"]})')

print('\n[장기기억 — Hurst 지수]')
for r in hurst_rows:
    print(f'  {r["구"]}: H={r["Hurst"]}  →  {r["해석"]}')

print('\n[ARCH 변동성 군집]')
for r in arch_rows:
    print(f'  {r["구"]}: ARCH p={r["p_value"]}  →  변동성 군집 {r["ARCH효과"]}')

print('\n[주기성 원인 — 이벤트 제거 전후]')
for d, v in fft_comparison.items():
    print(f'  {d}: 원본 주도 주기={v["dominant_original"]}w, '
          f'이벤트 제거 후={v["dominant_no_event"]}w, '
          f'{"동일(내재 주기)" if v["same_peak"] else "변화(이벤트 유발)"}')

print('\n[계절성 강도 요약]')
print(ss_df.to_string())

print('\n[Granger 인과 (유의한 것만)]')
for r in granger_results:
    if r['선행관계'] == '★ 유의':
        print(f'  {r["원인"]} → {r["결과"]}: p={r["F_p"]}, lag={r["최적_lag"]}주')

with open('output/deep_results.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print('\n결과 저장: output/deep_results.json')
