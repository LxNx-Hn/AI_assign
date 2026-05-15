"""
직관적인 시각화 차트 생성
"""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

import os; os.makedirs('output', exist_ok=True)

DISTRICTS = ['노원구','은평구','서대문구','서초구','강남구','송파구']
COLORS    = ['#4E79A7','#59A14F','#EDC948','#E15759','#B07AA1','#FF9DA7']

df = pd.read_csv('data/kb_apartment_index.csv', index_col='날짜', parse_dates=True)
df = df[DISTRICTS]
ret = df.pct_change().dropna() * 100   # 주간 변화율 %

# ══════════════════════════════════════════════════════════════════════════════
# V1. 핵심 스토리 차트 — 가격 흐름 + 사건 표시 (크고 깔끔)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))

# 배경 음영 (체제별)
ax.axvspan(pd.Timestamp('2021-05-01'), pd.Timestamp('2022-07-01'),
           alpha=0.12, color='#FFA500', label='_')
ax.axvspan(pd.Timestamp('2022-07-01'), pd.Timestamp('2024-01-01'),
           alpha=0.12, color='#FF4444', label='_')
ax.axvspan(pd.Timestamp('2024-01-01'), pd.Timestamp('2026-06-01'),
           alpha=0.12, color='#44BB44', label='_')

# 각 구 선
for i, d in enumerate(DISTRICTS):
    lw = 2.5 if d in ['강남구','노원구'] else 1.2
    alpha = 1.0 if d in ['강남구','노원구'] else 0.6
    ax.plot(df.index, df[d], color=COLORS[i], linewidth=lw, alpha=alpha, label=d)

# 이벤트 화살표
events = [
    ('2022-01-14', '금리 인상\n시작', 0.96, 'top'),
    ('2022-07-01', '금리\n2.5%', 0.85, 'top'),
    ('2023-01-13', '금리\n3.5%\n(최고)', 0.72, 'top'),
    ('2024-03-01', '금리\n인하 기대', 0.28, 'bottom'),
]
ymin, ymax = ax.get_ylim()
yrange = ymax - ymin
for edate, label, ypos, side in events:
    ed = pd.Timestamp(edate)
    y = ymin + yrange * ypos
    ax.axvline(ed, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    va = 'bottom' if side == 'bottom' else 'top'
    ax.text(ed, y, label, ha='center', va=va, fontsize=8,
            color='#333333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

# 체제 라벨
ax.text(pd.Timestamp('2021-09-01'), ax.get_ylim()[1]*0.99,
        '① 급등기', ha='center', fontsize=11, fontweight='bold', color='#CC7700', va='top')
ax.text(pd.Timestamp('2023-03-01'), ax.get_ylim()[1]*0.99,
        '② 급락기', ha='center', fontsize=11, fontweight='bold', color='#CC0000', va='top')
ax.text(pd.Timestamp('2025-03-01'), ax.get_ylim()[1]*0.99,
        '③ 회복기', ha='center', fontsize=11, fontweight='bold', color='#007700', va='top')

ax.legend(loc='lower left', fontsize=9, ncol=3, framealpha=0.9)
ax.set_ylabel('매매가격지수', fontsize=11)
ax.set_title('서울 6개 구 아파트 매매가격지수 (2021~2026) — 급등·급락·회복', fontsize=13, fontweight='bold', pad=12)
ax.grid(True, alpha=0.25, linestyle='-')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.setp(ax.get_xticklabels(), rotation=30, fontsize=9)
plt.tight_layout()
plt.savefig('output/vis_01_story.png', bbox_inches='tight')
plt.close()
print('vis_01 저장')

# ══════════════════════════════════════════════════════════════════════════════
# V2. 이벤트 충격 전후 비교 — 막대 차트 (가장 직관적)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('이벤트 전후 주간 가격 변화율(%) 비교', fontsize=13, fontweight='bold')

event_scenarios = [
    ('금리 인상 시작\n(2022-01-14)', '2022-01-14', 16),
    ('금리 급등 시작\n(2022-07-13)', '2022-07-13', 16),
    ('회복 전환\n(2024-01-25)',  '2024-01-25', 20),
]
for ai, (title, edate_str, weeks) in enumerate(event_scenarios):
    ax = axes[ai]
    ed = pd.Timestamp(edate_str)
    win = pd.Timedelta(weeks=weeks)
    befores, afters = [], []
    for d in DISTRICTS:
        b = ret[d].loc[(ret.index >= ed-win) & (ret.index < ed)].mean()
        a = ret[d].loc[(ret.index >= ed) & (ret.index < ed+win)].mean()
        befores.append(b); afters.append(a)

    x = np.arange(len(DISTRICTS))
    w = 0.35
    bars_b = ax.bar(x - w/2, befores, w, label='이전', color='#4E79A7', alpha=0.85)
    bars_a = ax.bar(x + w/2, afters,  w, label='이후', color='#E15759', alpha=0.85)

    for bar in bars_b:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h + (0.005 if h>=0 else -0.018),
                f'{h:.3f}', ha='center', va='bottom' if h>=0 else 'top', fontsize=7.5)
    for bar in bars_a:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h + (0.005 if h>=0 else -0.018),
                f'{h:.3f}', ha='center', va='bottom' if h>=0 else 'top', fontsize=7.5)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(DISTRICTS, fontsize=9, rotation=20)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel('주간 평균 변화율 (%)' if ai==0 else '', fontsize=9)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis='y')

plt.tight_layout()
plt.savefig('output/vis_02_event_impact.png', bbox_inches='tight')
plt.close()
print('vis_02 저장')

# ══════════════════════════════════════════════════════════════════════════════
# V3. 주기성 — 월별 평균 변화율 히트맵 (시각적으로 계절성 확인)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('주기성 분석 — "특정 시기마다 반복되는 패턴이 있는가?"', fontsize=13, fontweight='bold')

# 왼쪽: 연도별·월별 히트맵 (노원구 예시 → 서울 평균)
avg_ret = ret.mean(axis=1)
pivot = pd.DataFrame({'year': avg_ret.index.year, 'month': avg_ret.index.month, 'val': avg_ret.values})
heat = pivot.groupby(['year','month'])['val'].mean().unstack(fill_value=np.nan)
im = axes[0].imshow(heat.values, aspect='auto', cmap='RdYlGn', vmin=-0.5, vmax=0.5)
axes[0].set_xticks(range(12)); axes[0].set_xticklabels(['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월'], fontsize=8)
axes[0].set_yticks(range(len(heat.index))); axes[0].set_yticklabels([str(y) for y in heat.index], fontsize=9)
axes[0].set_title('연도×월별 주간 변화율\n(초록=상승 / 빨강=하락)', fontsize=10, fontweight='bold')
plt.colorbar(im, ax=axes[0], label='평균 주간 변화율(%)')
# 값 표시
for i in range(len(heat.index)):
    for j in range(12):
        v = heat.values[i,j]
        if not np.isnan(v):
            axes[0].text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.5,
                        color='white' if abs(v)>0.25 else 'black')
axes[0].set_xlabel('월', fontsize=9); axes[0].set_ylabel('연도', fontsize=9)

# 오른쪽: 월별 평균 변화율 막대 (전체 기간)
monthly_avg = avg_ret.groupby(avg_ret.index.month).mean()
monthly_std = avg_ret.groupby(avg_ret.index.month).std()
month_labels = ['1','2','3','4','5','6','7','8','9','10','11','12']
bar_colors = ['#E15759' if v < 0 else '#4E79A7' for v in monthly_avg]
axes[1].bar(range(12), monthly_avg.values, color=bar_colors, alpha=0.85, width=0.6)
axes[1].errorbar(range(12), monthly_avg.values, yerr=monthly_std.values,
                fmt='none', color='gray', capsize=4, linewidth=1.2)
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_xticks(range(12)); axes[1].set_xticklabels(month_labels, fontsize=9)
axes[1].set_xlabel('월', fontsize=9); axes[1].set_ylabel('평균 주간 변화율 (%)', fontsize=9)
axes[1].set_title('월별 평균 상승/하락률 (전체 기간)\n→ 특정 월에 반복 패턴 있으면 막대가 일정해야 함', fontsize=10, fontweight='bold')
axes[1].grid(True, alpha=0.25, axis='y')

plt.tight_layout()
plt.savefig('output/vis_03_seasonality.png', bbox_inches='tight')
plt.close()
print('vis_03 저장')

# ══════════════════════════════════════════════════════════════════════════════
# V4. 체제별 상승/하락 폭 비교 — 가로 막대
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
regimes = {
    '급등기\n(21.05~22.06)': (pd.Timestamp('2021-05-01'), pd.Timestamp('2022-07-01'), '#FFA500'),
    '급락기\n(22.07~23.12)': (pd.Timestamp('2022-07-01'), pd.Timestamp('2024-01-01'), '#E15759'),
    '회복기\n(24.01~26.05)': (pd.Timestamp('2024-01-01'), pd.Timestamp('2026-06-01'), '#59A14F'),
}
n_d = len(DISTRICTS); n_r = len(regimes)
group_h = 0.22; gap = 0.08
y_positions = {}
base_y = 0
for rname, (rs, re, rc) in regimes.items():
    ys = [base_y + i*group_h for i in range(n_d)]
    y_positions[rname] = ys
    base_y += n_d*group_h + gap

for ri, (rname, (rs, re, rc)) in enumerate(regimes.items()):
    ys = y_positions[rname]
    for di, d in enumerate(DISTRICTS):
        sub = ret[d].loc[(ret.index>=rs)&(ret.index<re)]
        cum = (1 + sub/100).prod() - 1   # 누적 변화율
        cum_pct = cum * 100
        color = rc if cum_pct >= 0 else '#E15759'
        bar = ax.barh(ys[di], cum_pct, height=group_h*0.8,
                      color=color, alpha=0.85)
        ax.text(cum_pct + (0.3 if cum_pct>=0 else -0.3), ys[di],
                f'{cum_pct:+.1f}%', va='center',
                ha='left' if cum_pct>=0 else 'right', fontsize=9, fontweight='bold')

# y축 라벨
all_ys = [y for ys in y_positions.values() for y in ys]
all_labels = [d for _ in range(n_r) for d in DISTRICTS]
ax.set_yticks(all_ys); ax.set_yticklabels(all_labels, fontsize=9)

# 체제 라벨 (그룹 중앙)
for rname, (rs, re, rc) in regimes.items():
    ys = y_positions[rname]
    mid_y = (ys[0] + ys[-1]) / 2
    ax.text(-13.5, mid_y, rname, va='center', ha='right', fontsize=10,
            fontweight='bold', color=rc,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=rc, alpha=0.8))

ax.axvline(0, color='black', linewidth=1.0)
ax.set_xlabel('누적 가격 변화율 (%)', fontsize=10)
ax.set_title('시기별 · 지역별 누적 가격 변화율\n(같은 이벤트에도 지역마다 반응이 다르다)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2, axis='x')
ax.set_xlim(-14, 20)
plt.tight_layout()
plt.savefig('output/vis_04_regime_bar.png', bbox_inches='tight')
plt.close()
print('vis_04 저장')

# ══════════════════════════════════════════════════════════════════════════════
# V5. 강남 선행 효과 — 직관적 시각화
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('강남 vs 외곽 — 강남이 먼저 움직이는가?', fontsize=13, fontweight='bold')

# 왼쪽: 강남 vs 노원 주간 변화율 비교
r_gn = ret['강남구']; r_nw = ret['노원구']
ax = axes[0]
ax.plot(r_gn.index, r_gn.rolling(8).mean(), color='#E15759', linewidth=2.0, label='강남구 (8주 이동평균)')
ax.plot(r_nw.index, r_nw.rolling(8).mean(), color='#4E79A7', linewidth=2.0, label='노원구 (8주 이동평균)', linestyle='--')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('주간 변화율 (%) — 이동평균', fontsize=9)
ax.set_title('강남구 vs 노원구 주간 변화율 추이\n→ 방향 전환 시 강남이 먼저 보임', fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax.get_xticklabels(), rotation=30, fontsize=8)

# 오른쪽: lag별 상관계수 (강남 → 노원, 1~8주 선행)
ax2 = axes[1]
lags = range(0, 9)
corrs_gn_nw = [r_gn.shift(lag).corr(r_nw) for lag in lags]
corrs_sc_nw = [ret['서초구'].shift(lag).corr(r_nw) for lag in lags]
ax2.plot(lags, corrs_gn_nw, 'o-', color='#E15759', linewidth=2, markersize=7, label='강남→노원 상관')
ax2.plot(lags, corrs_sc_nw, 's--', color='#B07AA1', linewidth=2, markersize=7, label='서초→노원 상관')
ax2.axvline(1, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
ax2.set_xlabel('선행 주수 (강남이 N주 앞설 때)', fontsize=9)
ax2.set_ylabel('상관계수', fontsize=9)
ax2.set_title('강남·서초 선행 주수 vs 노원 상관계수\n→ lag=1에서 상관이 가장 높음', fontsize=10, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25)
ax2.set_xticks(list(lags))
ax2.set_xticklabels([f'{l}주\n선행' for l in lags], fontsize=8)

plt.tight_layout()
plt.savefig('output/vis_05_lead_lag.png', bbox_inches='tight')
plt.close()
print('vis_05 저장')

# ══════════════════════════════════════════════════════════════════════════════
# V6. 모델 성능 비교 — 시각적 막대
# ══════════════════════════════════════════════════════════════════════════════
import json
with open('output/results.json', encoding='utf-8') as f: R = json.load(f)
DISTRICTS2 = ['노원구','은평구','서대문구','서초구','강남구','송파구']
mm = R['model_metrics']
MODELS = ['ARIMA','SimpleRNN','LSTM','GRU','BiLSTM']
avg_rmse = {mn: np.mean([mm[mn][d]['RMSE'] for d in DISTRICTS2]) for mn in MODELS}
avg_mape = {mn: np.mean([mm[mn][d]['MAPE'] for d in DISTRICTS2]) for mn in MODELS}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('모델 성능 비교 — 낮을수록 좋음', fontsize=13, fontweight='bold')

model_colors = ['#FFA500','#4E79A7','#59A14F','#EDC948','#B07AA1']
labels = ['ARIMA\n(통계기준)','SimpleRNN','LSTM','GRU','BiLSTM']

for ax, metric, values, unit in [
    (axes[0], 'RMSE', avg_rmse, '지수 포인트'),
    (axes[1], 'MAPE', avg_mape, '%'),
]:
    vals = [values[m] for m in MODELS]
    best_idx = int(np.argmin(vals))
    bars = ax.bar(labels, vals, color=model_colors, alpha=0.85, width=0.55, edgecolor='white', linewidth=1.5)
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(bar.get_x()+bar.get_width()/2, v + max(vals)*0.02,
                f'{v:.3f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold' if i==best_idx else 'normal',
                color='#CC0000' if i==best_idx else '#333333')
    # 최솟값 표시
    bars[best_idx].set_edgecolor('#CC0000')
    bars[best_idx].set_linewidth(3)
    ax.set_ylabel(f'{metric} ({unit})', fontsize=10)
    ax.set_title(f'{metric} 비교\n(빨간 테두리 = 최우수)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25, axis='y')
    ax.set_ylim(0, max(vals)*1.18)

plt.tight_layout()
plt.savefig('output/vis_06_model_compare.png', bbox_inches='tight')
plt.close()
print('vis_06 저장')

# ══════════════════════════════════════════════════════════════════════════════
# V7. 충격반응분석 (VAR + IRF) — 강남 충격이 다른 구에 전파되는 과정
# ══════════════════════════════════════════════════════════════════════════════
from statsmodels.tsa.vector_ar.var_model import VAR

# 1차 차분 (VAR은 정상 시계열 필요)
df_diff = df.diff().dropna()

# VAR 모델 적합 (AIC 기준 최적 시차 선택, 최대 8주)
model  = VAR(df_diff[DISTRICTS])
result = model.fit(maxlags=8, ic='aic')
lag_k  = result.k_ar
print(f'VAR 선택 시차: {lag_k}주')

# 12주 충격반응 계산
irf_obj = result.irf(12)
# irf_obj.orth_irfs shape: (periods+1, n_vars, n_vars)
# [기간, 반응변수, 충격변수]
orth_irf = irf_obj.orth_irfs   # (13, 6, 6)
lower_ci = irf_obj.cum_effect_stderr(orth=True)  # 표준오차

# 충격변수 인덱스
gn_idx = DISTRICTS.index('강남구')
sc_idx = DISTRICTS.index('서초구')
nw_idx = DISTRICTS.index('노원구')
ew_idx = DISTRICTS.index('은평구')
sd_idx = DISTRICTS.index('서대문구')
sp_idx = DISTRICTS.index('송파구')

periods = np.arange(13)  # 0~12주

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('충격반응분석 (VAR-IRF) — 한 지역에 가격 충격이 왔을 때 다른 지역의 반응',
             fontsize=13, fontweight='bold')

# ── 왼쪽: 강남구 충격 → 모든 구 반응
ax = axes[0]
colors_resp = ['#4E79A7','#59A14F','#EDC948','#E15759','#B07AA1','#FF9DA7']
for i, d in enumerate(DISTRICTS):
    resp = orth_irf[:, i, gn_idx]
    lw   = 2.5 if d in ['강남구','노원구'] else 1.5
    ls   = '-'  if d in ['강남구','노원구'] else '--'
    alpha = 1.0 if d in ['강남구','노원구'] else 0.75
    ax.plot(periods, resp, color=colors_resp[i], linewidth=lw,
            linestyle=ls, alpha=alpha, label=d, marker='o', markersize=4)

ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
ax.axvline(1, color='gray',  linewidth=1.0, linestyle=':', alpha=0.6)
ax.text(1.15, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.001,
        '1주 후\n반응 최대', fontsize=8.5, color='gray', va='top')

ax.set_xlabel('충격 후 경과 주수', fontsize=10)
ax.set_ylabel('가격지수 변화 (표준화)', fontsize=10)
ax.set_title('강남구에 가격 충격 → 각 구의 반응\n→ 노원·은평·서대문이 1~2주 후 따라 반응',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.set_xticks(periods)
ax.set_xticklabels([f'{p}주' for p in periods], fontsize=8)
ax.grid(True, alpha=0.25)

# ── 오른쪽: 노원구 시각에서 — 강남·서초·서대문 충격 각각 받았을 때
ax2 = axes[1]
shock_sources = [
    ('강남구', gn_idx, '#E15759', '-',  2.5),
    ('서초구', sc_idx, '#B07AA1', '--', 2.0),
    ('서대문구', sd_idx, '#4E79A7', ':',  1.8),
]
for name, sidx, color, ls, lw in shock_sources:
    resp = orth_irf[:, nw_idx, sidx]
    ax2.plot(periods, resp, color=color, linewidth=lw, linestyle=ls,
             marker='o', markersize=4, label=f'{name} 충격 → 노원구 반응')

ax2.axhline(0, color='black', linewidth=0.8)
ax2.axvline(1, color='gray', linewidth=1.0, linestyle=':', alpha=0.6)
ax2.set_xlabel('충격 후 경과 주수', fontsize=10)
ax2.set_ylabel('노원구 가격지수 변화 (표준화)', fontsize=10)
ax2.set_title('노원구는 어느 지역 충격에 가장 민감하게 반응하는가?\n→ 강남 충격 반응이 가장 크고 빠름',
              fontsize=10, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_xticks(periods)
ax2.set_xticklabels([f'{p}주' for p in periods], fontsize=8)
ax2.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('output/vis_07_irf.png', bbox_inches='tight')
plt.close()
print('vis_07 저장')

print('\n모든 시각화 완료!')
