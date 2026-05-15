"""
KB Land 주간 아파트 매매가격지수 예측 보고서
본문 A4 3쪽 이내 / 개조식 / 참고문헌 별도
"""
import json, os
import numpy as np
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

with open('output/results.json', encoding='utf-8') as f:
    R = json.load(f)

DISTRICTS    = ['노원구', '은평구', '서대문구', '서초구', '강남구', '송파구']
MODELS_ORDER = ['ARIMA', 'SimpleRNN', 'LSTM', 'GRU']
best_model   = R['best_model']          # 'ARIMA'

h3_rmse = R['h3_rmse']
h3_mae  = R['h3_mae']
h3_da   = R['h3_da']

fc_ar_t3  = R['arima_forecast_2026_05_25']
fc_ar_all = R.get('arima_forecast_all', {})
ar_dates  = sorted(fc_ar_all.keys())

arima_orders = R.get('arima_orders', {})
adf_data     = R.get('adf_1st_diff', {})
ev           = R.get('event_changes', {})
dp           = R.get('dominant_periods', {})
test_period  = R.get('test_period', '')

# ── 문서 기본 설정 ─────────────────────────────────────────────────────────
doc = Document()
for sec in doc.sections:
    sec.page_height   = Cm(29.7); sec.page_width    = Cm(21.0)
    sec.top_margin    = Cm(2.2);  sec.bottom_margin = Cm(2.2)
    sec.left_margin   = Cm(2.5);  sec.right_margin  = Cm(2.5)

# ── 헬퍼 함수 ──────────────────────────────────────────────────────────────
def _sf(run, size=10.5, bold=False, color=None, italic=False):
    run.font.name = '맑은 고딕'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    run.font.size = Pt(size); run.bold = bold; run.italic = italic
    if color: run.font.color.rgb = RGBColor(*color)

def h1(text):
    p = doc.add_heading(text, level=1)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for r in p.runs: _sf(r, 12, True, (0, 56, 117))
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after  = Pt(3)

def h2(text):
    p = doc.add_heading(text, level=2)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for r in p.runs: _sf(r, 10.5, True, (30, 100, 160))
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after  = Pt(2)

def bul(text, sub=False, sa=2):
    p = doc.add_paragraph()
    p.paragraph_format.space_after     = Pt(sa)
    p.paragraph_format.left_indent     = Cm(1.1 if sub else 0.4)
    p.paragraph_format.first_line_indent = Cm(-0.4)
    marker = '  -  ' if sub else '·  '
    r = p.add_run(marker + text)
    _sf(r, 9.5 if sub else 10.5)

def img(path, w=5.8, cap=''):
    if os.path.exists(path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after  = Pt(1)
        p.add_run().add_picture(path, width=Inches(w))
    if cap:
        c = doc.add_paragraph(cap)
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.space_after = Pt(5)
        for r in c.runs: _sf(r, 9, italic=True, color=(100,100,100))

def shd_cell(cell_obj, hx):
    tc = cell_obj._tc; tcPr = tc.get_or_add_tcPr()
    s  = OxmlElement('w:shd')
    s.set(qn('w:val'),'clear'); s.set(qn('w:color'),'auto'); s.set(qn('w:fill'),hx)
    tcPr.append(s)

def cell(c, text, bold=False, size=9.5, align='center', bg=None, fg=None):
    c.text = ''; c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    if bg: shd_cell(c, bg)
    p = c.paragraphs[0]
    p.alignment = {'center':WD_ALIGN_PARAGRAPH.CENTER,
                   'left':  WD_ALIGN_PARAGRAPH.LEFT,
                   'right': WD_ALIGN_PARAGRAPH.RIGHT}[align]
    r = p.add_run(text); _sf(r, size, bold, fg)

def thead(row, cols, bg='1F4E79'):
    for i,t in enumerate(cols):
        cell(row.cells[i], t, bold=True, size=9.5, bg=bg, fg=(255,255,255))

def callout(text, fill='EBF3FB', color=(0,84,166)):
    p = doc.add_paragraph()
    p.paragraph_format.space_after   = Pt(6)
    p.paragraph_format.left_indent   = Cm(0.3)
    p.paragraph_format.right_indent  = Cm(0.3)
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'),'clear'); shd.set(qn('w:color'),'auto'); shd.set(qn('w:fill'),fill)
    pPr.append(shd)
    r = p.add_run(text); _sf(r, 10.5, bold=True, color=color)

# ══════════════════════════════════════════════════════════════════════════════
# 제목 (표지 아님)
# ══════════════════════════════════════════════════════════════════════════════
p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_title.paragraph_format.space_before = Pt(0)
p_title.paragraph_format.space_after  = Pt(4)
r_t = p_title.add_run('주간 아파트 매매가격지수 예측 모델')
_sf(r_t, 16, True, (0, 56, 117))

p_sub = doc.add_paragraph()
p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_sub.paragraph_format.space_after = Pt(10)
r_s = p_sub.add_run('KB Land 데이터 기반  |  서울 6개 구  |  2026년 5월 25일 예측')
_sf(r_s, 10, italic=True, color=(100,100,100))

# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 및 탐색 분석
# ══════════════════════════════════════════════════════════════════════════════
h1('1. 데이터 및 탐색 분석')

h2('1.1 데이터 개요')
bul('출처: KB 부동산 데이터허브 — 주간 아파트 매매가격지수')
bul('기간: 2021-05-10 ~ 2026-05-04 (251주, 매주 월요일 기준)')
bul('대상: 서울 외곽 3구(노원·은평·서대문) + 강남 3구(서초·강남·송파)')
bul('지수 기준: 특정 기준 시점 = 100, 이후 상대 가격 변화를 주단위 추적')
bul('결측치: 없음')

img('output/vis_01_story.png', w=5.8,
    cap='[그림 1] 서울 6개 구 주간 아파트 매매가격지수 추이 (2021~2026) — 국면별 색상 구분')

bul('3개 국면 구분 (모델 학습 변수 아닌 탐색적 해석용)')
ev_names = list(ev.keys())
for en in ev_names:
    avg_d = np.mean([ev[en][d]['delta'] for d in DISTRICTS if d in ev[en]])
    avg_p = np.mean([ev[en][d]['delta_pct'] for d in DISTRICTS if d in ev[en]])
    bul(f'{en}: 6구 평균 누적 변화 {avg_d:+.1f}p ({avg_p:+.1f}%) — "같은 국면에서도 강남 3구와 외곽 3구 간 변화폭 차이 관찰됨"', sub=True)

h2('1.2 정상성 검정 및 주기성 분석')
bul('ADF 검정: 원시 지수 전 6구 단위근 존재(비정상, p > 0.05)')
bul('1차 차분 후 결과: 은평구만 완전 정상화(p=0.003) — 나머지 5구는 미완전 정상화')
bul('추가 차분 시 과차분·정보 손실 위험 → d=1 고정(ARIMA) 또는 차분 타겟 전략(NN) 사용')

img('output/fig_03_fft.png', w=5.8,
    cap='[그림 2] FFT 주기성 분석 (트렌드 제거 후)')

dp_avg = np.mean([dp[d][0] for d in DISTRICTS if d in dp])
bul(f'FFT 분석: 구별 주요 주기 42~63주 수준 — 연간 52주 계절성보다 장기 사이클에 해당')
bul('안정적인 반복 계절성은 뚜렷하지 않음 — 고정된 계절 패턴보다 거시적 국면 변화의 영향이 더 크게 나타난 것으로 해석됨')

h2('1.3 지역 간 관계 분석 (VAR-IRF)')
img('output/vis_07_irf.png', w=5.8,
    cap='[그림 3] VAR 충격반응함수 — 강남구 충격에 대한 6개 구 반응')
bul('VAR(1차 차분, 최적 시차 7주 AIC 기준) 충격반응 분석 결과')
bul('강남구 변화 충격 이후 외곽 3구가 일정 시차(1~2주)를 두고 반응하는 패턴이 관찰됨 — 보조적 근거로만 활용')
bul('외생 변수를 포함하지 않았으므로 인과 효과를 직접 추정한 것은 아님')
bul('IRF 충격 흡수 기간 ~6~8주 → SEQ_LEN 결정의 보조 근거 (VAR 시차 7주와 함께 16주 채택)')

# ══════════════════════════════════════════════════════════════════════════════
# 2. 모델 구성
# ══════════════════════════════════════════════════════════════════════════════
doc.add_page_break()
h1('2. 모델 구성')

h2('2.1 ARIMA (기준 모델)')
bul('구별 단변량 ARIMA(p, 1, q) — 6개 구 각각 독립 모델')
bul('차수 선택: 훈련 데이터(70%)에서만 AIC 최소화 그리드 탐색 (p ∈ {0,1,2,3}, q ∈ {0,1,2,3})')

# ARIMA 차수 표
tbl_ar = doc.add_table(rows=2, cols=len(DISTRICTS)+1)
tbl_ar.style = 'Table Grid'
thead(tbl_ar.rows[0], [''] + DISTRICTS)
cell(tbl_ar.rows[1].cells[0], 'ARIMA 차수', bold=True, bg='F5F5F5', align='left')
for ci, d in enumerate(DISTRICTS):
    o = arima_orders.get(d, ['-',1,'-'])
    cell(tbl_ar.rows[1].cells[ci+1], str(tuple(o)), size=9)
doc.add_paragraph().paragraph_format.space_after = Pt(3)

bul('평가: walk-forward rolling — 매 테스트 시점에서 실제 이력으로 3-step 예측 후 H+3 값을 t+3 실측값과 비교')
bul('스케일러 없음, 미래 정보 유입 없음')

h2('2.2 딥러닝 모델 (SimpleRNN / LSTM / GRU)')
bul(f'입력: SEQ_LEN=16주 × 164개 특성 (lag·rolling·차분·YoY·시간·교차구)')
bul('입력 투영(Proj): 164 → 32차원 (LayerNorm + ReLU) — 고차원 상관 특성 잡음 감소')
bul('출력: H+1·H+2·H+3 동시 예측 (multi-step 직접 출력) — 평가 기준은 H+3')
bul('공통: hidden=64, dropout=0.2, batch=16, Adam(lr=1e-3), early stop patience=20')
bul('데이터 분할: 훈련 70%(121 시퀀스) / 검증 10%(20) / 테스트 20%(40)')
bul('스케일러: MinMaxScaler — 훈련 구간에만 fit, 검증·테스트에는 transform만 적용')
bul('타겟: 절대 지수가 아닌 누적 차분(Δ) 예측 → 훈련·테스트 분포 불일치 해소')
bul(f'테스트 기간: {test_period}')

# ══════════════════════════════════════════════════════════════════════════════
# 3. H+3 평가 결과 및 최종 모델 선정
# ══════════════════════════════════════════════════════════════════════════════
h1('3. H+3 평가 결과 및 최종 모델 선정')

h2('3.1 모델별 H+3 성능 비교')
bul('평가 기준: H+3(3주 후) 예측값 vs 실제값 — 전 모델 동일 기준 적용')
bul('H+3 RMSE를 주지표로 사용, H+3 MAE와 DA는 보조 지표')

# H+3 성능 표
ranks = sorted(MODELS_ORDER, key=lambda x: h3_rmse[x])
tbl_perf = doc.add_table(rows=1+len(MODELS_ORDER), cols=4)
tbl_perf.style = 'Table Grid'
thead(tbl_perf.rows[0], ['모델', 'H+3 RMSE ↓', 'H+3 MAE ↓', 'H+3 DA(%) ↑'])
for ri, mn in enumerate(ranks):
    row = tbl_perf.rows[ri+1]
    is_best = (mn == best_model)
    bg = 'C8E6C9' if is_best else ('F5F5F5' if ri % 2 else 'FFFFFF')
    lbl = f'★ {mn} (채택)' if is_best else mn
    cell(row.cells[0], lbl,                   bold=is_best, bg=bg, align='left')
    cell(row.cells[1], f'{h3_rmse[mn]:.4f}',  bold=(ri==0), size=9.5, bg=bg)
    cell(row.cells[2], f'{h3_mae[mn]:.4f}',   size=9.5, bg=bg)
    cell(row.cells[3], f'{h3_da[mn]:.1f}%',   bold=is_best, size=9.5, bg=bg)
doc.add_paragraph().paragraph_format.space_after = Pt(3)

callout(
    f'최종 채택 모델: ARIMA — H+3 RMSE {h3_rmse["ARIMA"]:.4f}으로 전 모델 최우수 / H+3 DA {h3_da["ARIMA"]:.1f}%'
)
bul('현재 데이터와 H+3 평가 기준에서 ARIMA가 가장 낮은 3주 후 예측 오차를 보임')
bul('회복 구간(2025~2026)의 완만하고 방향성이 명확한 흐름에서 직전값 추종 구조가 유리하게 작동')
bul('LSTM은 H+3 DA 66.2%로 방향 예측에서도 열위 — SimpleRNN이 NN 중 최우수')

h2('3.2 ARIMA 평가 방식 상세')
bul('매 테스트 시점 t에서 t까지의 전체 이력으로 ARIMA 재적합 후 t+1·t+2·t+3 예측')
bul('H+3 RMSE = √(Σ(ŷ_{t+3} − y_{t+3})² / n) — NN의 H+3 평가와 동일한 실측값 기준')
bul('기존 보고서의 ARIMA 1-step 합산 RMSE(0.294)와 다름 — 본 보고서는 H+3 단독 RMSE(0.409) 기준')

# ══════════════════════════════════════════════════════════════════════════════
# 4. 최종 예측 결과
# ══════════════════════════════════════════════════════════════════════════════
h1('4. 최종 예측 결과 (2026-05-25)')

bul('채택 모델: ARIMA — 전체 데이터 학습 후 3주 연속 예측 (walk-forward 방식)')
bul('예측 목표일: 마지막 관측일 2026-05-04 기준 3주 후 = 2026-05-25')

# 3주 예측 표 (ARIMA)
tbl_fc = doc.add_table(rows=1+len(ar_dates), cols=1+len(DISTRICTS))
tbl_fc.style = 'Table Grid'
thead(tbl_fc.rows[0], ['예측일'] + DISTRICTS)
for ri, dt in enumerate(ar_dates):
    row = tbl_fc.rows[ri+1]
    is_t3 = (dt == '2026-05-25')
    bg = 'C8E6C9' if is_t3 else ('F5F5F5' if ri % 2 else 'FFFFFF')
    cell(row.cells[0], f'{"★ " if is_t3 else "  "}{dt}', bold=is_t3, size=9.5, bg=bg)
    for ci, d in enumerate(DISTRICTS):
        v = fc_ar_all[dt].get(d, fc_ar_t3.get(d, 0))
        cell(row.cells[ci+1], f'{v:.2f}', bold=is_t3, size=9.5, bg=bg)
doc.add_paragraph().paragraph_format.space_after = Pt(3)

bul('★ 2026-05-25: 최종 예측 목표일 — ARIMA H+3 예측값')
a_avg = np.mean([fc_ar_t3[d] for d in DISTRICTS])
bul(f'6구 평균 예측 지수: {a_avg:.2f} — 최근 회복 추세 반영, 완만한 상승 기조 지속 가능성 시사')

# ══════════════════════════════════════════════════════════════════════════════
# 5. 한계 및 제언
# ══════════════════════════════════════════════════════════════════════════════
h1('5. 한계 및 제언')

bul('훈련 데이터 부족: 251주(약 5년) — 레짐 변화를 충분히 학습하기 어려운 표본 규모')
bul('외생 변수 미포함: 금리·거래량·매수심리지수 등 추가 시 NN 모델의 차별화 가능성 있음')
bul('ARIMA 한계: 급격한 정책·금리 충격 국면에서 즉각 반응 어려움 — 회복 이후 평온 구간에 특히 유리')
bul('3주 초과 예측: 현 ARIMA 구조에서 오차 급증 예상 — 장기 예측 필요 시 NN 또는 외생 변수 도입 고려')
bul('일반적인 상황에서는 학습 데이터 확장 및 외생 변수 통합 시 NN 계열이 더 유리할 수 있음')

# ══════════════════════════════════════════════════════════════════════════════
# 참고문헌 (별도 페이지)
# ══════════════════════════════════════════════════════════════════════════════
doc.add_page_break()
h1('참고문헌')
refs = [
    'Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time Series Analysis: Forecasting and Control (5th ed.). Wiley.',
    'Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780.',
    'Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP 2014.',
    'Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts. https://otexts.com/fpp3/',
    'Sims, C. A. (1980). Macroeconomics and Reality. Econometrica, 48(1), 1–48.',
    'KB 부동산 데이터허브. (2026). 주간 아파트 매매가격지수. https://data.kbland.kr/',
]
for i, ref in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.first_line_indent = Cm(-0.8)
    r = p.add_run(f'[{i}] {ref}')
    _sf(r, 9.5)

# ── 저장 ──────────────────────────────────────────────────────────────────────
doc.save('output/report.docx')
print('저장 완료: output/report.docx')
