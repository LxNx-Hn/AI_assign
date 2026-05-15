"""
KB Land 주간 아파트 매매가격지수 예측 보고서
본문 A4 3쪽 이내 / 개조식 / 참고문헌 별도 페이지
"""
import json, os
import numpy as np
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
with open('output/results.json', encoding='utf-8') as f:
    R = json.load(f)

DISTRICTS       = ['노원구', '은평구', '서대문구', '서초구', '강남구', '송파구']
DISTRICTS_SHORT = ['노원',   '은평',   '서대문',   '서초',   '강남',   '송파']
MODELS_ORDER    = ['ARIMA', 'SimpleRNN', 'LSTM', 'GRU']
best_model      = R['best_model']

h3_rmse      = R['h3_rmse']
h3_mae       = R['h3_mae']
h3_da        = R['h3_da']
fc_ar_t3     = R.get('final_forecast_2026_05_25', R['arima_forecast_2026_05_25'])
fc_rnn       = R.get('simple_rnn_forecast_2026_05_25', {})
ev           = R.get('event_changes', {})
arima_orders = R.get('arima_orders', {})
model_metrics = R.get('model_metrics', {})
test_period  = R.get('test_period', '')

# 이벤트 구간 메타
# (표시 이름, JSON 키, 기간, 해석)
EVENT_META = [
    ('COVID 이후 급등기', 'COVID 급등기',   '2021.05~2022.06', '전반적 상승'),
    ('금리인상 하락기',   '금리인상 하락기', '2022.07~2023.12', '전반적 조정'),
    ('회복기',           '회복기',          '2024.01~2026.05', '회복 강도 확대'),
]

# ══════════════════════════════════════════════════════════════════════════════
# 문서 생성 및 전역 설정
# ══════════════════════════════════════════════════════════════════════════════
doc = Document()

ns = doc.styles['Normal']
ns.paragraph_format.space_before      = Pt(0)
ns.paragraph_format.space_after       = Pt(0)
ns.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
ns.paragraph_format.line_spacing      = 1.05

for sec in doc.sections:
    sec.page_height   = Cm(29.7); sec.page_width    = Cm(21.0)
    sec.top_margin    = Cm(1.2);  sec.bottom_margin = Cm(1.2)
    sec.left_margin   = Cm(1.4);  sec.right_margin  = Cm(1.4)

# ══════════════════════════════════════════════════════════════════════════════
# 헬퍼 함수
# ══════════════════════════════════════════════════════════════════════════════
def _tight(p, before=0, after=0, ls=1.05):
    pf = p.paragraph_format
    pf.space_before = Pt(before); pf.space_after = Pt(after)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE; pf.line_spacing = ls

def _sf(run, size=9.2, bold=False, color=None, italic=False):
    run.font.name = '맑은 고딕'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    run.font.size = Pt(size); run.bold = bold; run.italic = italic
    if color: run.font.color.rgb = RGBColor(*color)

def h1(text):
    p = doc.add_heading(text, level=1)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    _tight(p, before=5, after=2)
    for r in p.runs: _sf(r, 11.0, True, (0, 56, 117))

def h2(text):
    p = doc.add_heading(text, level=2)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    _tight(p, before=3, after=1)
    for r in p.runs: _sf(r, 10.0, True, (30, 100, 160))

def bul(text, sub=False, after=1):
    p = doc.add_paragraph()
    _tight(p, before=0, after=after)
    p.paragraph_format.left_indent       = Cm(0.85 if sub else 0.28)
    p.paragraph_format.first_line_indent = Cm(-0.28)
    _sf(p.add_run(('  -  ' if sub else '·  ') + text),
        8.8 if sub else 9.2)

def img(path, w=4.8, cap=''):
    if os.path.exists(path):
        p = doc.add_paragraph()
        _tight(p, before=2, after=0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(path, width=Inches(w))
    if cap:
        c = doc.add_paragraph(cap)
        _tight(c, before=0, after=2)
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in c.runs: _sf(r, 8.0, italic=True, color=(100, 100, 100))

def callout(text, fill='EBF3FB', color=(0, 84, 166)):
    p = doc.add_paragraph()
    _tight(p, before=2, after=2)
    p.paragraph_format.left_indent  = Cm(0.2)
    p.paragraph_format.right_indent = Cm(0.2)
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'),'clear'); shd.set(qn('w:color'),'auto'); shd.set(qn('w:fill'),fill)
    pPr.append(shd)
    _sf(p.add_run(text), 9.2, bold=True, color=color)

# ── 표 셀 보조 ────────────────────────────────────────────────────────────────
def _shd(co, hx):
    tc = co._tc; tcPr = tc.get_or_add_tcPr()
    s  = OxmlElement('w:shd')
    s.set(qn('w:val'),'clear'); s.set(qn('w:color'),'auto'); s.set(qn('w:fill'),hx)
    tcPr.append(s)

def _cell_margins(tbl, top=20, left=50, bottom=20, right=50):
    for row in tbl.rows:
        for c in row.cells:
            tc = c._tc; tcPr = tc.get_or_add_tcPr()
            tcMar = OxmlElement('w:tcMar')
            for side, val in [('top',top),('left',left),('bottom',bottom),('right',right)]:
                nd = OxmlElement(f'w:{side}')
                nd.set(qn('w:w'), str(val)); nd.set(qn('w:type'), 'dxa')
                tcMar.append(nd)
            tcPr.append(tcMar)

def _set_col_widths(tbl, widths_cm):
    TWP = 567
    for ci, wc in enumerate(widths_cm):
        for c in tbl.columns[ci].cells:
            tc = c._tc; tcPr = tc.get_or_add_tcPr()
            tcW = OxmlElement('w:tcW')
            tcW.set(qn('w:w'), str(int(wc * TWP))); tcW.set(qn('w:type'), 'dxa')
            tcPr.append(tcW)

def cell(c, text, bold=False, size=8.5, align='center', bg=None, fg=None):
    c.text = ''; c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    if bg: _shd(c, bg)
    p = c.paragraphs[0]; _tight(p)
    p.alignment = {'center':WD_ALIGN_PARAGRAPH.CENTER,'left':WD_ALIGN_PARAGRAPH.LEFT,
                   'right':WD_ALIGN_PARAGRAPH.RIGHT}[align]
    _sf(p.add_run(text), size, bold, fg)

def thead(row, cols, bg='1F4E79', size=8.5):
    for i, t in enumerate(cols):
        cell(row.cells[i], t, bold=True, size=size, bg=bg, fg=(255,255,255))

def gap(after=1):
    p = doc.add_paragraph(); _tight(p, after=after)

# ══════════════════════════════════════════════════════════════════════════════
# 제목 (표지 없음)
# ══════════════════════════════════════════════════════════════════════════════
p_title = doc.add_paragraph()
_tight(p_title, before=0, after=2)
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
_sf(p_title.add_run('주간 아파트 매매가격지수 예측 모델'), 15.0, True, (0,56,117))

p_sub = doc.add_paragraph()
_tight(p_sub, before=0, after=4)
p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
_sf(p_sub.add_run('KB Land 데이터 기반  |  서울 6개 구  |  2026년 5월 25일 예측'),
    9.0, italic=True, color=(100,100,100))

# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 및 탐색 분석
# ══════════════════════════════════════════════════════════════════════════════
h1('1. 데이터 및 탐색 분석')

h2('1.1 데이터 개요')
bul('출처: KB 부동산 데이터허브 — 주간 아파트 매매가격지수, 2021-05-10 ~ 2026-05-04 (251주, 결측치 없음)')
bul('대상: 서울 6구 — 외곽 3구(노원·은평·서대문) + 강남 3구(서초·강남·송파) / 지수 기준 시점 = 100')

img('output/vis_01_story.png', w=5.0,
    cap='[그림 1] 서울 6개 구 주간 아파트 매매가격지수 추이 (2021~2026) — 국면별 색상 구분')

bul('3개 국면 구분 (탐색적 해석용, 모델 학습 변수 아님)', after=2)

# ── 이벤트 구간 요약표 ────────────────────────────────────────────────────────
ev_tbl = doc.add_table(rows=1 + len(EVENT_META), cols=4)
ev_tbl.style = 'Table Grid'
thead(ev_tbl.rows[0], ['구간', '기간', '6구 평균 변화', '해석'], size=8.2)
for ri, (name, ev_key, period, interp) in enumerate(EVENT_META):
    row = ev_tbl.rows[ri + 1]
    if ev_key in ev:
        avg_d = np.mean([ev[ev_key][d]['delta']     for d in DISTRICTS if d in ev[ev_key]])
        avg_p = np.mean([ev[ev_key][d]['delta_pct'] for d in DISTRICTS if d in ev[ev_key]])
        chg   = f'{avg_d:+.1f}p ({avg_p:+.1f}%)'
    else:
        chg = '-'
    bg = 'F5F5F5' if ri % 2 == 0 else 'FFFFFF'
    cell(row.cells[0], name,   size=8.2, bg=bg, align='left')
    cell(row.cells[1], period, size=8.2, bg=bg)
    cell(row.cells[2], chg,    size=8.2, bg=bg)
    cell(row.cells[3], interp, size=8.2, bg=bg, align='left')
_cell_margins(ev_tbl, top=18, left=45, bottom=18, right=45)
_set_col_widths(ev_tbl, [3.6, 3.5, 4.0, 7.1])
gap(after=1)

h2('1.2 정상성 검정 및 주기성 분석')
bul('ADF 검정: 전 6구 원시 지수 비정상(단위근 존재); 1차 차분 후 은평구만 완전 정상화 → d=1 고정(ARIMA), 차분 타겟 전략(NN) 적용')

img('output/fig_03_fft.png', w=4.8,
    cap='[그림 2] FFT 주기성 분석 (트렌드 제거 후)')

bul('구별 주요 주기 약 42~63주 범위의 중기적 주기 관찰; 안정적 반복 계절성은 뚜렷하지 않음 — 거시적 국면 변화의 영향이 더 크게 나타난 것으로 해석됨')

h2('1.3 지역 간 관계 분석 (VAR-IRF)')
img('output/vis_07_irf.png', w=4.8,
    cap='[그림 3] VAR 충격반응함수 — 강남구 충격에 대한 6개 구 반응')
bul('강남구 충격 이후 외곽 3구가 1~2주 시차로 반응하는 패턴 관찰 — 보조적 근거; 외생 변수 미포함으로 인과 효과 직접 추정 아님')
bul('IRF 충격 흡수 기간 약 6~8주, VAR 최적 시차 7주(AIC) → SEQ_LEN=16주 결정의 보조 근거')

# ══════════════════════════════════════════════════════════════════════════════
# 2. 모델 구성
# ══════════════════════════════════════════════════════════════════════════════
h1('2. 모델 구성')

h2('2.1 ARIMA (기준 모델)')
bul('구별 단변량 ARIMA(p,1,q) — 훈련 데이터(70%)에서만 AIC 최소화 그리드 탐색 (p,q ∈ {0…3})')

# ARIMA 차수 미니표
ar_tbl = doc.add_table(rows=2, cols=1 + len(DISTRICTS))
ar_tbl.style = 'Table Grid'
thead(ar_tbl.rows[0], [''] + DISTRICTS_SHORT, size=8.2)
row_ar = ar_tbl.rows[1]
cell(row_ar.cells[0], 'ARIMA(p,1,q)', bold=True, size=8.2, bg='F5F5F5', align='left')
for ci, d in enumerate(DISTRICTS):
    o = arima_orders.get(d, ['-', 1, '-'])
    cell(row_ar.cells[ci + 1], f'({o[0]},1,{o[2]})', size=8.2)
_cell_margins(ar_tbl, top=15, left=40, bottom=15, right=40)
_set_col_widths(ar_tbl, [3.2, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
gap(after=1)

bul('평가: walk-forward rolling — 매 시점 전체 이력으로 3-step forecast 수행, H+3 값 사용; 스케일러 없음, 미래 정보 유입 없음')

h2('2.2 딥러닝 모델 (SimpleRNN / LSTM / GRU)')
bul(f'입력: SEQ_LEN=16주 × 164개 특성(lag·rolling·차분·YoY·교차구) → 32차원 투영(LayerNorm+ReLU); 출력: H+1·H+2·H+3 직접 예측')
bul('타겟: 누적 차분(Δ) 예측; 스케일러: MinMaxScaler(훈련 fit only); 분할: 훈련 70% / 검증 10% / 테스트 20%')
bul(f'공통 하이퍼파라미터: hidden=64, dropout=0.2, Adam(lr=1e-3), patience=20; 테스트 기간: {test_period}')

# ══════════════════════════════════════════════════════════════════════════════
# 3. H+3 평가 결과 및 최종 모델 선정
# ══════════════════════════════════════════════════════════════════════════════
h1('3. H+3 평가 결과 및 최종 모델 선정')

h2('3.1 모델별 H+3 성능 비교')
bul('과제 목표(3주 후 예측)에 맞게 전 모델을 H+3 단독 성능으로 비교·선정 — 1-step 또는 전체 step 합산 RMSE와 구별')
bul('평가 기준: H+3(3주 후) 예측값 vs 실제값 — 전 모델 동일 기준; H+3 RMSE 주지표, MAE·DA 보조')
bul('DA: 테스트 구간 내 방향 일치율 — 예측 방향(상승/하락)이 실제와 일치한 비율')

# H+3 성능 비교표
ranks    = sorted(MODELS_ORDER, key=lambda x: h3_rmse[x])
tbl_perf = doc.add_table(rows=1 + len(MODELS_ORDER), cols=4)
tbl_perf.style = 'Table Grid'
thead(tbl_perf.rows[0], ['모델', 'H+3 RMSE ↓', 'H+3 MAE ↓', 'H+3 DA(%) ↑'])
for ri, mn in enumerate(ranks):
    row     = tbl_perf.rows[ri + 1]
    is_best = (mn == best_model)
    bg      = 'C8E6C9' if is_best else ('F5F5F5' if ri % 2 else 'FFFFFF')
    lbl     = f'★ {mn} (채택)' if is_best else mn
    cell(row.cells[0], lbl,                  bold=is_best, bg=bg, align='left')
    cell(row.cells[1], f'{h3_rmse[mn]:.4f}', bold=(ri==0), size=8.5, bg=bg)
    cell(row.cells[2], f'{h3_mae[mn]:.4f}',  size=8.5, bg=bg)
    cell(row.cells[3], f'{h3_da[mn]:.1f}%',  bold=is_best, size=8.5, bg=bg)
_cell_margins(tbl_perf, top=20, left=50, bottom=20, right=50)
gap(after=1)

callout(
    f'최종 채택 모델: ARIMA — H+3 RMSE {h3_rmse["ARIMA"]:.4f}으로 전 모델 최우수 / H+3 DA {h3_da["ARIMA"]:.1f}%'
)

# ARIMA 선정 이유
bul('현재 데이터·H+3 평가 기준에서 ARIMA가 가장 낮은 3주 후 예측 오차; SimpleRNN 대비 RMSE 30.7%, LSTM 대비 41.1% 낮음')
bul('테스트 구간은 완만한 회복 흐름이 지속된 구간 — 직전 이력·단기 자기상관을 반영하는 ARIMA 구조가 H+3 예측에 유리하게 작동한 것으로 해석됨')
bul('NN 계열은 164개 특성을 사용했으나, 121개로 제한된 훈련 시퀀스로 복잡한 비선형 패턴을 안정적으로 학습하기 어려웠을 가능성')
bul('단, ARIMA가 항상 우수하다는 의미가 아니라 현재 데이터·H+3 단기 예측 조건에서 가장 안정적이었다는 해석에 한정')
bul(f'NN 3종 H+3 성능 비교: SimpleRNN(RMSE {h3_rmse["SimpleRNN"]:.4f} · DA {h3_da["SimpleRNN"]:.1f}%) · '
    f'GRU({h3_rmse["GRU"]:.4f} · DA {h3_da["GRU"]:.1f}%) · '
    f'LSTM({h3_rmse["LSTM"]:.4f} · DA {h3_da["LSTM"]:.1f}%) — SimpleRNN이 NN 3종 중 RMSE 최우수')
bul('GRU는 DA 96.7%로 방향 일치율은 높았으나, RMSE 기준에서는 SimpleRNN보다 오차가 커 최종 모델로 채택하지 않음')

h2('3.2 ARIMA 평가 방식 상세')
bul('walk-forward rolling — 매 시점 전체 이력으로 ARIMA 재적합, t+1·t+2·t+3 예측, H+3만 평가에 사용')
bul('주의: walk-forward 전체 step 합산 RMSE(0.294)와 H+3 단독 RMSE(0.409)는 다름 — 본 평가는 H+3 기준', after=2)

# 구별 H+3 RMSE 비교표 (전 모델 × 6구)
bul('구별 H+3 RMSE 세부: ARIMA는 6구 중 4구(노원·은평·서대문·송파)에서 최우수, 6구 평균 기준 전 모델 최우수', after=2)

dist_tbl = doc.add_table(rows=1 + len(MODELS_ORDER), cols=1 + len(DISTRICTS))
dist_tbl.style = 'Table Grid'
thead(dist_tbl.rows[0], ['모델'] + DISTRICTS_SHORT, size=8.0)
for ri, mn in enumerate(MODELS_ORDER):
    row     = dist_tbl.rows[ri + 1]
    is_best = (mn == best_model)
    bg_row  = 'C8E6C9' if is_best else ('F5F5F5' if ri % 2 else 'FFFFFF')
    cell(row.cells[0], f'★ {mn}' if is_best else mn,
         bold=is_best, size=8.0, bg=bg_row, align='left')
    for ci, d in enumerate(DISTRICTS):
        v    = model_metrics.get(mn, {}).get(d, {}).get('RMSE_h3', None)
        txt  = f'{v:.3f}' if v is not None else '-'
        # 해당 구에서 최솟값인지 확인
        all_v = [model_metrics.get(m,{}).get(d,{}).get('RMSE_h3', 9) for m in MODELS_ORDER]
        is_min = (v is not None and abs(v - min(all_v)) < 0.0001)
        bg_c  = 'C8E6C9' if is_min else bg_row
        cell(row.cells[ci + 1], txt, bold=is_min, size=8.0, bg=bg_c)
_cell_margins(dist_tbl, top=15, left=38, bottom=15, right=38)
_set_col_widths(dist_tbl, [3.0, 2.53, 2.53, 2.53, 2.53, 2.53, 2.53])
gap(after=1)

bul('DA 기준: 은평·서대문·서초·송파 100%, 강남 97.96%, 노원 93.88% — 전 구에서 방향 예측 정확도 93% 이상')

# ══════════════════════════════════════════════════════════════════════════════
# 4. 최종 예측 결과
# ══════════════════════════════════════════════════════════════════════════════
h1('4. 최종 예측 결과 (2026-05-25)')

bul('채택 모델: ARIMA — 전체 이력(251주)으로 3-step forecast 수행, H+3 값 사용')
bul('예측 목표일: 마지막 관측일 2026-05-04 기준 3주 후 = 2026-05-25')

# 단일 행 예측표
tbl_fc = doc.add_table(rows=2, cols=1 + len(DISTRICTS))
tbl_fc.style = 'Table Grid'
thead(tbl_fc.rows[0], ['예측일'] + DISTRICTS)
row_fc = tbl_fc.rows[1]
cell(row_fc.cells[0], '★ 2026-05-25', bold=True, size=8.5, bg='C8E6C9')
for ci, d in enumerate(DISTRICTS):
    cell(row_fc.cells[ci + 1], f'{fc_ar_t3[d]:.2f}', bold=True, size=8.5, bg='C8E6C9')
_cell_margins(tbl_fc, top=20, left=50, bottom=20, right=50)
gap(after=1)

a_avg = np.mean([fc_ar_t3[d] for d in DISTRICTS])
bul(f'6구 평균 예측 지수: {a_avg:.2f} — 테스트 후반 회복 흐름이 단기적으로 유지되는 형태')
bul(f'서대문구({fc_ar_t3["서대문구"]:.2f})·노원구({fc_ar_t3["노원구"]:.2f})는 상대적으로 높은 예측 지수; '
    f'강남구({fc_ar_t3["강남구"]:.2f})는 100 이하 수준 — 지역별 회복 속도 차이 반영')
bul('외부 충격은 직접 반영하지 않은 가격지수 기반 단기 예측으로, 최근 흐름이 단기적으로 이어지는 경우의 예측값으로 해석')
bul('회복기(2024.01~) 진입 이후 6구 모두 100 이상 또는 100 근방 유지 흐름의 연장선; 이 흐름이 단기적으로 지속된다는 가정에서 산출')
bul(f'강남구 {fc_ar_t3["강남구"]:.2f} — 회복기에도 기준점(100) 이하 예측 유지, 서초({fc_ar_t3["서초구"]:.2f})·송파({fc_ar_t3["송파구"]:.2f})와 구분되는 상대적 약세 패턴')

h2('4.1 비채택 모델 참고 (SimpleRNN)')
bul('SimpleRNN은 NN 계열 중 H+3 RMSE가 가장 낮았으나, 전체 모델 기준으로는 ARIMA보다 오차가 큼')
bul('일부 방향성은 유사했지만, 최종 예측값은 H+3 RMSE 기준 최우수인 ARIMA 결과만 사용')

# ══════════════════════════════════════════════════════════════════════════════
# 5. 한계 및 제언
# ══════════════════════════════════════════════════════════════════════════════
h1('5. 한계 및 제언')

bul('표본 수 제한: 251주(약 5년) — 레짐 변화를 충분히 학습하기 어려운 표본 규모; 데이터 누적에 따라 재검토 필요')
bul('외생 변수 미포함: 금리·거래량 등 추가 시 NN 계열의 성능 개선 가능성을 검토할 수 있음; 인과 효과 직접 추정 아님')
bul('3주 초과 예측: 장기 horizon에서는 추가 검증 필요 — NN 또는 외생 변수 도입 고려')
bul('단일 데이터소스 의존: KB 주간 지수 외 실거래가·공시가격·매물량 등 교차 데이터 미활용 — 보완 시 예측 안정성 개선 가능')
bul('예측 구간 미산출: 포인트 예측만 제공 — 불확실성 정량화는 bootstrapped ARIMA 또는 베이지안 접근을 통한 추후 과제')
bul('일반화 한계: 서울 6구 한정 분석 — 타 지역·타 부동산 유형(오피스텔·빌라 등)에 대한 직접 적용 부적절')

# ══════════════════════════════════════════════════════════════════════════════
# 참고문헌 (별도 페이지)
# ══════════════════════════════════════════════════════════════════════════════
h1('참고문헌')
refs = [
    'Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). '
    'Time Series Analysis: Forecasting and Control (5th ed.). Wiley.',
    'Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. '
    'Neural Computation, 9(8), 1735–1780.',
    'Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., '
    'Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using '
    'RNN Encoder-Decoder for Statistical Machine Translation. EMNLP 2014.',
    'Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and '
    'Practice (3rd ed.). OTexts. https://otexts.com/fpp3/',
    'Sims, C. A. (1980). Macroeconomics and Reality. Econometrica, 48(1), 1–48.',
    'KB 부동산 데이터허브. (2026). 주간 아파트 매매가격지수. https://data.kbland.kr/',
]
for i, ref in enumerate(refs, 1):
    p = doc.add_paragraph()
    _tight(p, before=0, after=3)
    p.paragraph_format.left_indent       = Cm(0.7)
    p.paragraph_format.first_line_indent = Cm(-0.7)
    _sf(p.add_run(f'[{i}] {ref}'), 8.0)

# ── 저장 ──────────────────────────────────────────────────────────────────────
doc.save('output/report.docx')
print('저장 완료: output/report.docx')
