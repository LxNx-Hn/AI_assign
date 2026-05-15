"""KB Land API에서 6개 구 5년치 주간 아파트 매매가격지수 다운로드"""
import requests, json, os
import pandas as pd

DISTRICTS = ['노원구', '은평구', '서대문구', '서초구', '강남구', '송파구']

params = {
    '기간': '5',
    '매매전세코드': '01',
    '매물종별구분': '01',
    '월간주간구분코드': '02',
    '지역코드': '1100000000',  # 서울 → 구 단위 반환
    '조회시작일자': '',
    '조회종료일자': '',
    'type': 'false',
    '메뉴코드': '1'
}

headers = {
    'Referer': 'https://data.kbland.kr/',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

print('KB Land API 호출 중...')
url = 'https://data-api.kbland.kr/bfmstat/weekMnthlyHuseTrnd/priceIndex'
resp = requests.get(url, params=params, headers=headers, timeout=30)
resp.raise_for_status()

data = resp.json()
body = data['dataBody']['data']

dates    = body['날짜리스트']
regions  = body['데이터리스트']

print(f'날짜: {dates[0]} ~ {dates[-1]}  ({len(dates)}주)')
print(f'지역 수: {len(regions)}')

# 6개 구 필터
target = {r['지역명']: r['dataList'] for r in regions if r['지역명'] in DISTRICTS}
missing = [d for d in DISTRICTS if d not in target]
if missing:
    print(f'경고: {missing} 를 찾지 못했습니다')
else:
    print(f'6개 구 모두 확인: {list(target.keys())}')

# 날짜/데이터 길이 맞춤 (API가 날짜보다 값 1개 더 줄 수 있음)
n = len(dates)
target_trimmed = {k: v[:n] for k, v in target.items()}

# DataFrame 생성
df = pd.DataFrame(target_trimmed, index=pd.to_datetime(dates, format='%Y%m%d'))
df.index.name = '날짜'
df = df[DISTRICTS]  # 순서 고정

os.makedirs('data', exist_ok=True)
out_path = 'data/kb_apartment_index.csv'
df.to_csv(out_path, encoding='utf-8-sig')
print(f'\n저장 완료: {out_path}  ({df.shape})')
print(df.tail(3).round(3).to_string())
