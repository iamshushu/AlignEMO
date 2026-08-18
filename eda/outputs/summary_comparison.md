# IEMOCAP vs MELD vs K-MIND : 데이터 비교

## 1. 감정 클래스 체계

| Dataset | #class | classes |
|---|---|---|
| IEMOCAP | 6 | ang, exc(excited), fru(frustrated), hap, neu, sad |
| MELD | 7 | anger, disgust, fear, joy, neutral, sadness, surprise |
| K-MIND(Ekman) | 7 | 기쁨, 슬픔, 화남, 놀람, 무서움, 혐오, 중립 |

- MELD·K-MIND 는 **Ekman 6 + neutral** 체계로 동일 정렬 가능.
- IEMOCAP 은 surprise/disgust/fear 없음, 대신 **excited·frustrated** 포함 (감정 축이 다름).

## 2. 구조 / 규모

| Dataset | #대화 | #발화 | 턴/대화(avg) | 화자/대화 | 형태 | 고유화자 |
|---|---|---|---|---|---|---|
| IEMOCAP | 151 | 10087 | 66.8 | 2.0 | dyadic(2인) | 302 |
| MELD | 1432 | 13708 | 9.6 | 2.73 | multi-party | 304 |
| K-MIND | 817 | 71705 | 87.8 | 2.0 | dyadic(2인) | 1634 |

## 3. 클래스 불균형 (전체 기준)

| Dataset | majority class % | IR(max/min) | entropy/max |
|---|---|---|---|
| IEMOCAP | 26.8% | 4.5x | 2.67/2.81 |
| MELD | 47.0% | 18.0x | 2.22/2.81 |
| K-MIND | 65.6% | 55.4x | 1.64/2.81 |

불균형이 심할수록 IR↑, entropy↓.
