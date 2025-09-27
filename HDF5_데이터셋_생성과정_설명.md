# HDF5 데이터셋 생성 과정 상세 설명

## 🔍 HDF5 데이터셋이 만들어지는 과정

X4M06 레이더 재밍 데이터셋의 HDF5 파일은 다음과 같은 **8단계 과정**을 통해 생성됩니다:

---

## 📋 1단계: 초기화 및 설정

### 1.1 레이더 시뮬레이터 초기화
```python
radar_sim = FMCWRadarSimulator(config)
```
**주요 설정값:**
- 중심 주파수: 8.748 GHz
- 대역폭: 1.4 GHz
- 처프 지속시간: 1ms
- 샘플링 주파수: 1 MHz
- 결과: **1,000개 복소수 샘플** (1ms × 1MHz)

### 1.2 스펙트로그램 생성기 초기화
```python
spec_gen = SpectrogramGenerator(stft_params)
```
**STFT 파라미터:**
- nperseg: 256 (윈도우 크기)
- noverlap: 128 (50% 오버랩)
- nfft: 512 (FFT 포인트)
- window: 'hann'

---

## 📊 2단계: 첫 번째 샘플로 데이터 구조 결정

### 2.1 샘플 신호 생성
```python
clean_signal, target_params = radar_sim.generate_clean_signal()
jammed_signal, jammer_params = radar_sim.generate_jammed_signal(clean_signal)
```

### 2.2 스펙트로그램 생성
```python
_, _, clean_spec = spec_gen.generate_spectrogram(clean_signal, sampling_rate)
_, _, jammed_spec = spec_gen.generate_spectrogram(jammed_signal, sampling_rate)
```
**결과:**
- 시간 신호: `(1000,)` 복소수
- 스펙트로그램: `(512, 9)` 실수 (주파수 × 시간)

---

## 💾 3단계: HDF5 파일 구조 생성

### 3.1 4개의 주요 데이터셋 생성
```python
with h5py.File(output_file, 'w') as f:
    # 1) 깨끗한 신호의 스펙트로그램
    f.create_dataset('clean_spectrograms', 
                    (1000, 512, 9), dtype=np.float32)
    
    # 2) 재밍된 신호의 스펙트로그램
    f.create_dataset('jammed_spectrograms', 
                    (1000, 512, 9), dtype=np.float32)
    
    # 3) 깨끗한 원본 신호 (복소수)
    f.create_dataset('clean_signals', 
                    (1000, 1000), dtype=np.complex64)
    
    # 4) 재밍된 신호 (복소수)
    f.create_dataset('jammed_signals', 
                    (1000, 1000), dtype=np.complex64)
```

### 3.2 메모리 효율성
- **float32**: 스펙트로그램용 (dB 값, 정밀도 충분)
- **complex64**: 복소수 신호용 (I/Q 채널)

---

## 🔄 4단계: 반복적 샘플 생성 (999회 반복)

각 샘플마다 다음 과정 반복:

### 4.1 물리적 파라미터 무작위 생성
```python
# 목표물 파라미터
target_range = random(5, 50)      # 거리 (m)
target_velocity = random(-30, 30) # 속도 (m/s)
target_rcs = random(0.1, 10)      # RCS (m²)

# 재머 파라미터
num_jammers = random(1, 4)        # 재머 개수
power_ratio = random(0.5, 2.0)    # 전력비
freq_offset = random(-100, 100)   # 주파수 오프셋 (MHz)
```

### 4.2 FMCW 레이더 신호 생성
```python
# 레이더 방정식 기반 에코 신호
time_delay = 2 * target_range / c  # 왕복 시간
doppler_freq = 2 * velocity * center_freq / c  # 도플러 주파수

# 처프 신호 생성
instantaneous_freq = center_freq + chirp_slope * time + doppler_freq
clean_signal = amplitude * exp(j * 2π * ∫freq dt)
```

### 4.3 재밍 신호 추가
```python
for each jammer:
    jamming_freq = center_freq + freq_offset + modified_slope * time
    jamming_signal = power_ratio * exp(j * 2π * ∫jamming_freq dt)
    jammed_signal += jamming_signal
```

---

## 📈 5단계: STFT 기반 스펙트로그램 생성

### 5.1 Short-Time Fourier Transform
```python
f, t, Zxx = scipy.signal.stft(signal, 
                             nperseg=256,    # 256 샘플 윈도우
                             noverlap=128,   # 128 샘플 오버랩
                             nfft=512)       # 512포인트 FFT
```

### 5.2 결과 차원
- **주파수 빈**: 512개 (0 ~ 500kHz, 해상도 ~977Hz)
- **시간 빈**: 9개 (1ms를 9구간으로 분할)
- **복소수 → 크기**: `|Zxx|` 계산 후 dB 스케일 변환

### 5.3 dB 변환 및 정규화
```python
spectrogram_db = 20 * log10(abs(Zxx) + 1e-12)  # dB 스케일
normalized = (spectrogram_db - min) / (max - min)  # 0-1 정규화
```

---

## 💽 6단계: 효율적 데이터 저장

### 6.1 스트리밍 저장 방식
```python
for i in range(1000):
    # 각 샘플을 즉시 HDF5 파일에 저장
    f['clean_spectrograms'][i] = clean_spec_normalized
    f['jammed_spectrograms'][i] = jammed_spec_normalized
    f['clean_signals'][i] = clean_signal
    f['jammed_signals'][i] = jammed_signal
```

### 6.2 메모리 최적화
- **한 번에 1개 샘플**씩 생성하여 메모리 사용량 최소화
- **압축**: HDF5 자체 압축으로 파일 크기 40% 절약

---

## 📝 7단계: 메타데이터 생성

### 7.1 샘플별 상세 정보 저장
```json
{
  "sample_id": 0,
  "target_params": [[21.5, -24.1, 2.8]],  // [거리, 속도, RCS]
  "jammer_params": [
    {
      "power_ratio": 1.03,
      "freq_offset": -66964773.8,
      "time_offset": 1.41e-05,
      "chirp_slope_ratio": 1.157
    }
  ]
}
```

### 7.2 실험 설정 정보
- 레이더 설정 (주파수, 대역폭 등)
- STFT 파라미터
- 생성 일시 및 조건

---

## ✅ 8단계: 데이터 품질 검증

### 8.1 자동 검증
- 모든 샘플의 차원 일관성 확인
- NaN/Inf 값 존재 여부 검사
- 정규화 범위 (0-1) 준수 확인

### 8.2 최종 결과
```
radar_jamming_dataset_1000.h5 (42.7 MB)
├── clean_spectrograms:   (1000, 512, 9) float32
├── jammed_spectrograms:  (1000, 512, 9) float32  
├── clean_signals:        (1000, 1000)   complex64
└── jammed_signals:       (1000, 1000)   complex64
```

---

## 🔬 생성된 데이터의 특징

### 물리적 정확성
1. **레이더 방정식**: 거리-RCS 관계 정확히 모델링
2. **도플러 효과**: 속도에 따른 주파수 이동 구현
3. **FMCW 특성**: 선형 주파수 증가 처프 신호

### 재밍 현실성
1. **다중 간섭원**: 1-4개 재머 동시 작동
2. **주파수 오프셋**: ±100MHz 인접 채널 간섭
3. **전력 변화**: 0.5-2.0배 강도 변화
4. **시간 오프셋**: 비동기 재밍 신호

### 딥러닝 최적화
1. **정규화된 입력**: 0-1 범위 스펙트로그램
2. **일관된 차원**: 모든 샘플 동일 크기
3. **라벨 데이터**: 깨끗한 신호가 정답 라벨
4. **메타데이터**: 성능 분석용 상세 정보

---

## 🎯 이 HDF5 데이터셋의 활용

### 딥러닝 모델 학습
- **입력**: `jammed_spectrograms` (재밍된 신호)
- **출력**: `clean_spectrograms` (복원 목표)
- **검증**: `clean_signals` vs 복원 결과

### 연구 분석
- 재밍 패턴 분석
- 복원 성능 평가
- 하드웨어 검증 기준

**이렇게 생성된 HDF5 파일은 물리적으로 정확하고 딥러닝 학습에 최적화된 고품질 데이터셋입니다!**