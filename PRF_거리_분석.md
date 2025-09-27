# X4M06 레이더 PRF와 측정 거리 분석

## 🎯 PRF와 최대 측정 거리 관계

### 📐 기본 공식
```
최대_측정_거리 = (빛의속도 × PRI) / 2
여기서 PRI (Pulse Repetition Interval) = 1 / PRF
```

### 📊 현재 시뮬레이션 설정 분석

#### **설정된 PRF: 1,000 Hz**
```python
prf = 1000  # Hz
PRI = 1 / prf = 1 / 1000 = 0.001초 = 1ms

최대_측정_거리 = (3 × 10^8 m/s × 0.001s) / 2
                = 300,000m / 2  
                = 150,000m = 150km
```

### ⚠️ **문제점 발견!**

| 항목 | 값 | 문제점 |
|------|----|---------| 
| **PRF** | 1,000 Hz | 너무 낮음 |
| **최대 거리** | 150 km | 실내 실험에 부적합 |
| **실제 실험 거리** | 1m 이내 | 설정과 큰 차이 |
| **거리 분해능** | 매우 낮음 | 1m 이내 정밀 측정 불가 |

---

## 🔧 X4M06 실제 사양 확인

### 📋 X4M06 데이터시트 기준

#### **실제 X4M06 사양**
- **주파수 대역**: 8.748 GHz (중심주파수)
- **대역폭**: 1.4 GHz
- **측정 범위**: 0.18m ~ 9.4m (일반적)
- **거리 분해능**: ~5-10cm
- **PRF**: 일반적으로 **23,328 Hz ~ 50,000 Hz**

#### **권장 PRF 계산**
```python
# 1m 이내 정밀 측정을 위한 PRF
목표_최대_거리 = 1.0  # m
안전_여유 = 2  # 배수

필요한_PRF = (3 × 10^8) / (2 × 목표_최대_거리 × 안전_여유)
            = 300,000,000 / (2 × 1.0 × 2)
            = 75,000,000 Hz = 75 MHz

# 하지만 실제 레이더는 처리 능력 한계로 이보다 낮음
실용적_PRF = 50,000 Hz  # 50 kHz
```

---

## 📊 PRF별 측정 거리 비교

| PRF (Hz) | PRI (μs) | 최대 거리 (m) | 적용 분야 |
|----------|----------|---------------|-----------|
| **1,000** | 1,000 | 150,000 | ❌ 항공 레이더 (부적합) |
| **10,000** | 100 | 15,000 | ❌ 원거리 감시 레이더 |
| **23,328** | 42.9 | 6,435 | ⚠️ X4M06 최소 PRF |
| **50,000** | 20 | 3,000 | ✅ 실용적 설정 |
| **100,000** | 10 | 1,500 | ✅ 근거리 정밀 측정 |
| **500,000** | 2 | 300 | ✅ **1m 이내 실험 최적** |
| **1,000,000** | 1 | 150 | ✅ 초근거리 정밀 |

---

## 🔧 시뮬레이션 파라미터 수정 권장사항

### 1m 이내 실험을 위한 설정

#### **수정된 레이더 설정**
```python
# 기존 설정 (문제)
radar_config_old = {
    'center_freq': 8.748e9,      # 8.748 GHz
    'bandwidth': 1.4e9,          # 1.4 GHz  
    'chirp_duration': 1e-3,      # 1 ms
    'prf': 1000,                 # 1 kHz ❌ 너무 낮음
    'sampling_rate': 1e6,        # 1 MHz
    'target_range': [5, 50],     # 5-50m ❌ 너무 먼 거리
}

# 수정된 설정 (권장)
radar_config_new = {
    'center_freq': 8.748e9,      # 8.748 GHz (유지)
    'bandwidth': 1.4e9,          # 1.4 GHz (유지)
    'chirp_duration': 10e-6,     # 10 μs (단축)
    'prf': 500000,               # 500 kHz ✅ 1m 이내 최적
    'sampling_rate': 5e6,        # 5 MHz (향상)
    'target_range': [0.1, 1.0],  # 0.1-1.0m ✅ 실내 실험 적합
}
```

#### **거리 분해능 계산**
```python
거리_분해능 = 빛의속도 / (2 × 대역폭)
           = 3 × 10^8 / (2 × 1.4 × 10^9)
           = 0.107m = 10.7cm

# 이는 1m 이내에서 약 9개 구간 분해 가능
```

---

## 🧪 실험 환경에 맞는 설정

### 실내 1m 이내 실험 시나리오

#### **타겟 배치 예시**
```python
실험_시나리오 = {
    '초근거리': [0.1, 0.15, 0.2, 0.25, 0.3],      # 10-30cm
    '근거리': [0.4, 0.5, 0.6, 0.7, 0.8],          # 40-80cm  
    '경계': [0.9, 0.95, 1.0],                      # 90-100cm
}
```

#### **PRF 최적화**
```python
def calculate_optimal_prf(max_range, safety_factor=2):
    """
    최대 측정 거리에 맞는 최적 PRF 계산
    """
    c = 3e8  # 빛의 속도
    prf = c / (2 * max_range * safety_factor)
    return prf

# 1m 실험용 PRF
optimal_prf = calculate_optimal_prf(1.0, 2)
print(f"1m 실험 최적 PRF: {optimal_prf:,.0f} Hz = {optimal_prf/1000:.0f} kHz")

# 결과: 75,000,000 Hz = 75,000 kHz (너무 높음)
# 실용적 선택: 500,000 Hz = 500 kHz
```

---

## 🔧 코드 수정 제안

### jamming_simulator.py 수정

#### **레이더 설정 부분**
```python
# 기존 설정 (Line 573-584 주변)
radar_config = {
    'center_freq': 8.748e9,
    'bandwidth': 1.4e9,
    'chirp_duration': 1e-3,      # ❌ 1ms는 너무 김
    'prf': 1000,                 # ❌ 1kHz는 너무 낮음
    'sampling_rate': 1e6,
    'target_range': [5, 50],     # ❌ 5-50m는 너무 멀음
    'target_velocity': [-30, 30],
    'target_rcs': [0.1, 10],
    'num_jammers': [1, 5],
    'jammer_power_ratio': [0.5, 2.0],
    'freq_offset_range': [-0.1e9, 0.1e9],
    'time_offset_range': [0, 0.8e-3],  # ❌ 0.8ms도 너무 김
    'snr_db': [15, 25]
}
```

#### **수정된 설정**
```python
radar_config = {
    'center_freq': 8.748e9,      # 8.748 GHz (유지)
    'bandwidth': 1.4e9,          # 1.4 GHz (유지)  
    'chirp_duration': 20e-6,     # ✅ 20μs (1m 이내 적합)
    'prf': 100000,               # ✅ 100 kHz (1.5km 최대, 실용적)
    'sampling_rate': 2e6,        # ✅ 2 MHz (해상도 향상)
    'target_range': [0.15, 0.9], # ✅ 15-90cm (실내 실험 적합)
    'target_velocity': [-2, 2],  # ✅ ±2m/s (실내 이동 속도)
    'target_rcs': [0.01, 1.0],   # ✅ 작은 물체 대응
    'num_jammers': [1, 3],       # ✅ 실내 환경 맞춤
    'jammer_power_ratio': [0.5, 2.0],  # 유지
    'freq_offset_range': [-0.05e9, 0.05e9],  # ✅ 범위 축소
    'time_offset_range': [0, 20e-6],    # ✅ 20μs 이내
    'snr_db': [10, 20]           # ✅ 실내 환경 맞춤
}
```

---

## 📋 실험 절차 수정 제안

### 현실적인 실험 환경

#### **1m 이내 타겟 배치**
1. **0.2m**: 책 또는 작은 상자
2. **0.4m**: 노트북 또는 의자
3. **0.6m**: 사람 (앉은 상태)
4. **0.8m**: 사람 (서 있는 상태)  
5. **1.0m**: 벽 또는 큰 물체

#### **측정 시나리오**
```python
실험_단계 = {
    '정적_타겟': '고정된 물체 측정',
    '동적_타겟': '사람의 느린 움직임 (±1m/s)',
    '다중_타겟': '여러 물체 동시 배치',
    '재질_변화': '금속, 플라스틱, 목재 등',
}
```

---

## 🎯 즉시 적용 가능한 수정

### main_experiment.py의 기본 설정 수정

```python
def _get_default_simulation_config(self):
    """시뮬레이션 기본 설정 - 1m 이내 실험 최적화"""
    return {
        'radar_config': {
            'center_freq': 8.748e9,
            'bandwidth': 1.4e9,
            'chirp_duration': 20e-6,     # ✅ 20μs
            'prf': 100000,               # ✅ 100 kHz  
            'sampling_rate': 2e6,        # ✅ 2 MHz
            'target_range': [0.15, 0.9], # ✅ 15-90cm
            'target_velocity': [-2, 2],  # ✅ ±2 m/s
            'target_rcs': [0.01, 1.0],   # ✅ 작은 물체
            'num_jammers': [1, 3],
            'jammer_power_ratio': [0.5, 2.0],
            'freq_offset_range': [-0.05e9, 0.05e9],
            'time_offset_range': [0, 20e-6],  # ✅ 20μs
            'snr_db': [10, 20]
        },
        'stft_params': {
            'nperseg': 128,    # ✅ 더 세밀한 분석
            'noverlap': 64,    # ✅ 50% 중첩
            'nfft': 256,
            'window': 'hann'
        },
        'num_samples': 1000,
        'num_visualize': 5,
        'save_format': 'hdf5'
    }
```

## 🏆 결론

**현재 PRF 1,000 Hz는 150km 측정용이므로 1m 이내 실험에 완전히 부적합합니다!**

### ✅ 권장 수정사항
1. **PRF**: 1,000 Hz → 100,000 Hz (100 kHz)
2. **측정 범위**: [5, 50]m → [0.15, 0.9]m  
3. **처프 지속시간**: 1ms → 20μs
4. **샘플링 레이트**: 1MHz → 2MHz

이렇게 수정하면 1m 이내에서 **약 10cm 분해능**으로 정밀한 측정이 가능합니다! 🎯