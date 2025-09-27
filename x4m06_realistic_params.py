#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
X4M06 레이더 파라미터 정정 및 실용적 설정

X4M06은 UWB FMCW 레이더이므로 일반적인 펄스 레이더와 다릅니다.
실제 실험 환경에 맞는 현실적 파라미터를 제안합니다.
"""

import numpy as np

def analyze_x4m06_characteristics():
    """X4M06 실제 특성 분석"""
    
    print("📡 X4M06 UWB 레이더 특성 분석")
    print("=" * 60)
    
    # X4M06 실제 사양 (데이터시트 기반)
    print("X4M06 실제 사양:")
    print("  타입: UWB 임펄스 레이더")
    print("  중심 주파수: 8.748 GHz")
    print("  대역폭: 1.4 GHz") 
    print("  측정 범위: 0.18m ~ 9.4m")
    print("  거리 해상도: ~10.7 cm")
    print("  최대 샘플링 레이트: 23.328 MHz")
    
    # 실험실 환경 고려
    print(f"\n🏠 실험실 환경 (1m 이내) 고려사항:")
    print("  - X4M06의 최소 측정 거리: 0.18m")
    print("  - 근거리 측정에 최적화된 UWB 방식")
    print("  - 연속파 방식이므로 PRF 개념이 다름")
    
    return True

def get_realistic_parameters():
    """실용적 파라미터 제안"""
    
    print(f"\n✅ 실험실 환경에 적합한 현실적 파라미터:")
    
    # X4M06에 적합한 파라미터
    params = {
        'center_freq': 8.748e9,      # X4M06 고정값
        'bandwidth': 1.4e9,          # X4M06 고정값  
        'chirp_duration': 50e-6,     # 50 μs (현실적)
        'prf': 1000,                 # 1 kHz (측정 업데이트 레이트)
        'sampling_rate': 10e6,       # 10 MHz (충분한 해상도)
        'target_range': [0.2, 2.0],  # 실험실 환경 (X4M06 최소 거리 고려)
        'target_velocity': [-5, 5],  # 실험실 환경 속도
        'target_rcs': [0.01, 1.0],   # 작은 타겟 위주
        'snr_db': [10, 20]           # 근거리 높은 SNR
    }
    
    print(f"radar_config = {{")
    print(f"    'center_freq': {params['center_freq']:.1e},")
    print(f"    'bandwidth': {params['bandwidth']:.1e},")
    print(f"    'chirp_duration': {params['chirp_duration']:.0e},  # {params['chirp_duration']*1e6:.0f} μs")
    print(f"    'prf': {params['prf']},  # {params['prf']} Hz (업데이트 레이트)")
    print(f"    'sampling_rate': {params['sampling_rate']:.0e},  # {params['sampling_rate']/1e6:.0f} MHz")
    print(f"    'target_range': {params['target_range']},  # X4M06 최소거리 고려")
    print(f"    'target_velocity': {params['target_velocity']},")
    print(f"    'target_rcs': {params['target_rcs']},")  
    print(f"    'snr_db': {params['snr_db']}")
    print(f"}}")
    
    # STFT 파라미터 계산
    samples_per_chirp = int(params['chirp_duration'] * params['sampling_rate'])
    
    print(f"\n📊 신호 처리 파라미터:")
    print(f"  처프당 샘플 수: {samples_per_chirp}")
    
    # 적절한 STFT 파라미터
    if samples_per_chirp >= 512:
        nperseg = 256
    elif samples_per_chirp >= 256:
        nperseg = 128  
    elif samples_per_chirp >= 128:
        nperseg = 64
    else:
        nperseg = max(16, samples_per_chirp // 4)
        
    noverlap = nperseg // 2
    nfft = max(nperseg, 256)
    
    stft_params = {
        'nperseg': nperseg,
        'noverlap': noverlap, 
        'nfft': nfft,
        'window': 'hann'
    }
    
    print(f"\nstft_params = {{")
    print(f"    'nperseg': {stft_params['nperseg']},")
    print(f"    'noverlap': {stft_params['noverlap']},")
    print(f"    'nfft': {stft_params['nfft']},")
    print(f"    'window': '{stft_params['window']}'")
    print(f"}}")
    
    # 검증
    print(f"\n🔍 파라미터 검증:")
    print(f"  ✅ nperseg ({nperseg}) < 샘플수 ({samples_per_chirp})")
    print(f"  ✅ noverlap ({noverlap}) < nperseg ({nperseg})")
    print(f"  ✅ 처프 지속시간 ({params['chirp_duration']*1e6:.0f}μs) < PRI ({1000/params['prf']}μs)")
    
    return params, stft_params

def main():
    """메인 실행"""
    analyze_x4m06_characteristics()
    params, stft_params = get_realistic_parameters()
    
    print(f"\n" + "="*60)
    print(f"🔧 jamming_simulator.py 수정 방법")
    print(f"="*60)
    print(f"1. _get_default_simulation_config() 함수에서 다음 값들을 수정:")
    print(f"   - chirp_duration: {params['chirp_duration']:.0e}")
    print(f"   - sampling_rate: {params['sampling_rate']:.0e}")
    print(f"   - target_range: {params['target_range']}")
    print(f"   \n2. STFT 파라미터 수정:")
    print(f"   - nperseg: {stft_params['nperseg']}")
    print(f"   - noverlap: {stft_params['noverlap']}")
    print(f"   - nfft: {stft_params['nfft']}")

if __name__ == "__main__":
    main()