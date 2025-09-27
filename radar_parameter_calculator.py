#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
X4M06 레이더 PRF와 측정 거리 계산 및 검증

펄스 반복 주파수(PRF)와 최대 측정 거리의 관계를 분석하고
실제 실험 환경에 맞는 파라미터를 계산합니다.
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_radar_parameters():
    """레이더 파라미터 계산 및 검증"""
    
    print("📡 X4M06 레이더 파라미터 분석")
    print("=" * 60)
    
    # 기본 상수
    c = 3e8  # 광속 (m/s)
    
    # 현재 설정값들
    current_prf = 1000  # Hz
    current_chirp_duration = 1e-3  # 1ms
    center_freq = 8.748e9  # Hz
    bandwidth = 1.4e9  # Hz
    
    print(f"현재 설정값:")
    print(f"  PRF: {current_prf:,} Hz")
    print(f"  처프 지속시간: {current_chirp_duration*1000:.3f} ms")
    print(f"  중심 주파수: {center_freq/1e9:.3f} GHz")
    print(f"  대역폭: {bandwidth/1e9:.1f} GHz")
    
    # PRF와 최대 거리 관계 계산
    print(f"\n🎯 PRF와 최대 측정 거리 관계:")
    
    # PRI (Pulse Repetition Interval) 계산
    pri = 1 / current_prf
    print(f"  PRI (펄스 반복 간격): {pri*1000:.3f} ms")
    
    # 최대 명확 거리 (Maximum Unambiguous Range)
    max_range = (c * pri) / 2
    print(f"  최대 명확 거리: {max_range:.1f} m")
    
    # 문제점 확인
    print(f"\n⚠️  현재 설정의 문제점:")
    if current_chirp_duration >= pri:
        print(f"  ❌ 처프 지속시간({current_chirp_duration*1000:.3f}ms) ≥ PRI({pri*1000:.3f}ms)")
        print(f"     → 다음 펄스가 오기 전에 현재 펄스가 끝나지 않음!")
    
    if max_range > 100:
        print(f"  ⚠️  최대 측정 거리({max_range:.1f}m)가 실험실 환경에 비해 너무 큼")
        print(f"     → 1m 이내 실험에는 과도한 설정")
    
    # 실험실 환경에 적합한 파라미터 제안
    print(f"\n🔧 실험실 환경(1m 이내)에 적합한 파라미터:")
    
    # 목표: 최대 거리 2~5m 정도로 설정
    target_max_ranges = [2, 5, 10]
    
    for target_range in target_max_ranges:
        required_pri = (2 * target_range) / c
        required_prf = 1 / required_pri
        
        print(f"\n  목표 최대 거리: {target_range}m")
        print(f"    필요한 PRI: {required_pri*1e6:.1f} μs")
        print(f"    필요한 PRF: {required_prf:,.0f} Hz")
        
        # 처프 지속시간 제약 확인
        max_chirp_duration = required_pri * 0.8  # PRI의 80%만 사용
        print(f"    최대 처프 지속시간: {max_chirp_duration*1e6:.1f} μs")
    
    # 권장 설정 계산
    print(f"\n✅ 권장 설정 (1m 이내 실험용):")
    
    # 목표: 5m 최대 거리, 여유있는 설정
    recommended_max_range = 5  # m
    recommended_pri = (2 * recommended_max_range) / c
    recommended_prf = 1 / recommended_pri
    recommended_chirp_duration = recommended_pri * 0.5  # PRI의 50% 사용
    
    print(f"  PRF: {recommended_prf:,.0f} Hz")
    print(f"  PRI: {recommended_pri*1e6:.1f} μs")  
    print(f"  처프 지속시간: {recommended_chirp_duration*1e6:.1f} μs")
    print(f"  최대 측정 거리: {recommended_max_range} m")
    
    # 샘플링 관련 계산
    print(f"\n📊 샘플링 파라미터:")
    sampling_rate = 1e6  # 1MHz
    samples_per_chirp = int(recommended_chirp_duration * sampling_rate)
    
    print(f"  샘플링 레이트: {sampling_rate/1e6:.1f} MHz")
    print(f"  처프당 샘플 수: {samples_per_chirp}")
    
    if samples_per_chirp < 100:
        print(f"  ⚠️  샘플 수가 적습니다. STFT 분석에 부족할 수 있음")
        
        # 더 많은 샘플을 위한 대안
        alternative_sampling_rate = 10e6  # 10MHz
        alternative_samples = int(recommended_chirp_duration * alternative_sampling_rate)
        print(f"  대안: 샘플링 레이트 {alternative_sampling_rate/1e6:.1f} MHz → {alternative_samples} 샘플")
    
    return {
        'recommended_prf': recommended_prf,
        'recommended_chirp_duration': recommended_chirp_duration,
        'recommended_sampling_rate': max(sampling_rate, 10e6),
        'samples_per_chirp': max(samples_per_chirp, 100),
        'max_range': recommended_max_range
    }

def calculate_stft_parameters(samples_per_chirp):
    """STFT 파라미터 계산"""
    
    print(f"\n🎵 STFT 파라미터 계산:")
    print(f"  처프당 샘플 수: {samples_per_chirp}")
    
    # nperseg는 샘플 수보다 작아야 함
    if samples_per_chirp >= 256:
        nperseg = 256
    elif samples_per_chirp >= 128:
        nperseg = 128
    elif samples_per_chirp >= 64:
        nperseg = 64
    else:
        nperseg = max(8, samples_per_chirp // 4)
    
    # noverlap은 nperseg보다 작아야 함
    noverlap = nperseg // 2
    
    # nfft는 nperseg보다 크거나 같아야 함
    nfft = max(nperseg, 512)
    
    print(f"  권장 nperseg: {nperseg}")
    print(f"  권장 noverlap: {noverlap}")
    print(f"  권장 nfft: {nfft}")
    
    # 검증
    if noverlap >= nperseg:
        print(f"  ❌ noverlap({noverlap}) >= nperseg({nperseg})")
        noverlap = nperseg - 1
        print(f"  수정된 noverlap: {noverlap}")
    
    if nperseg > samples_per_chirp:
        print(f"  ❌ nperseg({nperseg}) > 샘플수({samples_per_chirp})")
        nperseg = samples_per_chirp
        noverlap = nperseg // 2
        print(f"  수정된 nperseg: {nperseg}")
        print(f"  수정된 noverlap: {noverlap}")
    
    return {
        'nperseg': nperseg,
        'noverlap': noverlap,
        'nfft': nfft
    }

def plot_range_vs_prf():
    """PRF vs 최대 거리 관계 시각화"""
    
    c = 3e8
    prf_range = np.logspace(2, 6, 100)  # 100Hz ~ 1MHz
    max_ranges = (c / (2 * prf_range))
    
    plt.figure(figsize=(10, 6))
    plt.loglog(prf_range, max_ranges)
    plt.axhline(y=1, color='r', linestyle='--', label='1m (실험 거리)')
    plt.axhline(y=5, color='orange', linestyle='--', label='5m (권장 최대)')
    plt.axvline(x=1000, color='b', linestyle='--', label='현재 PRF (1kHz)')
    plt.axvline(x=30000, color='g', linestyle='--', label='권장 PRF (~30kHz)')
    
    plt.xlabel('PRF (Hz)')
    plt.ylabel('최대 측정 거리 (m)')
    plt.title('PRF vs 최대 측정 거리 관계')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 한글 폰트 설정
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    plt.tight_layout()
    plt.savefig('prf_range_relationship.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 그래프 저장: prf_range_relationship.png")

def main():
    """메인 실행 함수"""
    
    # 파라미터 계산
    params = calculate_radar_parameters()
    
    # STFT 파라미터 계산
    stft_params = calculate_stft_parameters(params['samples_per_chirp'])
    
    # 시각화
    plot_range_vs_prf()
    
    # 최종 권장 설정 출력
    print(f"\n" + "="*60)
    print(f"🎯 최종 권장 설정 (jamming_simulator.py 수정용)")
    print(f"="*60)
    
    print(f"radar_config = {{")
    print(f"    'center_freq': 8.748e9,")
    print(f"    'bandwidth': 1.4e9,")
    print(f"    'chirp_duration': {params['recommended_chirp_duration']:.2e},  # {params['recommended_chirp_duration']*1e6:.1f} μs")
    print(f"    'prf': {params['recommended_prf']:.0f},  # {params['recommended_prf']/1000:.1f} kHz")
    print(f"    'sampling_rate': {params['recommended_sampling_rate']:.0e},  # {params['recommended_sampling_rate']/1e6:.1f} MHz")
    print(f"    'target_range': [0.5, {params['max_range']}],  # 실험실 환경에 적합")
    print(f"    # ... 기타 파라미터")
    print(f"}}")
    
    print(f"\nstft_params = {{")
    print(f"    'nperseg': {stft_params['nperseg']},")
    print(f"    'noverlap': {stft_params['noverlap']},")
    print(f"    'nfft': {stft_params['nfft']},")
    print(f"    'window': 'hann'")
    print(f"}}")
    
    return params, stft_params

if __name__ == "__main__":
    main()