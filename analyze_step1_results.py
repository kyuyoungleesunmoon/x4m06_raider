#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1단계 시뮬레이션 실험 결과 분석 스크립트
"""

import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_dataset(session_dir):
    """데이터셋 상세 분석"""
    
    # 경로 설정
    sim_dir = Path(session_dir) / "simulation"
    dataset_file = sim_dir / "radar_jamming_dataset_1000.h5"
    metadata_file = sim_dir / "metadata_1000.json"
    
    print("="*80)
    print("X4M06 레이더 재밍 신호 시뮬레이션 실험 - 1단계 결과 분석")
    print("="*80)
    
    # 1. 메타데이터 분석
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"\n📋 실험 메타데이터:")
    print(f"실험 날짜: {metadata['creation_date']}")
    print(f"총 샘플 수: {len(metadata['samples'])}")
    
    # 레이더 설정 출력
    radar_config = metadata['radar_config']
    print(f"\n📡 레이더 설정:")
    print(f"  중심 주파수: {radar_config['center_freq']/1e9:.3f} GHz")
    print(f"  대역폭: {radar_config['bandwidth']/1e9:.1f} GHz")
    print(f"  처프 지속시간: {radar_config['chirp_duration']*1000:.1f} ms")
    print(f"  샘플링 주파수: {radar_config['sampling_rate']/1e6:.1f} MHz")
    print(f"  PRF: {radar_config['prf']} Hz")
    
    # STFT 설정 출력
    stft_params = metadata['stft_params']
    print(f"\n🔍 STFT 파라미터:")
    print(f"  세그먼트 길이: {stft_params['nperseg']}")
    print(f"  오버랩: {stft_params['noverlap']}")
    print(f"  FFT 포인트: {stft_params['nfft']}")
    print(f"  윈도우: {stft_params['window']}")
    
    # 2. 데이터셋 구조 분석
    with h5py.File(dataset_file, 'r') as f:
        print(f"\n📊 데이터셋 구조:")
        print(f"  데이터셋 키: {list(f.keys())}")
        print(f"  Clean signals 형태: {f['clean_signals'].shape}")
        print(f"  Jammed signals 형태: {f['jammed_signals'].shape}")
        print(f"  Clean spectrograms 형태: {f['clean_spectrograms'].shape}")
        print(f"  Jammed spectrograms 형태: {f['jammed_spectrograms'].shape}")
        
        # 데이터 타입 확인
        print(f"\n💾 데이터 타입:")
        print(f"  Clean signals: {f['clean_signals'].dtype}")
        print(f"  Jammed signals: {f['jammed_signals'].dtype}")
        print(f"  Clean spectrograms: {f['clean_spectrograms'].dtype}")
        print(f"  Jammed spectrograms: {f['jammed_spectrograms'].dtype}")
    
    # 3. 통계 분석
    samples = metadata['samples']
    
    # 목표물 통계
    target_ranges = []
    target_velocities = []
    target_rcs = []
    
    # 재밍 통계
    num_jammers_list = []
    jammer_powers = []
    freq_offsets = []
    snr_values = []
    
    print(f"\n📈 샘플 통계 분석 (총 {len(samples)}개):")
    
    for sample in samples:
        # 목표물 파라미터
        for target in sample['target_params']:
            target_ranges.append(target[0])
            target_velocities.append(target[1])
            target_rcs.append(target[2])
        
        # 재밍 파라미터
        num_jammers_list.append(len(sample['jammer_params']))
        
        for jammer in sample['jammer_params']:
            jammer_powers.append(jammer['power_ratio'])
            freq_offsets.append(jammer['freq_offset'])
    
    # SNR은 레이더 설정에서 가져오기
    snr_range = radar_config['snr_db']
    
    # 목표물 통계 출력
    print(f"\n🎯 목표물 통계:")
    print(f"  거리 범위: {min(target_ranges):.1f} - {max(target_ranges):.1f} m")
    print(f"  거리 평균: {np.mean(target_ranges):.1f} ± {np.std(target_ranges):.1f} m")
    print(f"  속도 범위: {min(target_velocities):.1f} - {max(target_velocities):.1f} m/s")
    print(f"  속도 평균: {np.mean(target_velocities):.1f} ± {np.std(target_velocities):.1f} m/s")
    print(f"  RCS 범위: {min(target_rcs):.2f} - {max(target_rcs):.2f} m²")
    print(f"  RCS 평균: {np.mean(target_rcs):.2f} ± {np.std(target_rcs):.2f} m²")
    
    # 재밍 통계 출력
    print(f"\n⚡ 재밍 통계:")
    print(f"  재머 개수 범위: {min(num_jammers_list)} - {max(num_jammers_list)}")
    print(f"  재머 개수 평균: {np.mean(num_jammers_list):.1f} ± {np.std(num_jammers_list):.1f}")
    print(f"  전력비 범위: {min(jammer_powers):.2f} - {max(jammer_powers):.2f}")
    print(f"  전력비 평균: {np.mean(jammer_powers):.2f} ± {np.std(jammer_powers):.2f}")
    print(f"  주파수 오프셋 범위: {min(freq_offsets)/1e6:.1f} - {max(freq_offsets)/1e6:.1f} MHz")
    print(f"  주파수 오프셋 평균: {np.mean(freq_offsets)/1e6:.1f} ± {np.std(freq_offsets)/1e6:.1f} MHz")
    print(f"  SNR 설정 범위: {snr_range[0]} - {snr_range[1]} dB")
    
    # 4. 분포 분석
    print(f"\n📊 분포 분석:")
    print(f"  재머 개수 분포:")
    for i in range(1, 6):
        count = num_jammers_list.count(i)
        percentage = count / len(num_jammers_list) * 100
        print(f"    {i}개 재머: {count}회 ({percentage:.1f}%)")
    
    return {
        'metadata': metadata,
        'target_ranges': target_ranges,
        'target_velocities': target_velocities,
        'target_rcs': target_rcs,
        'num_jammers': num_jammers_list,
        'jammer_powers': jammer_powers,
        'freq_offsets': freq_offsets,
        'snr_range': snr_range
    }

if __name__ == "__main__":
    session_dir = "experiment_results/session_20250927_190149"
    results = analyze_dataset(session_dir)
    
    print(f"\n✅ 1단계 시뮬레이션 실험 분석 완료!")
    print(f"   세션 디렉토리: {session_dir}")