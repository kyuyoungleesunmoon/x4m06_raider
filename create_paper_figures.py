#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1단계 실험 결과 시각화 생성 스크립트
논문용 고품질 그래프 및 차트 생성
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import json
from pathlib import Path

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정"""
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'
        print("한글 폰트 설정 완료")
    except Exception as e:
        print(f"폰트 설정 오류: {e}")
        plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

def create_comprehensive_analysis_plots(session_dir, output_dir="paper_figures"):
    """논문용 종합 분석 플롯 생성"""
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 데이터 로드
    sim_dir = Path(session_dir) / "simulation"
    dataset_file = sim_dir / "radar_jamming_dataset_1000.h5"
    metadata_file = sim_dir / "metadata_1000.json"
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 통계 데이터 추출
    samples = metadata['samples']
    target_ranges = []
    target_velocities = []
    target_rcs = []
    num_jammers = []
    jammer_powers = []
    freq_offsets = []
    
    for sample in samples:
        for target in sample['target_params']:
            target_ranges.append(target[0])
            target_velocities.append(target[1])
            target_rcs.append(target[2])
        
        num_jammers.append(len(sample['jammer_params']))
        
        for jammer in sample['jammer_params']:
            jammer_powers.append(jammer['power_ratio'])
            freq_offsets.append(jammer['freq_offset'] / 1e6)  # MHz로 변환
    
    # 1. 목표물 특성 분포 (Figure 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('목표물 특성 분포 분석', fontsize=16, fontweight='bold')
    
    # 거리 분포
    axes[0, 0].hist(target_ranges, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('거리 (m)')
    axes[0, 0].set_ylabel('빈도수')
    axes[0, 0].set_title('목표물 거리 분포')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 속도 분포
    axes[0, 1].hist(target_velocities, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('속도 (m/s)')
    axes[0, 1].set_ylabel('빈도수')
    axes[0, 1].set_title('목표물 속도 분포')
    axes[0, 1].grid(True, alpha=0.3)
    
    # RCS 분포
    axes[1, 0].hist(target_rcs, bins=20, alpha=0.7, color='salmon', edgecolor='black')
    axes[1, 0].set_xlabel('RCS (m²)')
    axes[1, 0].set_ylabel('빈도수')
    axes[1, 0].set_title('목표물 RCS 분포')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 재머 개수 분포
    jammer_counts = list(range(1, 6))
    jammer_freq = [num_jammers.count(i) for i in jammer_counts]
    
    bars = axes[1, 1].bar(jammer_counts, jammer_freq, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('재머 개수')
    axes[1, 1].set_ylabel('샘플 수')
    axes[1, 1].set_title('재머 개수 분포')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 각 바 위에 값 표시
    for bar, freq in zip(bars, jammer_freq):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{freq}\\n({freq/len(samples)*100:.1f}%)', 
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure1_target_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 재밍 특성 분석 (Figure 2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('재밍 신호 특성 분석', fontsize=16, fontweight='bold')
    
    # 전력비 분포
    axes[0].hist(jammer_powers, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[0].set_xlabel('전력비')
    axes[0].set_ylabel('빈도수')
    axes[0].set_title('재머 전력비 분포')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(np.mean(jammer_powers), color='red', linestyle='--', 
                   label=f'평균: {np.mean(jammer_powers):.2f}')
    axes[0].legend()
    
    # 주파수 오프셋 분포
    axes[1].hist(freq_offsets, bins=20, alpha=0.7, color='gold', edgecolor='black')
    axes[1].set_xlabel('주파수 오프셋 (MHz)')
    axes[1].set_ylabel('빈도수')
    axes[1].set_title('주파수 오프셋 분포')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(np.mean(freq_offsets), color='red', linestyle='--', 
                   label=f'평균: {np.mean(freq_offsets):.1f} MHz')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure2_jamming_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 신호 샘플 시각화 (Figure 3)
    with h5py.File(dataset_file, 'r') as f:
        # 첫 번째 샘플 로드
        clean_signal = f['clean_signals'][0]
        jammed_signal = f['jammed_signals'][0]
        clean_spec = f['clean_spectrograms'][0]
        jammed_spec = f['jammed_spectrograms'][0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('레이더 신호 및 스펙트로그램 비교 (샘플 #1)', fontsize=16, fontweight='bold')
    
    # 시간 축 생성
    time_axis = np.arange(len(clean_signal)) / metadata['radar_config']['sampling_rate'] * 1000
    
    # 깨끗한 신호
    axes[0, 0].plot(time_axis, np.abs(clean_signal), 'b-', linewidth=1.5, label='Clean Signal')
    axes[0, 0].set_xlabel('시간 (ms)')
    axes[0, 0].set_ylabel('진폭')
    axes[0, 0].set_title('원본 신호 (시간 영역)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 재밍된 신호
    axes[0, 1].plot(time_axis, np.abs(jammed_signal), 'r-', linewidth=1.5, label='Jammed Signal')
    axes[0, 1].set_xlabel('시간 (ms)')
    axes[0, 1].set_ylabel('진폭')
    axes[0, 1].set_title('재밍된 신호 (시간 영역)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 스펙트로그램 주파수 축 (kHz로 변환)
    freq_axis = np.linspace(0, metadata['radar_config']['sampling_rate']/2/1000, clean_spec.shape[0])
    time_spec_axis = np.linspace(0, metadata['radar_config']['chirp_duration']*1000, clean_spec.shape[1])
    
    # 깨끗한 신호 스펙트로그램
    im1 = axes[1, 0].imshow(clean_spec, aspect='auto', origin='lower', 
                           extent=[time_spec_axis[0], time_spec_axis[-1], 
                                  freq_axis[0], freq_axis[-1]], cmap='viridis')
    axes[1, 0].set_xlabel('시간 (ms)')
    axes[1, 0].set_ylabel('주파수 (kHz)')
    axes[1, 0].set_title('원본 신호 스펙트로그램')
    plt.colorbar(im1, ax=axes[1, 0], label='크기 (dB)')
    
    # 재밍된 신호 스펙트로그램
    im2 = axes[1, 1].imshow(jammed_spec, aspect='auto', origin='lower',
                           extent=[time_spec_axis[0], time_spec_axis[-1], 
                                  freq_axis[0], freq_axis[-1]], cmap='viridis')
    axes[1, 1].set_xlabel('시간 (ms)')
    axes[1, 1].set_ylabel('주파수 (kHz)')
    axes[1, 1].set_title('재밍된 신호 스펙트로그램')
    plt.colorbar(im2, ax=axes[1, 1], label='크기 (dB)')
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure3_signal_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 데이터셋 요약 통계표 (Figure 4)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('1단계 실험 데이터셋 요약 통계', fontsize=16, fontweight='bold')
    
    # 통계 데이터 준비
    stats_data = [
        ['파라미터', '최솟값', '최댓값', '평균값', '표준편차', '단위'],
        ['목표물 거리', f'{min(target_ranges):.1f}', f'{max(target_ranges):.1f}', 
         f'{np.mean(target_ranges):.1f}', f'{np.std(target_ranges):.1f}', 'm'],
        ['목표물 속도', f'{min(target_velocities):.1f}', f'{max(target_velocities):.1f}', 
         f'{np.mean(target_velocities):.1f}', f'{np.std(target_velocities):.1f}', 'm/s'],
        ['목표물 RCS', f'{min(target_rcs):.2f}', f'{max(target_rcs):.2f}', 
         f'{np.mean(target_rcs):.2f}', f'{np.std(target_rcs):.2f}', 'm²'],
        ['재머 전력비', f'{min(jammer_powers):.2f}', f'{max(jammer_powers):.2f}', 
         f'{np.mean(jammer_powers):.2f}', f'{np.std(jammer_powers):.2f}', '-'],
        ['주파수 오프셋', f'{min(freq_offsets):.1f}', f'{max(freq_offsets):.1f}', 
         f'{np.mean(freq_offsets):.1f}', f'{np.std(freq_offsets):.1f}', 'MHz'],
        ['재머 개수', f'{min(num_jammers)}', f'{max(num_jammers)}', 
         f'{np.mean(num_jammers):.1f}', f'{np.std(num_jammers):.1f}', '개']
    ]
    
    # 테이블 생성
    table = ax.table(cellText=stats_data[1:], colLabels=stats_data[0], 
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 헤더 스타일링
    for i in range(len(stats_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 데이터 행 스타일링
    for i in range(1, len(stats_data)):
        for j in range(len(stats_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.axis('off')
    plt.savefig(output_path / 'figure4_summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\\n✅ 논문용 시각화 완료!")
    print(f"저장 위치: {output_path.absolute()}")
    print(f"생성된 파일:")
    print(f"  - figure1_target_characteristics.png")
    print(f"  - figure2_jamming_characteristics.png")
    print(f"  - figure3_signal_comparison.png")
    print(f"  - figure4_summary_statistics.png")

if __name__ == "__main__":
    session_dir = "experiment_results/session_20250927_190149"
    create_comprehensive_analysis_plots(session_dir)