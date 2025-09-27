#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PRF와 측정 거리 관계 검증 및 수정된 설정 테스트

수정된 레이더 파라미터가 1m 이내 실험에 적합한지 검증합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
def setup_korean_font():
    try:
        font_path = 'C:/Windows/Fonts/malgun.ttf'
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("한글 폰트 설정 실패, 기본 폰트 사용")

setup_korean_font()

def calculate_max_range(prf):
    """PRF로부터 최대 측정 거리 계산"""
    c = 3e8  # 빛의 속도 m/s
    pri = 1 / prf  # Pulse Repetition Interval
    max_range = (c * pri) / 2
    return max_range

def calculate_range_resolution(bandwidth):
    """대역폭으로부터 거리 분해능 계산"""
    c = 3e8
    resolution = c / (2 * bandwidth)
    return resolution

def analyze_radar_config():
    """레이더 설정 분석"""
    print("=" * 80)
    print("🎯 X4M06 레이더 설정 분석 - PRF와 측정 거리 관계")
    print("=" * 80)
    
    # 기존 설정 (문제 있던 것)
    old_config = {
        'name': '기존 설정 (부적합)',
        'prf': 1000,        # 1 kHz
        'bandwidth': 1.4e9, # 1.4 GHz
        'target_range': [5, 50],
        'color': 'red'
    }
    
    # 수정된 설정 (1m 이내 실험용)
    new_config = {
        'name': '수정된 설정 (1m 이내 최적)',
        'prf': 100000,      # 100 kHz
        'bandwidth': 1.4e9, # 1.4 GHz
        'target_range': [0.15, 0.9],
        'color': 'green'
    }
    
    configs = [old_config, new_config]
    
    print(f"\n📊 설정별 분석 결과:")
    print(f"{'설정':<25} {'PRF':<15} {'최대거리':<15} {'분해능':<15} {'타겟범위':<15}")
    print("-" * 85)
    
    for config in configs:
        max_range = calculate_max_range(config['prf'])
        resolution = calculate_range_resolution(config['bandwidth'])
        
        print(f"{config['name']:<25} {config['prf']:>10,} Hz {max_range:>10,.1f} m "
              f"{resolution*100:>10.1f} cm {str(config['target_range']):<15}")
    
    # 시각화
    create_prf_range_plot()
    create_experimental_setup_plot()
    
    return configs

def create_prf_range_plot():
    """PRF와 최대 측정 거리 관계 플롯"""
    prf_values = np.logspace(3, 6, 100)  # 1kHz to 1MHz
    max_ranges = [calculate_max_range(prf) for prf in prf_values]
    
    plt.figure(figsize=(12, 8))
    
    # 메인 플롯
    plt.subplot(2, 1, 1)
    plt.loglog(prf_values, max_ranges, 'b-', linewidth=2, label='최대 측정 거리')
    
    # 중요 포인트 표시
    key_points = [
        (1000, calculate_max_range(1000), '기존 설정\n(150km)', 'red'),
        (100000, calculate_max_range(100000), '수정된 설정\n(1.5km)', 'green'),
        (500000, calculate_max_range(500000), '1m 이내 최적\n(300m)', 'blue')
    ]
    
    for prf, max_range, label, color in key_points:
        plt.plot(prf, max_range, 'o', color=color, markersize=10)
        plt.annotate(label, (prf, max_range), xytext=(10, 10), 
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    plt.axhline(y=1, color='orange', linestyle='--', alpha=0.7, label='1m 실험 목표')
    plt.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10m 경계')
    
    plt.xlabel('PRF (Hz)')
    plt.ylabel('최대 측정 거리 (m)')
    plt.title('PRF와 최대 측정 거리 관계')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 확대된 실용적 범위
    plt.subplot(2, 1, 2)
    prf_practical = np.linspace(10000, 1000000, 100)
    max_ranges_practical = [calculate_max_range(prf) for prf in prf_practical]
    
    plt.plot(prf_practical/1000, max_ranges_practical, 'g-', linewidth=2)
    plt.axhline(y=1, color='red', linestyle='--', label='1m 실험 범위')
    plt.axvline(x=100, color='green', linestyle='--', label='수정된 PRF (100kHz)')
    
    plt.xlabel('PRF (kHz)')
    plt.ylabel('최대 측정 거리 (m)')
    plt.title('실용적 PRF 범위 (10kHz - 1MHz)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 20)
    
    plt.tight_layout()
    plt.savefig('PRF_range_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_experimental_setup_plot():
    """실험 환경 설정 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 기존 설정 (부적합)
    ax1.set_xlim(-10, 60)
    ax1.set_ylim(-5, 5)
    
    # 레이더 위치
    ax1.plot(0, 0, 's', markersize=15, color='blue', label='X4M06 레이더')
    
    # 기존 타겟 범위 (5-50m)
    targets_old = [5, 10, 20, 30, 40, 50]
    for i, dist in enumerate(targets_old):
        ax1.plot(dist, 0, 'o', markersize=8, color='red', alpha=0.7)
        ax1.text(dist, 1, f'{dist}m', ha='center', fontsize=8)
    
    # 1m 실험 범위 표시
    ax1.axvspan(0, 1, alpha=0.2, color='green', label='1m 실험 범위')
    ax1.text(25, 3, '기존 설정: 5-50m\n(실내 실험 부적합)', 
             ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    ax1.set_xlabel('거리 (m)')
    ax1.set_title('기존 설정 - 원거리 측정용')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 수정된 설정 (적합)
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.3, 0.3)
    
    # 레이더 위치
    ax2.plot(0, 0, 's', markersize=15, color='blue', label='X4M06 레이더')
    
    # 수정된 타겟 범위 (0.15-0.9m)
    targets_new = [0.15, 0.25, 0.4, 0.6, 0.8, 0.9]
    target_names = ['작은 물체', '책', '노트북', '의자', '사람(앉음)', '사람(섬)']
    
    for i, (dist, name) in enumerate(zip(targets_new, target_names)):
        ax2.plot(dist, 0, 'o', markersize=10, color='green')
        ax2.text(dist, 0.15 if i%2==0 else -0.15, f'{name}\n{dist*100:.0f}cm', 
                ha='center', fontsize=8, va='center')
    
    # 거리 분해능 표시
    resolution = calculate_range_resolution(1.4e9)
    ax2.text(0.5, 0.25, f'거리 분해능: {resolution*100:.1f}cm\n(1m에서 약 9개 구간)', 
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    ax2.set_xlabel('거리 (m)')
    ax2.set_title('수정된 설정 - 1m 이내 정밀 측정')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('experimental_setup_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def validate_new_settings():
    """수정된 설정 검증"""
    print(f"\n🔍 수정된 설정 검증")
    print("-" * 50)
    
    # 수정된 설정
    new_prf = 100000  # 100 kHz
    new_bandwidth = 1.4e9  # 1.4 GHz
    new_chirp_duration = 8e-6  # 8 μs (PRI의 80%)
    new_target_range = [0.15, 0.9]  # 15-90 cm
    
    # 계산
    max_range = calculate_max_range(new_prf)
    range_resolution = calculate_range_resolution(new_bandwidth)
    num_range_bins = int((new_target_range[1] - new_target_range[0]) / range_resolution)
    
    # PRI 계산
    pri = 1 / new_prf
    
    print(f"📊 수정된 설정 분석:")
    print(f"   PRF: {new_prf:,} Hz = {new_prf/1000:.0f} kHz")
    print(f"   PRI: {pri*1e6:.1f} μs")
    print(f"   처프 지속시간: {new_chirp_duration*1e6:.0f} μs")
    print(f"   최대 측정 거리: {max_range:,.0f} m")
    print(f"   거리 분해능: {range_resolution*100:.1f} cm")
    print(f"   타겟 범위: {new_target_range[0]*100:.0f}-{new_target_range[1]*100:.0f} cm")
    print(f"   측정 가능 거리 구간: {num_range_bins}개")
    
    # 검증 결과
    print(f"\n✅ 검증 결과:")
    
    if max_range >= 10:  # 최소 10m 이상 측정 가능해야 안전
        print(f"   ✅ 최대 측정 거리 적합: {max_range:,.0f}m > 10m")
    else:
        print(f"   ❌ 최대 측정 거리 부족: {max_range:,.0f}m < 10m")
    
    if range_resolution <= 0.15:  # 15cm 이하 분해능 필요
        print(f"   ✅ 거리 분해능 적합: {range_resolution*100:.1f}cm ≤ 15cm")
    else:
        print(f"   ❌ 거리 분해능 부족: {range_resolution*100:.1f}cm > 15cm")
    
    if new_chirp_duration < pri:
        print(f"   ✅ 처프 지속시간 적합: {new_chirp_duration*1e6:.0f}μs < {pri*1e6:.1f}μs (PRI)")
    else:
        print(f"   ❌ 처프 지속시간 과도: {new_chirp_duration*1e6:.0f}μs ≥ {pri*1e6:.1f}μs (PRI)")
    
    if num_range_bins >= 5:
        print(f"   ✅ 측정 구간 적합: {num_range_bins}개 ≥ 5개")
    else:
        print(f"   ❌ 측정 구간 부족: {num_range_bins}개 < 5개")
    
    return {
        'max_range': max_range,
        'range_resolution': range_resolution,
        'num_range_bins': num_range_bins,
        'pri': pri
    }

def main():
    """메인 실행 함수"""
    print("🚀 PRF-거리 분석 및 설정 검증 시작")
    
    # 1. 기본 분석
    configs = analyze_radar_config()
    
    # 2. 수정된 설정 검증
    validation_results = validate_new_settings()
    
    # 3. 권장사항
    print(f"\n💡 실험 권장사항:")
    print(f"   1. 수정된 PRF (100kHz) 사용으로 1.5km 최대 측정")
    print(f"   2. 15-90cm 범위에서 {validation_results['range_resolution']*100:.1f}cm 분해능")
    print(f"   3. 실내 실험에 최적화된 설정")
    print(f"   4. 기존 대비 100배 향상된 PRF로 정밀 측정 가능")
    
    print(f"\n🎯 즉시 실행 가능:")
    print(f"   python main_experiment.py --mode simulation --samples 1000")
    print(f"   (수정된 설정이 자동 적용됨)")

if __name__ == "__main__":
    main()