#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X4M06 5미터 환경 CW 재밍 파라미터 계산기
Created: 2024-09-27
Purpose: 5미터 이내 실내 환경에서 CW 재밍 시뮬레이션을 위한 최적 파라미터 계산
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class X4M06CWJammingCalculator:
    """X4M06 레이더 5미터 환경 CW 재밍 파라미터 계산기"""
    
    def __init__(self):
        # X4M06 기본 스펙
        self.center_freq = 8.748e9  # Hz (8.748 GHz)
        self.bandwidth = 1.4e9      # Hz (1.4 GHz)
        self.freq_min = self.center_freq - self.bandwidth/2  # 8.048 GHz
        self.freq_max = self.center_freq + self.bandwidth/2  # 9.448 GHz
        
        # 5미터 환경 설정
        self.max_range = 5.0        # 미터
        self.min_range = 0.18       # 미터 (X4M06 최소 거리)
        self.c = 3e8                # 빛의 속도
        
        print("=" * 60)
        print("🎯 X4M06 5미터 환경 CW 재밍 파라미터 계산기")
        print("=" * 60)
        print(f"📡 레이더 스펙:")
        print(f"   - 중심 주파수: {self.center_freq/1e9:.3f} GHz")
        print(f"   - 대역폭: {self.bandwidth/1e9:.1f} GHz") 
        print(f"   - 주파수 범위: {self.freq_min/1e9:.3f} - {self.freq_max/1e9:.3f} GHz")
        print(f"   - 측정 범위: {self.min_range:.2f} - {self.max_range:.1f} m")
        
    def calculate_optimal_params(self):
        """5미터 환경에 최적화된 레이더 파라미터 계산"""
        
        # 5미터 환경에 맞는 PRF 계산
        # PRF = c / (2 * max_range * safety_factor)
        safety_factor = 1.2  # 안전 계수
        prf_max = self.c / (2 * self.max_range * safety_factor)
        prf = 15000  # 15 kHz (5m 환경에 적합)
        
        # 임펄스 레이더 파라미터
        pulse_width = 2e-9      # 2ns (UWB 임펄스)
        sampling_rate = 23.328e9 # 23.328 GHz (X4M06 표준)
        
        # 거리 해상도
        range_resolution = self.c / (2 * self.bandwidth)
        
        # 시간 파라미터
        max_delay = 2 * self.max_range / self.c
        samples_per_frame = int(sampling_rate * max_delay)
        
        params = {
            'prf': prf,
            'pulse_width': pulse_width,
            'sampling_rate': sampling_rate,
            'range_resolution': range_resolution,
            'max_delay': max_delay,
            'samples_per_frame': samples_per_frame,
            'center_frequency': self.center_freq,
            'bandwidth': self.bandwidth
        }
        
        print(f"\n📊 5미터 환경 최적화 파라미터:")
        print(f"   - PRF: {prf/1000:.1f} kHz")
        print(f"   - 펄스 폭: {pulse_width*1e9:.1f} ns")
        print(f"   - 샘플링 주파수: {sampling_rate/1e9:.3f} GHz")
        print(f"   - 거리 해상도: {range_resolution*100:.2f} cm")
        print(f"   - 최대 지연시간: {max_delay*1e9:.1f} ns")
        print(f"   - 프레임당 샘플수: {samples_per_frame}")
        
        return params
    
    def calculate_cw_jamming_params(self):
        """CW 재밍 파라미터 계산"""
        
        print(f"\n⚔️  CW 재밍 파라미터 계산:")
        
        # CW 재밍 주파수 후보들 (X4M06 대역 내)
        jamming_freqs = [
            self.center_freq - 0.3e9,  # 8.448 GHz
            self.center_freq,          # 8.748 GHz (중심)
            self.center_freq + 0.3e9,  # 9.048 GHz
        ]
        
        # CW 재밍 전력 레벨 (신호 대비)
        jamming_powers = [0.5, 1.0, 2.0, 5.0, 10.0]  # 배수
        
        cw_params = {
            'jamming_frequencies': jamming_freqs,
            'jamming_powers': jamming_powers,
            'jamming_types': ['single_tone', 'multi_tone', 'swept_cw'],
            'interference_scenarios': {
                'mild': {'power_ratio': 0.5, 'frequencies': [jamming_freqs[0]]},
                'moderate': {'power_ratio': 2.0, 'frequencies': [jamming_freqs[1]]},
                'severe': {'power_ratio': 10.0, 'frequencies': jamming_freqs}
            }
        }
        
        print(f"   - 재밍 주파수 후보:")
        for i, freq in enumerate(jamming_freqs):
            print(f"     {i+1}. {freq/1e9:.3f} GHz")
        
        print(f"   - 재밍 전력 비율: {jamming_powers}")
        print(f"   - 재밍 시나리오:")
        for scenario, params in cw_params['interference_scenarios'].items():
            print(f"     - {scenario.upper()}: {params['power_ratio']}배 전력, "
                  f"{len(params['frequencies'])}개 주파수")
        
        return cw_params
    
    def calculate_detection_thresholds(self):
        """탐지 임계값 계산"""
        
        # 5미터 환경에서의 신호 감쇠
        distances = np.linspace(self.min_range, self.max_range, 100)
        
        # 자유공간 경로 손실
        path_loss_db = 20 * np.log10(4 * np.pi * distances * self.center_freq / self.c)
        
        # 신호 전력 (임의의 기준값에서 계산)
        tx_power_dbm = 0  # 0 dBm 기준
        rx_power_dbm = tx_power_dbm - path_loss_db
        
        # SNR 기반 탐지 임계값
        noise_floor_dbm = -90  # 일반적인 노이즈 플로어
        snr_threshold_db = 10   # 10dB SNR 임계값
        detection_threshold_dbm = noise_floor_dbm + snr_threshold_db
        
        # 탐지 가능한 최대 거리
        max_detection_range = None
        for i, power in enumerate(rx_power_dbm):
            if power < detection_threshold_dbm:
                max_detection_range = distances[i]
                break
        
        if max_detection_range is None:
            max_detection_range = self.max_range
        
        threshold_params = {
            'path_loss_db': path_loss_db,
            'rx_power_dbm': rx_power_dbm,
            'detection_threshold_dbm': detection_threshold_dbm,
            'max_detection_range': max_detection_range,
            'distances': distances
        }
        
        print(f"\n🎯 탐지 성능 분석:")
        print(f"   - 노이즈 플로어: {noise_floor_dbm} dBm")
        print(f"   - SNR 임계값: {snr_threshold_db} dB")
        print(f"   - 탐지 임계값: {detection_threshold_dbm} dBm")
        print(f"   - 최대 탐지 거리: {max_detection_range:.2f} m")
        print(f"   - 5m에서 경로손실: {path_loss_db[-1]:.1f} dB")
        
        return threshold_params
    
    def visualize_parameters(self, radar_params, cw_params, threshold_params):
        """파라미터 시각화"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('X4M06 5미터 환경 CW 재밍 파라미터 분석', fontsize=16, fontweight='bold')
        
        # 1. 주파수 스펙트럼
        freqs = np.linspace(self.freq_min, self.freq_max, 1000) / 1e9
        spectrum = np.exp(-0.5 * ((freqs - self.center_freq/1e9) / (self.bandwidth/2e9))**2)
        
        ax1.plot(freqs, spectrum, 'b-', linewidth=2, label='X4M06 스펙트럼')
        for i, jf in enumerate(cw_params['jamming_frequencies']):
            ax1.axvline(jf/1e9, color='red', linestyle='--', alpha=0.7, 
                       label=f'CW 재밍 {i+1}' if i == 0 else '')
        ax1.set_xlabel('주파수 (GHz)')
        ax1.set_ylabel('정규화된 전력')
        ax1.set_title('주파수 도메인: 레이더 스펙트럼 vs CW 재밍')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 거리별 수신 전력
        distances = threshold_params['distances']
        rx_power = threshold_params['rx_power_dbm']
        threshold = threshold_params['detection_threshold_dbm']
        
        ax2.plot(distances, rx_power, 'g-', linewidth=2, label='수신 전력')
        ax2.axhline(threshold, color='red', linestyle='--', label='탐지 임계값')
        ax2.axvline(threshold_params['max_detection_range'], color='orange', 
                   linestyle=':', label='최대 탐지 거리')
        ax2.set_xlabel('거리 (m)')
        ax2.set_ylabel('수신 전력 (dBm)')
        ax2.set_title('거리별 수신 전력 및 탐지 임계값')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. CW 재밍 시나리오
        scenarios = list(cw_params['interference_scenarios'].keys())
        power_ratios = [cw_params['interference_scenarios'][s]['power_ratio'] 
                       for s in scenarios]
        freq_counts = [len(cw_params['interference_scenarios'][s]['frequencies']) 
                      for s in scenarios]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        ax3.bar(x - width/2, power_ratios, width, label='전력 비율', alpha=0.7)
        ax3.bar(x + width/2, freq_counts, width, label='주파수 개수', alpha=0.7)
        ax3.set_xlabel('재밍 시나리오')
        ax3.set_ylabel('값')
        ax3.set_title('CW 재밍 시나리오별 파라미터')
        ax3.set_xticks(x)
        ax3.set_xticklabels([s.upper() for s in scenarios])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 시간 도메인 파라미터
        time_params = ['PRF (kHz)', '펄스폭 (ns)', '샘플링 (GHz)', '거리해상도 (cm)']
        time_values = [
            radar_params['prf']/1000,
            radar_params['pulse_width']*1e9,
            radar_params['sampling_rate']/1e9,
            radar_params['range_resolution']*100
        ]
        
        bars = ax4.bar(time_params, time_values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        ax4.set_ylabel('값')
        ax4.set_title('5미터 환경 최적화 파라미터')
        ax4.tick_params(axis='x', rotation=45)
        
        # 막대 위에 값 표시
        for bar, value in zip(bars, time_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(time_values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('x4m06_5m_cw_jamming_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_report_data(self):
        """리포트 생성용 데이터 수집"""
        
        radar_params = self.calculate_optimal_params()
        cw_params = self.calculate_cw_jamming_params()
        threshold_params = self.calculate_detection_thresholds()
        
        # 시각화
        fig = self.visualize_parameters(radar_params, cw_params, threshold_params)
        
        report_data = {
            'radar_params': radar_params,
            'cw_params': cw_params,
            'threshold_params': threshold_params,
            'environment': {
                'max_range': self.max_range,
                'min_range': self.min_range,
                'center_freq': self.center_freq,
                'bandwidth': self.bandwidth
            }
        }
        
        print(f"\n✅ 파라미터 계산 완료!")
        print(f"📊 그래프 저장: x4m06_5m_cw_jamming_analysis.png")
        
        return report_data

def main():
    """메인 실행 함수"""
    calculator = X4M06CWJammingCalculator()
    report_data = calculator.generate_report_data()
    
    print(f"\n🎯 다음 단계:")
    print(f"   1. CW 재밍 시뮬레이터 업데이트")
    print(f"   2. 5미터 환경 데이터셋 생성")
    print(f"   3. 성능 분석 리포트 작성")
    
    return report_data

if __name__ == "__main__":
    report_data = main()