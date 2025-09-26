#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
재밍 신호 생성 알고리즘 시각화 데모
다양한 재밍 기법과 그 효과를 시각적으로 보여주는 예제

사용법:
    python jamming_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from jamming_simulator import FMCWRadarSimulator, SpectrogramGenerator

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

class JammingVisualizationDemo:
    """재밍 신호 생성 알고리즘 시각화 데모 클래스"""
    
    def __init__(self):
        """초기화"""
        self.radar_sim = FMCWRadarSimulator()
        self.spec_gen = SpectrogramGenerator()
        
        # 시각화 색상 팔레트
        self.colors = sns.color_palette("husl", 8)
        
    def demo_jamming_parameters(self):
        """재밍 파라미터의 효과 시각화"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 기본 깨끗한 신호 생성
        clean_signal, _ = self.radar_sim.generate_clean_signal()
        
        # 재밍 파라미터 시나리오들
        jamming_scenarios = [
            {'name': 'Power Jamming', 'params': {'power_ratio': 2.5, 'freq_offset': 0, 'time_offset': 0, 'chirp_slope_ratio': 1.0}},
            {'name': 'Frequency Offset', 'params': {'power_ratio': 1.0, 'freq_offset': 50e6, 'time_offset': 0, 'chirp_slope_ratio': 1.0}},
            {'name': 'Time Offset', 'params': {'power_ratio': 1.5, 'freq_offset': 0, 'time_offset': 0.0003, 'chirp_slope_ratio': 1.0}},
            {'name': 'Chirp Slope Change', 'params': {'power_ratio': 1.2, 'freq_offset': 0, 'time_offset': 0, 'chirp_slope_ratio': 1.3}},
        ]
        
        # 깨끗한 신호 플롯
        ax_clean = fig.add_subplot(gs[0, :2])
        time_axis = np.arange(len(clean_signal)) / self.radar_sim.config['sampling_rate'] * 1000  # ms
        ax_clean.plot(time_axis, np.abs(clean_signal), color=self.colors[0], linewidth=1.5, label='Clean Signal')
        ax_clean.set_title('Original Clean Radar Signal', fontsize=14, fontweight='bold')
        ax_clean.set_xlabel('Time (ms)')
        ax_clean.set_ylabel('Magnitude')
        ax_clean.grid(True, alpha=0.3)
        ax_clean.legend()
        
        # 각 재밍 시나리오별 시각화
        for i, scenario in enumerate(jamming_scenarios):
            # 재밍 신호 생성
            jamming_signal = self.radar_sim._generate_jamming_signal(scenario['params'])
            jammed_signal = clean_signal + jamming_signal
            
            # 시간 영역 플롯
            row = (i // 2) + 1
            col = (i % 2) * 2
            
            ax_time = fig.add_subplot(gs[row, col])
            ax_time.plot(time_axis, np.abs(clean_signal), color=self.colors[0], 
                        alpha=0.7, linewidth=1, label='Clean')
            ax_time.plot(time_axis, np.abs(jammed_signal), color=self.colors[i+1], 
                        linewidth=1.5, label='Jammed')
            ax_time.set_title(f'{scenario["name"]} - Time Domain', fontweight='bold')
            ax_time.set_xlabel('Time (ms)')
            ax_time.set_ylabel('Magnitude')
            ax_time.legend()
            ax_time.grid(True, alpha=0.3)
            
            # 주파수 영역 플롯
            ax_freq = fig.add_subplot(gs[row, col+1])
            
            # FFT 계산
            clean_fft = np.abs(np.fft.fft(clean_signal))
            jammed_fft = np.abs(np.fft.fft(jammed_signal))
            freq_axis = np.fft.fftfreq(len(clean_signal), 1/self.radar_sim.config['sampling_rate']) / 1e6  # MHz
            
            # 양의 주파수만 플롯
            positive_mask = freq_axis >= 0
            ax_freq.semilogy(freq_axis[positive_mask], clean_fft[positive_mask], 
                           color=self.colors[0], alpha=0.7, linewidth=1, label='Clean')
            ax_freq.semilogy(freq_axis[positive_mask], jammed_fft[positive_mask], 
                           color=self.colors[i+1], linewidth=1.5, label='Jammed')
            ax_freq.set_title(f'{scenario["name"]} - Frequency Domain', fontweight='bold')
            ax_freq.set_xlabel('Frequency (MHz)')
            ax_freq.set_ylabel('Magnitude (log scale)')
            ax_freq.legend()
            ax_freq.grid(True, alpha=0.3)
        
        plt.suptitle('Jamming Parameter Effects Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig('jamming_parameters_demo.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def demo_multiple_jammers(self):
        """다중 재머 효과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Multiple Jammers Effect Demonstration', fontsize=16, fontweight='bold')
        
        # 기본 신호 생성
        clean_signal, _ = self.radar_sim.generate_clean_signal()
        
        jammer_counts = [1, 2, 4, 6, 8]
        
        for i, num_jammers in enumerate(jammer_counts):
            row = i // 3
            col = i % 3
            
            # 다중 재머로 재밍된 신호 생성
            self.radar_sim.config['num_jammers'] = [num_jammers, num_jammers]
            jammed_signal, jammer_params = self.radar_sim.generate_jammed_signal(clean_signal)
            
            # 시간 영역 플롯
            time_axis = np.arange(len(clean_signal)) / self.radar_sim.config['sampling_rate'] * 1000
            
            axes[row, col].plot(time_axis, np.abs(clean_signal), 
                              color='blue', alpha=0.6, linewidth=1, label='Clean')
            axes[row, col].plot(time_axis, np.abs(jammed_signal), 
                              color='red', linewidth=1.5, alpha=0.8, label=f'{num_jammers} Jammers')
            
            axes[row, col].set_title(f'{num_jammers} Jammer(s)', fontweight='bold')
            axes[row, col].set_xlabel('Time (ms)')
            axes[row, col].set_ylabel('Magnitude')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
            
            # SNR 계산 및 표시
            signal_power = np.mean(np.abs(clean_signal)**2)
            noise_power = np.mean(np.abs(jammed_signal - clean_signal)**2)
            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-12))
            
            axes[row, col].text(0.05, 0.95, f'SNR: {snr_db:.1f} dB', 
                              transform=axes[row, col].transAxes, 
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                              verticalalignment='top')
        
        # 빈 서브플롯 숨기기
        if len(jammer_counts) < 6:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('multiple_jammers_demo.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def demo_spectrogram_comparison(self):
        """스펙트로그램 비교 시각화"""
        # 신호 생성
        clean_signal, _ = self.radar_sim.generate_clean_signal()
        jammed_signal, jammer_params = self.radar_sim.generate_jammed_signal(clean_signal)
        
        # 스펙트로그램 생성
        f_clean, t_clean, spec_clean = self.spec_gen.generate_spectrogram(
            clean_signal, self.radar_sim.config['sampling_rate']
        )
        f_jammed, t_jammed, spec_jammed = self.spec_gen.generate_spectrogram(
            jammed_signal, self.radar_sim.config['sampling_rate']
        )
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Time-Frequency Analysis: Clean vs Jammed Signals', 
                    fontsize=16, fontweight='bold')
        
        # 깨끗한 신호 - 시간 영역
        time_axis = np.arange(len(clean_signal)) / self.radar_sim.config['sampling_rate'] * 1000
        axes[0, 0].plot(time_axis, np.abs(clean_signal), color='blue', linewidth=1.5)
        axes[0, 0].set_title('Clean Signal - Time Domain', fontweight='bold')
        axes[0, 0].set_xlabel('Time (ms)')
        axes[0, 0].set_ylabel('Magnitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 재밍된 신호 - 시간 영역
        axes[0, 1].plot(time_axis, np.abs(jammed_signal), color='red', linewidth=1.5)
        axes[0, 1].set_title('Jammed Signal - Time Domain', fontweight='bold')
        axes[0, 1].set_xlabel('Time (ms)')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 깨끗한 신호 - 스펙트로그램
        im1 = axes[1, 0].imshow(spec_clean, aspect='auto', origin='lower',
                               extent=[t_clean[0]*1000, t_clean[-1]*1000, 
                                     f_clean[0]/1e6, f_clean[-1]/1e6],
                               cmap='viridis')
        axes[1, 0].set_title('Clean Signal - Spectrogram', fontweight='bold')
        axes[1, 0].set_xlabel('Time (ms)')
        axes[1, 0].set_ylabel('Frequency (MHz)')
        plt.colorbar(im1, ax=axes[1, 0], label='Magnitude (dB)')
        
        # 재밍된 신호 - 스펙트로그램
        im2 = axes[1, 1].imshow(spec_jammed, aspect='auto', origin='lower',
                               extent=[t_jammed[0]*1000, t_jammed[-1]*1000, 
                                     f_jammed[0]/1e6, f_jammed[-1]/1e6],
                               cmap='viridis')
        axes[1, 1].set_title('Jammed Signal - Spectrogram', fontweight='bold')
        axes[1, 1].set_xlabel('Time (ms)')
        axes[1, 1].set_ylabel('Frequency (MHz)')
        plt.colorbar(im2, ax=axes[1, 1], label='Magnitude (dB)')
        
        plt.tight_layout()
        plt.savefig('spectrogram_comparison_demo.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def demo_jamming_statistics(self):
        """재밍 통계 분석 시각화"""
        # 다양한 재밍 강도로 실험
        power_ratios = np.linspace(0.1, 3.0, 20)
        snr_values = []
        correlation_values = []
        
        print("재밍 강도별 성능 분석 중...")
        
        for power_ratio in power_ratios:
            # 신호 생성
            clean_signal, _ = self.radar_sim.generate_clean_signal()
            
            # 재밍 파라미터 설정
            jammer_params = {
                'power_ratio': power_ratio,
                'freq_offset': np.random.uniform(-20e6, 20e6),
                'time_offset': np.random.uniform(0, 0.2e-3),
                'chirp_slope_ratio': np.random.uniform(0.9, 1.1)
            }
            
            jamming_signal = self.radar_sim._generate_jamming_signal(jammer_params)
            jammed_signal = clean_signal + jamming_signal
            
            # SNR 계산
            signal_power = np.mean(np.abs(clean_signal)**2)
            noise_power = np.mean(np.abs(jammed_signal - clean_signal)**2)
            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-12))
            snr_values.append(snr_db)
            
            # 상관계수 계산
            correlation = np.corrcoef(np.abs(clean_signal), np.abs(jammed_signal))[0, 1]
            correlation_values.append(correlation)
        
        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Jamming Performance Analysis', fontsize=16, fontweight='bold')
        
        # SNR vs 재밍 전력비
        axes[0].plot(power_ratios, snr_values, 'o-', color='red', linewidth=2, markersize=6)
        axes[0].set_xlabel('Jammer Power Ratio')
        axes[0].set_ylabel('SNR (dB)')
        axes[0].set_title('SNR Degradation vs Jamming Power', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='10 dB threshold')
        axes[0].legend()
        
        # 상관계수 vs 재밍 전력비
        axes[1].plot(power_ratios, correlation_values, 's-', color='blue', linewidth=2, markersize=6)
        axes[1].set_xlabel('Jammer Power Ratio')
        axes[1].set_ylabel('Correlation Coefficient')
        axes[1].set_title('Signal Correlation vs Jamming Power', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='0.5 threshold')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('jamming_statistics_demo.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 통계 요약 출력
        print("\n재밍 분석 통계 요약:")
        print(f"최소 SNR: {min(snr_values):.2f} dB (전력비: {power_ratios[np.argmin(snr_values)]:.2f})")
        print(f"최대 SNR: {max(snr_values):.2f} dB (전력비: {power_ratios[np.argmax(snr_values)]:.2f})")
        print(f"최소 상관계수: {min(correlation_values):.3f} (전력비: {power_ratios[np.argmin(correlation_values)]:.2f})")
        print(f"최대 상관계수: {max(correlation_values):.3f} (전력비: {power_ratios[np.argmax(correlation_values)]:.2f})")

def main():
    """메인 실행 함수"""
    print("재밍 신호 생성 알고리즘 시각화 데모 시작")
    print("=" * 50)
    
    demo = JammingVisualizationDemo()
    
    try:
        print("\n1. 재밍 파라미터 효과 분석...")
        demo.demo_jamming_parameters()
        
        print("\n2. 다중 재머 효과 시연...")
        demo.demo_multiple_jammers()
        
        print("\n3. 스펙트로그램 비교 분석...")
        demo.demo_spectrogram_comparison()
        
        print("\n4. 재밍 통계 분석...")
        demo.demo_jamming_statistics()
        
        print("\n✅ 모든 시각화 데모 완료!")
        print("생성된 이미지 파일:")
        print("  - jamming_parameters_demo.png")
        print("  - multiple_jammers_demo.png")
        print("  - spectrogram_comparison_demo.png")
        print("  - jamming_statistics_demo.png")
        
    except Exception as e:
        print(f"❌ 데모 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()