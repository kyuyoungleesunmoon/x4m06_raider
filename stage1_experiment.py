#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1단계 실험: 1미터 범위 내 타겟 특화 시뮬레이션
X4M06 레이더 재밍 신호 생성 및 분석

목표:
- 1미터 범위 내 고정밀 타겟 탐지
- 다양한 재밍 시나리오 시뮬레이션
- 성능 분석 및 시각화
- 2단계 하드웨어 실험 기준점 설정
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
import h5py
import json
import time
from datetime import datetime
import os
from tqdm import tqdm

# 한글 폰트 설정
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

class ShortRangeRadarSimulator:
    """1미터 범위 특화 FMCW 레이더 시뮬레이터"""
    
    def __init__(self):
        """1미터 범위에 최적화된 레이더 파라미터 설정"""
        self.config = {
            # 기본 레이더 파라미터 (1미터 범위 최적화)
            'center_frequency': 60e9,        # 60GHz
            'bandwidth': 1.5e9,              # 1.5GHz (거리 분해능 0.1m)
            'chirp_duration': 0.001,         # 1ms
            'sampling_rate': 23.328e6,       # 23.328 MHz
            'max_range': 1.0,                # 1미터 최대 거리
            'range_resolution': 0.1,         # 10cm 거리 분해능
            
            # 타겟 파라미터
            'target_distances': [0.2, 0.5, 0.8],  # 20cm, 50cm, 80cm
            'target_rcs': [0.1, 0.2, 0.15],       # RCS (m²)
            'target_velocities': [0, 0.5, -0.3],  # 속도 (m/s)
            
            # 재밍 파라미터 (1미터 범위 특화)
            'power_ratios': [0.2, 0.5, 1.0, 2.0, 5.0],
            'freq_offsets': [-10e6, -5e6, 0, 5e6, 10e6],  # ±10MHz
            'time_delays': [0, 1e-9, 3.33e-9, 6.67e-9],   # 0~1m 대응
            'chirp_slope_ratios': [0.9, 0.95, 1.0, 1.05, 1.1],
            
            # 시뮬레이션 설정
            'num_samples': 1000,
            'snr_range': [10, 30],           # dB
            'noise_level': -80,              # dBm
        }
        
        # 계산된 파라미터
        self.samples_per_chirp = int(self.config['chirp_duration'] * 
                                   self.config['sampling_rate'])
        self.range_bins = int(self.config['max_range'] / 
                            self.config['range_resolution'])
        
        print(f"📡 1미터 범위 레이더 시뮬레이터 초기화 완료")
        print(f"   - 거리 분해능: {self.config['range_resolution']*100:.1f}cm")
        print(f"   - 거리 빈 수: {self.range_bins}개")
        print(f"   - 샘플 수: {self.samples_per_chirp}")
    
    def generate_clean_signal(self, target_config=None):
        """깨끗한 FMCW 레이더 신호 생성"""
        if target_config is None:
            target_config = {
                'distances': self.config['target_distances'],
                'rcs': self.config['target_rcs'],
                'velocities': self.config['target_velocities']
            }
        
        # 시간 축
        t = np.linspace(0, self.config['chirp_duration'], self.samples_per_chirp)
        
        # 처프 기울기
        chirp_slope = self.config['bandwidth'] / self.config['chirp_duration']
        
        # 송신 신호 (처프)
        tx_signal = np.exp(1j * 2 * np.pi * (
            self.config['center_frequency'] * t + 
            0.5 * chirp_slope * t**2
        ))
        
        # 수신 신호 초기화
        rx_signal = np.zeros_like(tx_signal, dtype=complex)
        
        # 각 타겟에서의 반사 신호
        for dist, rcs, vel in zip(target_config['distances'], 
                                target_config['rcs'], 
                                target_config['velocities']):
            # 왕복 시간 지연
            time_delay = 2 * dist / 3e8
            
            # 도플러 주파수
            doppler_freq = 2 * vel * self.config['center_frequency'] / 3e8
            
            # 지연된 시간축
            t_delayed = t - time_delay
            valid_idx = t_delayed >= 0
            
            if np.any(valid_idx):
                # 반사 신호 (RCS에 비례하는 진폭)
                reflection = np.sqrt(rcs) * np.exp(1j * 2 * np.pi * (
                    (self.config['center_frequency'] + doppler_freq) * t_delayed +
                    0.5 * chirp_slope * t_delayed**2
                )) * valid_idx.astype(float)
                
                rx_signal += reflection
        
        # 노이즈 추가
        noise_power = 10**(self.config['noise_level']/10)
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(rx_signal)) + 
                                         1j * np.random.randn(len(rx_signal)))
        
        clean_signal = rx_signal + noise
        
        return clean_signal, target_config
    
    def generate_jamming_signal(self, jamming_params):
        """재밍 신호 생성"""
        # 기본 신호 생성
        clean_signal, _ = self.generate_clean_signal()
        
        # 시간 축
        t = np.linspace(0, self.config['chirp_duration'], self.samples_per_chirp)
        
        jamming_signal = np.zeros_like(clean_signal, dtype=complex)
        
        # 각 재밍 파라미터에 대해 재밍 신호 생성
        for params in jamming_params:
            # 전력 재밍
            power_component = params['power_ratio'] * clean_signal
            
            # 주파수 오프셋 재밍
            freq_component = clean_signal * np.exp(1j * 2 * np.pi * 
                                                 params['freq_offset'] * t)
            
            # 시간 지연 재밍
            delay_samples = int(params['time_delay'] * self.config['sampling_rate'])
            time_component = np.roll(clean_signal, delay_samples)
            
            # 처프율 조작 재밍
            chirp_slope = (self.config['bandwidth'] * params['chirp_slope_ratio'] / 
                          self.config['chirp_duration'])
            slope_component = np.exp(1j * 2 * np.pi * (
                self.config['center_frequency'] * t + 
                0.5 * chirp_slope * t**2
            ))
            
            # 복합 재밍 신호
            jammer = (power_component + freq_component + 
                     time_component + slope_component) / 4
            
            jamming_signal += jammer
        
        return jamming_signal
    
    def range_fft(self, signal):
        """거리 FFT 처리"""
        # 윈도우 적용
        windowed_signal = signal * np.hanning(len(signal))
        
        # FFT
        range_spectrum = fft(windowed_signal, n=2*len(signal))
        
        # 거리 축 계산
        range_axis = np.arange(len(range_spectrum)) * (
            3e8 / (2 * self.config['bandwidth'])
        )
        
        # 1미터 범위만 추출
        valid_range_idx = range_axis <= self.config['max_range']
        
        return range_spectrum[valid_range_idx], range_axis[valid_range_idx]
    
    def calculate_metrics(self, clean_signal, jammed_signal):
        """성능 지표 계산"""
        # SNR 계산
        signal_power = np.mean(np.abs(clean_signal)**2)
        noise_power = np.mean(np.abs(jammed_signal - clean_signal)**2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-12))
        
        # 상관계수
        correlation = np.corrcoef(np.abs(clean_signal), 
                                np.abs(jammed_signal))[0, 1]
        
        # 거리 스펙트럼 비교
        clean_spectrum, range_axis = self.range_fft(clean_signal)
        jammed_spectrum, _ = self.range_fft(jammed_signal)
        
        # 피크 탐지 성능
        clean_peaks = self._detect_peaks(np.abs(clean_spectrum))
        jammed_peaks = self._detect_peaks(np.abs(jammed_spectrum))
        
        return {
            'snr_db': snr_db,
            'correlation': correlation,
            'clean_peaks': clean_peaks,
            'jammed_peaks': jammed_peaks,
            'peak_shift': len(jammed_peaks) - len(clean_peaks),
            'range_axis': range_axis,
            'clean_spectrum': clean_spectrum,
            'jammed_spectrum': jammed_spectrum
        }
    
    def _detect_peaks(self, spectrum, prominence=0.1):
        """피크 탐지"""
        peaks, _ = signal.find_peaks(spectrum, prominence=prominence)
        return peaks

class Stage1ExperimentRunner:
    """1단계 실험 실행기"""
    
    def __init__(self):
        self.simulator = ShortRangeRadarSimulator()
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 결과 저장 디렉토리
        self.results_dir = f"stage1_results_{self.experiment_id}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"🔬 1단계 실험 준비 완료")
        print(f"   실험 ID: {self.experiment_id}")
        print(f"   결과 저장: {self.results_dir}/")
    
    def run_jamming_scenarios(self, num_scenarios=100):
        """다양한 재밍 시나리오 실행"""
        print(f"\n🎯 재밍 시나리오 실험 시작 ({num_scenarios}개)")
        
        scenarios = []
        
        for i in tqdm(range(num_scenarios), desc="재밍 시나리오 생성"):
            # 랜덤 재밍 파라미터 생성
            scenario = {
                'scenario_id': i,
                'jamming_params': [{
                    'power_ratio': np.random.choice(
                        self.simulator.config['power_ratios']),
                    'freq_offset': np.random.choice(
                        self.simulator.config['freq_offsets']),
                    'time_delay': np.random.choice(
                        self.simulator.config['time_delays']),
                    'chirp_slope_ratio': np.random.choice(
                        self.simulator.config['chirp_slope_ratios'])
                }]
            }
            
            # 신호 생성
            clean_signal, target_config = self.simulator.generate_clean_signal()
            jamming_signal = self.simulator.generate_jamming_signal(
                scenario['jamming_params'])
            jammed_signal = clean_signal + jamming_signal
            
            # 성능 분석
            metrics = self.simulator.calculate_metrics(clean_signal, jammed_signal)
            
            # 결과 저장
            result = {
                'scenario': scenario,
                'target_config': target_config,
                'metrics': metrics,
                'signals': {
                    'clean': clean_signal,
                    'jamming': jamming_signal,
                    'jammed': jammed_signal
                }
            }
            
            scenarios.append(result)
        
        self.results = scenarios
        print(f"✅ {len(scenarios)}개 시나리오 완료")
        
        return scenarios
    
    def analyze_results(self):
        """결과 분석"""
        print(f"\n📊 결과 분석 중...")
        
        # 성능 지표 추출
        snr_values = [r['metrics']['snr_db'] for r in self.results]
        correlation_values = [r['metrics']['correlation'] for r in self.results]
        peak_shift_values = [r['metrics']['peak_shift'] for r in self.results]
        
        # 재밥 파라미터 추출
        power_ratios = [r['scenario']['jamming_params'][0]['power_ratio'] 
                       for r in self.results]
        freq_offsets = [r['scenario']['jamming_params'][0]['freq_offset'] 
                       for r in self.results]
        time_delays = [r['scenario']['jamming_params'][0]['time_delay'] 
                      for r in self.results]
        
        analysis = {
            'performance_stats': {
                'snr_mean': np.mean(snr_values),
                'snr_std': np.std(snr_values),
                'correlation_mean': np.mean(correlation_values),
                'correlation_std': np.std(correlation_values),
                'peak_shift_mean': np.mean(peak_shift_values),
                'peak_shift_std': np.std(peak_shift_values)
            },
            'jamming_impact': {
                'power_ratio_effect': np.corrcoef(power_ratios, snr_values)[0,1],
                'freq_offset_effect': np.corrcoef(np.abs(freq_offsets), 
                                                np.abs(peak_shift_values))[0,1],
                'time_delay_effect': np.corrcoef(time_delays, 
                                               np.abs(peak_shift_values))[0,1]
            },
            'critical_scenarios': {
                'worst_snr_idx': np.argmin(snr_values),
                'best_snr_idx': np.argmax(snr_values),
                'most_peaks_shifted': np.argmax(np.abs(peak_shift_values))
            }
        }
        
        print(f"   📈 평균 SNR: {analysis['performance_stats']['snr_mean']:.2f} ±{analysis['performance_stats']['snr_std']:.2f} dB")
        print(f"   🔗 평균 상관계수: {analysis['performance_stats']['correlation_mean']:.3f} ±{analysis['performance_stats']['correlation_std']:.3f}")
        print(f"   🎯 평균 피크 변화: {analysis['performance_stats']['peak_shift_mean']:.1f} ±{analysis['performance_stats']['peak_shift_std']:.1f}")
        
        return analysis
    
    def visualize_results(self, analysis):
        """결과 시각화"""
        print(f"\n🎨 결과 시각화 중...")
        
        # 컬러 팔레트
        colors = sns.color_palette("husl", 8)
        
        # 1. 전체 성능 개요
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'그래프 1: 1단계 실험 결과 종합 분석\n실험 ID: {self.experiment_id}', 
                    fontsize=16, fontweight='bold')
        
        # SNR 분포
        snr_values = [r['metrics']['snr_db'] for r in self.results]
        axes[0,0].hist(snr_values, bins=20, color=colors[0], alpha=0.7, edgecolor='black')
        axes[0,0].axvline(np.mean(snr_values), color='red', linestyle='--', 
                         label=f'평균: {np.mean(snr_values):.1f} dB')
        axes[0,0].set_xlabel('SNR (dB)')
        axes[0,0].set_ylabel('빈도')
        axes[0,0].set_title('1-A: SNR 분포')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 상관계수 분포
        corr_values = [r['metrics']['correlation'] for r in self.results]
        axes[0,1].hist(corr_values, bins=20, color=colors[1], alpha=0.7, edgecolor='black')
        axes[0,1].axvline(np.mean(corr_values), color='red', linestyle='--',
                         label=f'평균: {np.mean(corr_values):.3f}')
        axes[0,1].set_xlabel('상관계수')
        axes[0,1].set_ylabel('빈도')
        axes[0,1].set_title('1-B: 신호 상관계수 분포')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 피크 변화 분포
        peak_shifts = [r['metrics']['peak_shift'] for r in self.results]
        axes[0,2].hist(peak_shifts, bins=20, color=colors[2], alpha=0.7, edgecolor='black')
        axes[0,2].axvline(np.mean(peak_shifts), color='red', linestyle='--',
                         label=f'평균: {np.mean(peak_shifts):.1f}')
        axes[0,2].set_xlabel('피크 개수 변화')
        axes[0,2].set_ylabel('빈도')
        axes[0,2].set_title('1-C: 피크 탐지 성능')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 재밍 파라미터 vs 성능
        power_ratios = [r['scenario']['jamming_params'][0]['power_ratio'] 
                       for r in self.results]
        axes[1,0].scatter(power_ratios, snr_values, color=colors[3], alpha=0.6)
        axes[1,0].set_xlabel('전력비')
        axes[1,0].set_ylabel('SNR (dB)')
        axes[1,0].set_title('1-D: 전력 재밍 효과')
        axes[1,0].grid(True, alpha=0.3)
        
        freq_offsets = [abs(r['scenario']['jamming_params'][0]['freq_offset'])/1e6 
                       for r in self.results]
        axes[1,1].scatter(freq_offsets, [abs(p) for p in peak_shifts], 
                         color=colors[4], alpha=0.6)
        axes[1,1].set_xlabel('주파수 오프셋 (MHz)')
        axes[1,1].set_ylabel('|피크 변화|')
        axes[1,1].set_title('1-E: 주파수 재밍 효과')
        axes[1,1].grid(True, alpha=0.3)
        
        time_delays = [r['scenario']['jamming_params'][0]['time_delay']*1e9 
                      for r in self.results]
        axes[1,2].scatter(time_delays, [abs(p) for p in peak_shifts], 
                         color=colors[5], alpha=0.6)
        axes[1,2].set_xlabel('시간 지연 (ns)')
        axes[1,2].set_ylabel('|피크 변화|')
        axes[1,2].set_title('1-F: 시간 지연 재밍 효과')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/graph1_stage1_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 대표적인 재밍 시나리오 상세 분석
        self._plot_critical_scenarios(analysis)
        
        # 3. 거리 스펙트럼 비교
        self._plot_range_spectra()
        
        print(f"✅ 시각화 완료 - 결과 저장: {self.results_dir}/")
        print(f"   📊 그래프 1: 전체 성능 개요 (6개 서브그래프)")
        print(f"   📊 그래프 2: 주요 재밍 시나리오 분석 (6개 서브그래프)")  
        print(f"   📊 그래프 3: 거리 분해능 분석 (4개 서브그래프)")
    
    def _plot_critical_scenarios(self, analysis):
        """주요 시나리오 상세 분석"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('그래프 2: 주요 재밍 시나리오 상세 분석', fontsize=16, fontweight='bold')
        
        scenarios_to_plot = [
            ('최악 SNR', analysis['critical_scenarios']['worst_snr_idx']),
            ('최고 SNR', analysis['critical_scenarios']['best_snr_idx']),
            ('최대 피크 변화', analysis['critical_scenarios']['most_peaks_shifted'])
        ]
        
        for i, (title, idx) in enumerate(scenarios_to_plot):
            result = self.results[idx]
            
            # 시간 도메인 신호
            t = np.linspace(0, self.simulator.config['chirp_duration'], 
                          len(result['signals']['clean'])) * 1000  # ms
            
            axes[i,0].plot(t, np.abs(result['signals']['clean']), 
                          label='깨끗한 신호', color='blue', linewidth=1.5)
            axes[i,0].plot(t, np.abs(result['signals']['jammed']), 
                          label='재밍된 신호', color='red', linewidth=1.5, alpha=0.8)
            axes[i,0].set_xlabel('시간 (ms)')
            axes[i,0].set_ylabel('신호 크기')
            axes[i,0].set_title(f'2-{chr(65+i*2)}: {title} - 시간 도메인')
            axes[i,0].legend()
            axes[i,0].grid(True, alpha=0.3)
            
            # 거리 스펙트럼
            range_axis = result['metrics']['range_axis']
            clean_spectrum = np.abs(result['metrics']['clean_spectrum'])
            jammed_spectrum = np.abs(result['metrics']['jammed_spectrum'])
            
            axes[i,1].plot(range_axis*100, clean_spectrum, 
                          label='깨끗한 신호', color='blue', linewidth=2)
            axes[i,1].plot(range_axis*100, jammed_spectrum, 
                          label='재밍된 신호', color='red', linewidth=2, alpha=0.8)
            axes[i,1].set_xlabel('거리 (cm)')
            axes[i,1].set_ylabel('신호 크기')
            axes[i,1].set_title(f'2-{chr(66+i*2)}: {title} - 거리 스펙트럼')
            axes[i,1].legend()
            axes[i,1].grid(True, alpha=0.3)
            axes[i,1].set_xlim([0, 100])  # 1미터 범위
            
            # 성능 지표 텍스트 추가
            metrics = result['metrics']
            info_text = f"SNR: {metrics['snr_db']:.1f} dB\n상관계수: {metrics['correlation']:.3f}\n피크 변화: {metrics['peak_shift']}"
            axes[i,1].text(0.02, 0.98, info_text, transform=axes[i,1].transAxes,
                          verticalalignment='top', bbox=dict(boxstyle='round', 
                          facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/graph2_critical_scenarios.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_range_spectra(self):
        """거리 스펙트럼 비교 분석"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('그래프 3: 1미터 범위 내 거리 분해능 분석', fontsize=16, fontweight='bold')
        
        # 다양한 재밍 강도별 거리 스펙트럼
        power_levels = [0.2, 1.0, 2.0, 5.0]
        
        for i, power_ratio in enumerate(power_levels):
            row, col = i // 2, i % 2
            
            # 해당 전력비의 결과 찾기
            matching_results = [r for r in self.results 
                              if abs(r['scenario']['jamming_params'][0]['power_ratio'] - power_ratio) < 0.1]
            
            if matching_results:
                result = matching_results[0]
                range_axis = result['metrics']['range_axis'] * 100  # cm
                clean_spectrum = np.abs(result['metrics']['clean_spectrum'])
                jammed_spectrum = np.abs(result['metrics']['jammed_spectrum'])
                
                axes[row,col].plot(range_axis, clean_spectrum, 
                                 label='깨끗한 신호', color='blue', linewidth=2)
                axes[row,col].plot(range_axis, jammed_spectrum, 
                                 label=f'재밍 (전력비 {power_ratio})', 
                                 color='red', linewidth=2, alpha=0.8)
                
                # 타겟 위치 표시
                target_positions = [d*100 for d in self.simulator.config['target_distances']]
                for pos in target_positions:
                    axes[row,col].axvline(pos, color='green', linestyle='--', 
                                        alpha=0.7, label='실제 타겟' if pos == target_positions[0] else '')
                
                axes[row,col].set_xlabel('거리 (cm)')
                axes[row,col].set_ylabel('신호 크기')
                axes[row,col].set_title(f'3-{chr(65+i)}: 전력비 {power_ratio} 재밍')
                axes[row,col].legend()
                axes[row,col].grid(True, alpha=0.3)
                axes[row,col].set_xlim([0, 100])
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/graph3_range_spectra_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, analysis):
        """결과 저장"""
        print(f"\n💾 결과 저장 중...")
        
        # HDF5 형식으로 신호 데이터 저장
        with h5py.File(f'{self.results_dir}/stage1_signals.h5', 'w') as f:
            for i, result in enumerate(self.results):
                grp = f.create_group(f'scenario_{i:03d}')
                
                # 신호 저장
                grp.create_dataset('clean_signal', data=result['signals']['clean'])
                grp.create_dataset('jamming_signal', data=result['signals']['jamming'])
                grp.create_dataset('jammed_signal', data=result['signals']['jammed'])
                
                # 파라미터 저장
                grp.attrs['power_ratio'] = result['scenario']['jamming_params'][0]['power_ratio']
                grp.attrs['freq_offset'] = result['scenario']['jamming_params'][0]['freq_offset']
                grp.attrs['time_delay'] = result['scenario']['jamming_params'][0]['time_delay']
                grp.attrs['chirp_slope_ratio'] = result['scenario']['jamming_params'][0]['chirp_slope_ratio']
                
                # 성능 지표 저장
                grp.attrs['snr_db'] = result['metrics']['snr_db']
                grp.attrs['correlation'] = result['metrics']['correlation']
                grp.attrs['peak_shift'] = result['metrics']['peak_shift']
        
        # JSON 형식으로 분석 결과 저장
        analysis_json = {
            'experiment_info': {
                'experiment_id': self.experiment_id,
                'num_scenarios': len(self.results),
                'radar_config': self.simulator.config,
                'timestamp': datetime.now().isoformat()
            },
            'performance_analysis': analysis
        }
        
        with open(f'{self.results_dir}/stage1_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_json, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ 결과 저장 완료:")
        print(f"   - 신호 데이터: {self.results_dir}/stage1_signals.h5")
        print(f"   - 분석 결과: {self.results_dir}/stage1_analysis.json")
        print(f"   - 시각화: {self.results_dir}/*.png")
    
    def generate_stage2_baseline(self, analysis):
        """2단계 하드웨어 실험을 위한 기준점 생성"""
        print(f"\n🎯 2단계 실험 기준점 생성 중...")
        
        baseline = {
            'stage1_summary': {
                'total_scenarios': len(self.results),
                'avg_snr': analysis['performance_stats']['snr_mean'],
                'avg_correlation': analysis['performance_stats']['correlation_mean'],
                'detection_success_rate': sum(1 for r in self.results 
                                            if abs(r['metrics']['peak_shift']) <= 1) / len(self.results)
            },
            'hardware_test_targets': {
                'target_distances_cm': [d*100 for d in self.simulator.config['target_distances']],
                'expected_snr_range': [analysis['performance_stats']['snr_mean'] - 5,
                                     analysis['performance_stats']['snr_mean'] + 5],
                'correlation_threshold': analysis['performance_stats']['correlation_mean'] - 0.1
            },
            'recommended_jamming_tests': [
                {'power_ratio': 1.0, 'freq_offset': 5e6, 'time_delay': 3.33e-9, 'description': '중간 강도 복합 재밍'},
                {'power_ratio': 2.0, 'freq_offset': 0, 'time_delay': 0, 'description': '강력한 전력 재밍'},
                {'power_ratio': 0.5, 'freq_offset': 10e6, 'time_delay': 6.67e-9, 'description': '주파수+지연 재밍'}
            ],
            'performance_thresholds': {
                'minimum_snr': 10.0,  # dB
                'minimum_correlation': 0.7,
                'maximum_false_peaks': 2
            }
        }
        
        with open(f'{self.results_dir}/stage2_baseline.json', 'w', encoding='utf-8') as f:
            json.dump(baseline, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 2단계 기준점 생성 완료:")
        print(f"   - 타겟 거리: {baseline['hardware_test_targets']['target_distances_cm']} cm")
        print(f"   - 예상 SNR: {baseline['hardware_test_targets']['expected_snr_range']} dB")
        print(f"   - 탐지 성공률: {baseline['stage1_summary']['detection_success_rate']:.1%}")
        
        return baseline

def main():
    """1단계 실험 메인 실행"""
    print("🚀 X4M06 레이더 1단계 실험 시작")
    print("=" * 60)
    
    # 실험 실행기 초기화
    runner = Stage1ExperimentRunner()
    
    try:
        # 1. 재밍 시나리오 실행
        scenarios = runner.run_jamming_scenarios(num_scenarios=200)
        
        # 2. 결과 분석
        analysis = runner.analyze_results()
        
        # 3. 시각화
        runner.visualize_results(analysis)
        
        # 4. 결과 저장
        runner.save_results(analysis)
        
        # 5. 2단계 기준점 생성
        baseline = runner.generate_stage2_baseline(analysis)
        
        print(f"\n🎉 1단계 실험 완료!")
        print(f"📊 총 {len(scenarios)}개 시나리오 분석 완료")
        print(f"📁 결과 저장 위치: {runner.results_dir}/")
        print(f"🎯 2단계 하드웨어 실험 준비 완료")
        
        return runner, analysis, baseline
        
    except Exception as e:
        print(f"❌ 실험 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    runner, analysis, baseline = main()