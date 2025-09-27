#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X4M06 5미터 환경 CW 재밍 시뮬레이터
Created: 2024-09-27
Purpose: 5미터 이내 실내 환경에서 CW 재밍을 적용한 UWB 임펄스 레이더 시뮬레이션
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import font_manager
from datetime import datetime
import json
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class X4M06CWJammingSimulator:
    """X4M06 5미터 환경 CW 재밍 시뮬레이터"""
    
    def __init__(self):
        # X4M06 5미터 환경 최적화 파라미터
        self.config = {
            # 레이더 기본 파라미터
            'center_freq': 8.748e9,        # 8.748 GHz
            'bandwidth': 1.4e9,            # 1.4 GHz
            'prf': 15000,                  # 15 kHz (5m 환경 최적화)
            'pulse_width': 2e-9,           # 2 ns
            'sampling_rate': 23.328e9,     # 23.328 GHz
            'max_range': 5.0,              # 5 미터
            'min_range': 0.18,             # 0.18 미터
            
            # CW 재밍 파라미터
            'jamming_frequencies': [8.448e9, 8.748e9, 9.048e9],  # GHz
            'jamming_powers': [0.5, 1.0, 2.0, 5.0, 10.0],       # 신호 대비 배수
            'jamming_scenarios': {
                'mild': {'power_ratio': 0.5, 'frequencies': [8.448e9]},
                'moderate': {'power_ratio': 2.0, 'frequencies': [8.748e9]},
                'severe': {'power_ratio': 10.0, 'frequencies': [8.448e9, 8.748e9, 9.048e9]}
            },
            
            # 환경 파라미터
            'num_targets': (1, 4),         # 타겟 개수 범위
            'target_ranges': (0.5, 4.8),  # 타겟 거리 범위 (m)
            'noise_floor': -90,            # dBm
            'snr_range': (5, 25),          # SNR 범위 (dB)
        }
        
        # 물리 상수
        self.c = 3e8  # 빛의 속도
        
        # 계산된 파라미터
        self.range_resolution = self.c / (2 * self.config['bandwidth'])
        self.max_delay = 2 * self.config['max_range'] / self.c
        self.samples_per_frame = int(self.config['sampling_rate'] * self.max_delay)
        self.time_vector = np.linspace(0, self.max_delay, self.samples_per_frame)
        self.range_vector = self.time_vector * self.c / 2
        
        print("🎯 X4M06 5미터 환경 CW 재밍 시뮬레이터 초기화")
        print(f"   - 거리 해상도: {self.range_resolution*100:.1f} cm")
        print(f"   - 프레임당 샘플: {self.samples_per_frame}")
        print(f"   - 시뮬레이션 범위: {self.config['min_range']:.1f} - {self.config['max_range']:.1f} m")
    
    def generate_uwb_impulse(self, target_range, target_rcs, noise_level):
        """UWB 임펄스 레이더 신호 생성"""
        
        # 타겟까지의 지연시간
        delay_time = 2 * target_range / self.c
        delay_samples = int(delay_time * self.config['sampling_rate'])
        
        # 가우시안 임펄스 생성 (UWB 특성)
        pulse_samples = int(self.config['pulse_width'] * self.config['sampling_rate'])
        t_pulse = np.linspace(-self.config['pulse_width']/2, self.config['pulse_width']/2, pulse_samples)
        
        # UWB 가우시안 임펄스 (미분된 가우시안)
        sigma = self.config['pulse_width'] / 6  # 펄스 폭의 1/6
        gaussian_pulse = np.exp(-0.5 * (t_pulse / sigma)**2)
        uwb_pulse = -2 * (t_pulse / sigma**2) * gaussian_pulse  # 1차 미분
        
        # 중심 주파수로 변조
        carrier = np.cos(2 * np.pi * self.config['center_freq'] * t_pulse)
        modulated_pulse = uwb_pulse * carrier
        
        # RCS 및 거리에 따른 감쇠 적용
        path_loss = (4 * np.pi * target_range * self.config['center_freq'] / self.c)**2
        amplitude = np.sqrt(target_rcs) / path_loss
        
        # 전체 신호 배열에 펄스 배치
        signal = np.zeros(self.samples_per_frame, dtype=complex)
        
        if delay_samples + len(modulated_pulse) < self.samples_per_frame:
            # I/Q 채널로 복소수 신호 생성 (힐버트 변환 대신 간단한 위상 시프트 사용)
            from scipy.signal import hilbert
            quadrature_pulse = np.imag(hilbert(modulated_pulse))
            signal[delay_samples:delay_samples + len(modulated_pulse)] = \
                amplitude * (modulated_pulse + 1j * quadrature_pulse)
        
        # 노이즈 추가
        noise_power = 10**(noise_level/10) / 1000  # dBm to W
        noise = np.sqrt(noise_power/2) * (np.random.normal(size=self.samples_per_frame) + 
                                         1j * np.random.normal(size=self.samples_per_frame))
        
        return signal + noise
    
    def generate_cw_jamming(self, scenario='moderate', custom_params=None):
        """CW 재밍 신호 생성"""
        
        if custom_params:
            jam_params = custom_params
        else:
            jam_params = self.config['jamming_scenarios'][scenario]
        
        jamming_signal = np.zeros(self.samples_per_frame, dtype=complex)
        
        # 각 재밍 주파수에 대해 CW 신호 생성
        for jam_freq in jam_params['frequencies']:
            # CW 신호 (연속파)
            cw_signal = jam_params['power_ratio'] * np.exp(1j * 2 * np.pi * jam_freq * self.time_vector)
            
            # 위상 노이즈 추가 (실제 발진기 특성)
            phase_noise = np.random.normal(0, 0.1, self.samples_per_frame)  # 라디안
            cw_signal *= np.exp(1j * phase_noise)
            
            # 진폭 변동 추가 (실제 CW 신호 특성)
            amplitude_variation = 1 + 0.05 * np.random.normal(size=self.samples_per_frame)
            cw_signal *= amplitude_variation
            
            jamming_signal += cw_signal
        
        # 재밍 신호의 시간 변동성 (실제 환경 모사)
        modulation_freq = np.random.uniform(10, 100)  # 10-100 Hz 변조
        time_modulation = 1 + 0.1 * np.sin(2 * np.pi * modulation_freq * self.time_vector)
        jamming_signal *= time_modulation
        
        return jamming_signal
    
    def generate_clean_signal(self):
        """재밍 없는 깨끗한 신호 생성"""
        
        # 랜덤 타겟 생성
        num_targets = np.random.randint(*self.config['num_targets'])
        targets = []
        
        total_signal = np.zeros(self.samples_per_frame, dtype=complex)
        
        for _ in range(num_targets):
            # 타겟 파라미터
            target_range = np.random.uniform(*self.config['target_ranges'])
            target_rcs = np.random.uniform(0.1, 10.0)  # m²
            
            # SNR 기반 노이즈 레벨 계산
            snr_db = np.random.uniform(*self.config['snr_range'])
            noise_level = self.config['noise_floor'] - snr_db
            
            # 타겟 정보 저장
            targets.append({
                'range': target_range,
                'rcs': target_rcs,
                'snr_db': snr_db
            })
            
            # 신호 생성 및 합성
            target_signal = self.generate_uwb_impulse(target_range, target_rcs, noise_level)
            total_signal += target_signal
        
        return total_signal, targets
    
    def generate_jammed_signal(self, jamming_scenario='moderate'):
        """CW 재밍이 적용된 신호 생성"""
        
        # 베이스 클린 신호 생성
        clean_signal, targets = self.generate_clean_signal()
        
        # CW 재밍 신호 생성
        jamming_signal = self.generate_cw_jamming(jamming_scenario)
        
        # 신호 합성
        jammed_signal = clean_signal + jamming_signal
        
        # 재밍 파라미터 정보
        jam_info = {
            'scenario': jamming_scenario,
            'parameters': self.config['jamming_scenarios'][jamming_scenario],
            'jamming_power_ratio': self.config['jamming_scenarios'][jamming_scenario]['power_ratio'],
            'frequencies': self.config['jamming_scenarios'][jamming_scenario]['frequencies']
        }
        
        return jammed_signal, targets, jam_info
    
    def calculate_spectral_features(self, signal):
        """스펙트럼 특성 계산"""
        
        # STFT 파라미터 (5미터 환경 최적화)
        nperseg = 128
        noverlap = 64
        nfft = 256
        
        # STFT 계산
        from scipy.signal import stft
        frequencies, times, Zxx = stft(signal, 
                                      fs=self.config['sampling_rate'], 
                                      nperseg=nperseg, 
                                      noverlap=noverlap, 
                                      nfft=nfft)
        
        # 스펙트로그램 크기
        spectrogram = np.abs(Zxx)**2
        
        # 주요 특성 추출
        features = {
            'mean_power': np.mean(spectrogram),
            'max_power': np.max(spectrogram),
            'peak_freq': frequencies[np.argmax(np.sum(spectrogram, axis=1))],
            'bandwidth_3db': self._calculate_3db_bandwidth(frequencies, spectrogram),
            'spectral_centroid': np.sum(frequencies[:, np.newaxis] * spectrogram) / np.sum(spectrogram)
        }
        
        return spectrogram, frequencies, times, features
    
    def _calculate_3db_bandwidth(self, frequencies, spectrogram):
        """3dB 대역폭 계산"""
        
        power_spectrum = np.mean(spectrogram, axis=1)
        max_power = np.max(power_spectrum)
        half_power = max_power / 2
        
        # 3dB 포인트 찾기
        indices = np.where(power_spectrum >= half_power)[0]
        if len(indices) > 0:
            bandwidth = frequencies[indices[-1]] - frequencies[indices[0]]
        else:
            bandwidth = 0
        
        return bandwidth
    
    def visualize_signals(self, clean_signal, jammed_signal, targets, jam_info):
        """신호 시각화"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('X4M06 5미터 환경 CW 재밍 시뮬레이션', fontsize=16, fontweight='bold')
        
        # 1. 시간 도메인 비교
        time_ms = self.time_vector * 1e9  # ns로 변환
        
        ax1.plot(time_ms, np.abs(clean_signal), 'g-', label='Clean Signal', alpha=0.7)
        ax1.plot(time_ms, np.abs(jammed_signal), 'r-', label='Jammed Signal', alpha=0.7)
        ax1.set_xlabel('시간 (ns)')
        ax1.set_ylabel('신호 크기')
        ax1.set_title('시간 도메인: Clean vs Jammed')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 거리 도메인
        ax2.plot(self.range_vector, np.abs(clean_signal), 'g-', label='Clean Signal', alpha=0.7)
        ax2.plot(self.range_vector, np.abs(jammed_signal), 'r-', label='Jammed Signal', alpha=0.7)
        
        # 타겟 위치 표시
        for i, target in enumerate(targets):
            ax2.axvline(target['range'], color='blue', linestyle='--', alpha=0.5,
                       label=f"Target {i+1}" if i == 0 else '')
        
        ax2.set_xlabel('거리 (m)')
        ax2.set_ylabel('신호 크기')
        ax2.set_title('거리 도메인: 타겟 탐지')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 5)
        
        # 3. 주파수 스펙트럼
        freq_clean = np.fft.fftfreq(len(clean_signal), 1/self.config['sampling_rate'])
        fft_clean = np.abs(np.fft.fft(clean_signal))
        fft_jammed = np.abs(np.fft.fft(jammed_signal))
        
        # 관심 주파수 대역만 표시
        freq_mask = (freq_clean >= 8e9) & (freq_clean <= 10e9)
        
        ax3.plot(freq_clean[freq_mask]/1e9, fft_clean[freq_mask], 'g-', label='Clean', alpha=0.7)
        ax3.plot(freq_clean[freq_mask]/1e9, fft_jammed[freq_mask], 'r-', label='Jammed', alpha=0.7)
        
        # 재밍 주파수 표시
        for jf in jam_info['frequencies']:
            ax3.axvline(jf/1e9, color='red', linestyle=':', alpha=0.8, label='CW Jamming')
        
        ax3.set_xlabel('주파수 (GHz)')
        ax3.set_ylabel('크기')
        ax3.set_title(f'주파수 스펙트럼 - {jam_info["scenario"].upper()} 재밍')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 재밍 효과 분석
        scenarios = ['mild', 'moderate', 'severe']
        power_ratios = [self.config['jamming_scenarios'][s]['power_ratio'] for s in scenarios]
        freq_counts = [len(self.config['jamming_scenarios'][s]['frequencies']) for s in scenarios]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        ax4.bar(x - width/2, power_ratios, width, label='전력 비율', alpha=0.7, color='red')
        ax4.bar(x + width/2, freq_counts, width, label='주파수 개수', alpha=0.7, color='blue')
        
        ax4.set_xlabel('재밍 시나리오')
        ax4.set_ylabel('값')
        ax4.set_title('CW 재밍 시나리오별 파라미터')
        ax4.set_xticks(x)
        ax4.set_xticklabels([s.upper() for s in scenarios])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 현재 시나리오 강조
        current_idx = scenarios.index(jam_info['scenario'])
        ax4.axvline(current_idx, color='orange', linestyle='--', alpha=0.8, linewidth=3)
        
        plt.tight_layout()
        return fig
    
    def generate_dataset(self, num_samples=1000, output_file=None):
        """CW 재밍 데이터셋 생성"""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"x4m06_5m_cw_jamming_dataset_{timestamp}.h5"
        
        print(f"\n🚀 X4M06 5미터 환경 CW 재밍 데이터셋 생성 시작")
        print(f"   - 샘플 수: {num_samples}")
        print(f"   - 출력 파일: {output_file}")
        
        # HDF5 파일 생성
        with h5py.File(output_file, 'w') as f:
            # 메타데이터
            meta_group = f.create_group('metadata')
            meta_group.attrs['creation_date'] = datetime.now().isoformat()
            meta_group.attrs['simulator_version'] = '1.0_5m_cw'
            meta_group.attrs['description'] = 'X4M06 5m CW Jamming Dataset'
            
            # 설정 정보 저장
            config_group = meta_group.create_group('config')
            for key, value in self.config.items():
                if isinstance(value, dict):
                    subgroup = config_group.create_group(key)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, list):
                            subgroup.create_dataset(subkey, data=np.array(subvalue))
                        elif isinstance(subvalue, (int, float, str)):
                            subgroup.attrs[subkey] = subvalue
                        else:
                            # 복잡한 객체는 문자열로 변환
                            subgroup.attrs[subkey] = str(subvalue)
                elif isinstance(value, (list, tuple)):
                    config_group.create_dataset(key, data=np.array(value))
                elif isinstance(value, (int, float, str)):
                    config_group.attrs[key] = value
                else:
                    config_group.attrs[key] = str(value)
            
            # 데이터셋 그룹
            data_group = f.create_group('radar_data')
            
            # 데이터 배열 초기화
            clean_signals = data_group.create_dataset('clean_signals', 
                                                    shape=(num_samples, self.samples_per_frame), 
                                                    dtype=np.complex64)
            jammed_signals = data_group.create_dataset('jammed_signals', 
                                                     shape=(num_samples, self.samples_per_frame), 
                                                     dtype=np.complex64)
            
            # 타겟 정보
            target_group = f.create_group('targets')
            jamming_group = f.create_group('jamming_info')
            
            # 시나리오 분포
            scenarios = list(self.config['jamming_scenarios'].keys())
            scenario_counts = {s: 0 for s in scenarios}
            
            # 데이터 생성 루프
            for i in range(num_samples):
                if i % 100 == 0:
                    print(f"   진행률: {i}/{num_samples} ({100*i/num_samples:.1f}%)")
                
                # 시나리오 선택 (균등 분포)
                scenario = scenarios[i % len(scenarios)]
                scenario_counts[scenario] += 1
                
                # 신호 생성
                clean_signal, targets = self.generate_clean_signal()
                jammed_signal, targets, jam_info = self.generate_jammed_signal(scenario)
                
                # 데이터 저장
                clean_signals[i] = clean_signal
                jammed_signals[i] = jammed_signal
                
                # 타겟 정보 저장
                target_subgroup = target_group.create_group(f'sample_{i:06d}')
                for j, target in enumerate(targets):
                    target_info = target_subgroup.create_group(f'target_{j}')
                    target_info.attrs['range'] = target['range']
                    target_info.attrs['rcs'] = target['rcs']
                    target_info.attrs['snr_db'] = target['snr_db']
                
                # 재밍 정보 저장
                jam_subgroup = jamming_group.create_group(f'sample_{i:06d}')
                jam_subgroup.attrs['scenario'] = jam_info['scenario']
                jam_subgroup.attrs['power_ratio'] = jam_info['jamming_power_ratio']
                jam_subgroup.create_dataset('frequencies', data=jam_info['frequencies'])
            
            # 통계 정보
            stats_group = f.create_group('statistics')
            for scenario, count in scenario_counts.items():
                stats_group.attrs[f'{scenario}_count'] = count
            
            print(f"\n✅ 데이터셋 생성 완료!")
            print(f"   📊 시나리오별 분포: {scenario_counts}")
            print(f"   💾 파일 크기: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        return output_file, scenario_counts

def main():
    """메인 실행 함수"""
    
    # 시뮬레이터 초기화
    simulator = X4M06CWJammingSimulator()
    
    # 샘플 신호 생성 및 시각화
    print("\n🔍 샘플 신호 생성 및 시각화...")
    clean_signal, targets = simulator.generate_clean_signal()
    jammed_signal, targets, jam_info = simulator.generate_jammed_signal('severe')
    
    # 시각화
    fig = simulator.visualize_signals(clean_signal, jammed_signal, targets, jam_info)
    plt.savefig('x4m06_5m_cw_jamming_sample.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 전체 데이터셋 생성
    dataset_file, stats = simulator.generate_dataset(1000)
    
    print(f"\n🎯 X4M06 5미터 환경 CW 재밍 시뮬레이션 완료!")
    print(f"   📁 데이터셋: {dataset_file}")
    print(f"   📊 샘플 시각화: x4m06_5m_cw_jamming_sample.png")
    
    return dataset_file, stats

if __name__ == "__main__":
    dataset_file, stats = main()