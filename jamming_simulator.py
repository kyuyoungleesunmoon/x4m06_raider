#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
X4M06 레이더 재밍 환경 시뮬레이터
차량 레이더 간 상호 간섭(재밍) 시뮬레이션 및 합성 데이터셋 생성

연구목적: 자율주행 레이더의 재밍 신호 복원을 위한 Deep Learning 모델 학습용 데이터셋 구축
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import signal
from scipy.fft import fft, fftfreq
import pickle
import os
import json
from datetime import datetime
from tqdm import tqdm
import h5py

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정"""
    try:
        # Windows에서 주로 사용되는 한글 폰트들 시도
        korean_fonts = ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'Dotum']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font in korean_fonts:
            if font in available_fonts:
                plt.rcParams['font.family'] = font
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                break
        else:
            print("한글 폰트를 찾을 수 없어 기본 설정 사용")
        
        # 마이너스 기호 문제 해결
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        
        print("한글 폰트 설정 완료")
        return True
        
    except Exception as e:
        print(f"폰트 설정 중 오류: {e}")
        plt.rcParams['axes.unicode_minus'] = False
        return False

# 폰트 설정 적용
setup_korean_font()


class FMCWRadarSimulator:
    """FMCW 레이더 신호 시뮬레이터"""
    
    def __init__(self, config=None):
        """
        초기화
        Args:
            config (dict): 레이더 파라미터 설정
        """
        if config is None:
            config = self.get_default_config()
        
        self.config = config
        self._validate_config()
        self._initialize_parameters()
    
    def get_default_config(self):
        """기본 레이더 파라미터 설정 (X4M06 기반)"""
        return {
            # 레이더 기본 파라미터
            'center_freq': 8.748e9,        # 중심 주파수 (Hz) - X4M06 기본값
            'bandwidth': 1.4e9,            # 대역폭 (Hz)
            'chirp_duration': 1e-3,        # 처프 지속시간 (s)
            'prf': 1000,                   # 펄스 반복 주파수 (Hz)
            'sampling_rate': 1e6,          # 샘플링 주파수 (Hz)
            
            # 목표물 파라미터
            'target_range': [5, 50],       # 목표물 거리 범위 (m)
            'target_velocity': [-30, 30],  # 목표물 속도 범위 (m/s)
            'target_rcs': [0.1, 10],       # 레이더 반사 단면적 범위 (m²)
            
            # 재밍 파라미터
            'num_jammers': [1, 8],         # 재머 개수 범위
            'jammer_power_ratio': [0.5, 3.0], # 재머 신호 강도 비율
            'freq_offset_range': [-0.1e9, 0.1e9], # 주파수 오프셋 범위 (Hz)
            'time_offset_range': [0, 0.8e-3],     # 시간 오프셋 범위 (s)
            
            # 노이즈 파라미터
            'snr_db': [10, 30],            # 신호 대 잡음비 범위 (dB)
        }
    
    def _validate_config(self):
        """설정 파라미터 유효성 검사"""
        required_keys = [
            'center_freq', 'bandwidth', 'chirp_duration', 'prf', 
            'sampling_rate', 'target_range', 'num_jammers'
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Required parameter '{key}' missing from config")
    
    def _initialize_parameters(self):
        """내부 파라미터 초기화"""
        self.c = 3e8  # 광속 (m/s)
        
        # 처프 기울기 (Hz/s)
        self.chirp_slope = self.config['bandwidth'] / self.config['chirp_duration']
        
        # 시간 벡터
        self.num_samples = int(self.config['chirp_duration'] * self.config['sampling_rate'])
        self.time_vector = np.linspace(0, self.config['chirp_duration'], self.num_samples)
        
        # 주파수 벡터
        self.freq_vector = fftfreq(self.num_samples, 1/self.config['sampling_rate'])
    
    def generate_clean_signal(self, num_targets=1, target_params=None):
        """
        재밍이 없는 깨끗한 목표 신호 생성
        
        Args:
            num_targets (int): 목표물 개수
            target_params (list): 목표물 파라미터 [(range, velocity, rcs), ...]
        
        Returns:
            tuple: (시간 도메인 신호, 목표물 파라미터)
        """
        if target_params is None:
            target_params = self._generate_random_targets(num_targets)
        
        signal_clean = np.zeros(self.num_samples, dtype=complex)
        
        for target_range, target_velocity, target_rcs in target_params:
            # 지연 시간 계산
            time_delay = 2 * target_range / self.c
            
            # 도플러 주파수 계산
            doppler_freq = 2 * target_velocity * self.config['center_freq'] / self.c
            
            # 목표물 반사 신호 생성
            target_signal = self._generate_target_echo(
                time_delay, doppler_freq, target_rcs
            )
            
            signal_clean += target_signal
        
        # 노이즈 추가
        snr_db = np.random.uniform(*self.config['snr_db'])
        signal_clean = self._add_noise(signal_clean, snr_db)
        
        return signal_clean, target_params
    
    def generate_jammed_signal(self, clean_signal, target_params=None):
        """
        재밍 신호가 추가된 오염된 신호 생성
        
        Args:
            clean_signal (np.ndarray): 깨끗한 신호
            target_params (list): 목표물 파라미터
        
        Returns:
            tuple: (재밍된 신호, 재밍 파라미터)
        """
        jammed_signal = clean_signal.copy()
        
        # 재머 개수 결정
        num_jammers = np.random.randint(*self.config['num_jammers'])
        jammer_params = []
        
        for _ in range(num_jammers):
            # 재머 파라미터 생성
            jammer_param = self._generate_jammer_parameters()
            jammer_params.append(jammer_param)
            
            # 재밍 신호 생성
            jamming_signal = self._generate_jamming_signal(jammer_param)
            jammed_signal += jamming_signal
        
        return jammed_signal, jammer_params
    
    def _generate_random_targets(self, num_targets):
        """무작위 목표물 파라미터 생성"""
        targets = []
        for _ in range(num_targets):
            target_range = np.random.uniform(*self.config['target_range'])
            target_velocity = np.random.uniform(*self.config['target_velocity'])
            target_rcs = np.random.uniform(*self.config['target_rcs'])
            targets.append((target_range, target_velocity, target_rcs))
        return targets
    
    def _generate_target_echo(self, time_delay, doppler_freq, rcs):
        """목표물 에코 신호 생성"""
        # 진폭 계산 (레이더 방정식 기반)
        amplitude = np.sqrt(rcs) / (4 * np.pi * (time_delay * self.c / 2) ** 2)
        
        # 지연된 처프 신호
        delayed_time = self.time_vector - time_delay
        valid_mask = delayed_time >= 0
        
        echo_signal = np.zeros(self.num_samples, dtype=complex)
        
        if np.any(valid_mask):
            # 처프 신호 생성
            instantaneous_freq = (
                self.config['center_freq'] + 
                self.chirp_slope * delayed_time[valid_mask] +
                doppler_freq
            )
            
            phase = 2 * np.pi * np.cumsum(instantaneous_freq) / self.config['sampling_rate']
            echo_signal[valid_mask] = amplitude * np.exp(1j * phase)
        
        return echo_signal
    
    def _generate_jammer_parameters(self):
        """재머 파라미터 생성"""
        return {
            'power_ratio': np.random.uniform(*self.config['jammer_power_ratio']),
            'freq_offset': np.random.uniform(*self.config['freq_offset_range']),
            'time_offset': np.random.uniform(*self.config['time_offset_range']),
            'chirp_slope_ratio': np.random.uniform(0.8, 1.2),  # 처프 기울기 변화
        }
    
    def _generate_jamming_signal(self, jammer_params):
        """재밍 신호 생성"""
        # 시간 오프셋 적용
        offset_samples = int(jammer_params['time_offset'] * self.config['sampling_rate'])
        shifted_time = self.time_vector - jammer_params['time_offset']
        
        # 수정된 처프 기울기
        modified_slope = self.chirp_slope * jammer_params['chirp_slope_ratio']
        
        # 재밍 신호 생성
        jamming_freq = (
            self.config['center_freq'] + 
            jammer_params['freq_offset'] +
            modified_slope * shifted_time
        )
        
        phase = 2 * np.pi * np.cumsum(jamming_freq) / self.config['sampling_rate']
        jamming_signal = jammer_params['power_ratio'] * np.exp(1j * phase)
        
        # 시간 오프셋이 있는 경우 처리
        if offset_samples > 0:
            jamming_signal = np.roll(jamming_signal, offset_samples)
            jamming_signal[:offset_samples] = 0
        
        return jamming_signal
    
    def _add_noise(self, signal, snr_db):
        """가우시안 노이즈 추가"""
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        noise = np.sqrt(noise_power / 2) * (
            np.random.normal(size=self.num_samples) + 
            1j * np.random.normal(size=self.num_samples)
        )
        
        return signal + noise


class SpectrogramGenerator:
    """스펙트로그램 생성 클래스"""
    
    def __init__(self, stft_params=None):
        """
        초기화
        Args:
            stft_params (dict): STFT 파라미터
        """
        if stft_params is None:
            stft_params = self.get_default_stft_params()
        
        self.stft_params = stft_params
    
    def get_default_stft_params(self):
        """기본 STFT 파라미터 - X4M06 실험실 환경 최적화"""
        return {
            'nperseg': 128,      # 세그먼트 길이 (500 샘플에 적합)
            'noverlap': 64,      # 오버랩 샘플 수 (50% 중첩)
            'nfft': 256,         # FFT 포인트 수
            'window': 'hann',    # 윈도우 함수
        }
    
    def generate_spectrogram(self, input_signal, sampling_rate):
        """
        신호로부터 스펙트로그램 생성
        
        Args:
            input_signal (np.ndarray): 입력 신호
            sampling_rate (float): 샘플링 주파수
        
        Returns:
            tuple: (주파수, 시간, 스펙트로그램)
        """
        f, t, Zxx = signal.stft(
            input_signal,
            fs=sampling_rate,
            **self.stft_params
        )
        
        # 크기 스펙트로그램 계산 (dB 스케일)
        spectrogram = 20 * np.log10(np.abs(Zxx) + 1e-12)
        
        return f, t, spectrogram
    
    def normalize_spectrogram(self, spectrogram):
        """스펙트로그램 정규화 (0-1 범위)"""
        spec_min = np.min(spectrogram)
        spec_max = np.max(spectrogram)
        
        if spec_max > spec_min:
            normalized = (spectrogram - spec_min) / (spec_max - spec_min)
        else:
            normalized = np.zeros_like(spectrogram)
        
        return normalized


class DatasetGenerator:
    """합성 데이터셋 생성 클래스"""
    
    def __init__(self, radar_simulator, spectrogram_generator, output_dir):
        """
        초기화
        Args:
            radar_simulator (FMCWRadarSimulator): 레이더 시뮬레이터
            spectrogram_generator (SpectrogramGenerator): 스펙트로그램 생성기
            output_dir (str): 출력 디렉토리
        """
        self.radar_sim = radar_simulator
        self.spec_gen = spectrogram_generator
        self.output_dir = output_dir
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 메타데이터 저장용
        self.metadata = {
            'creation_date': datetime.now().isoformat(),
            'radar_config': radar_simulator.config,
            'stft_params': spectrogram_generator.stft_params,
            'samples': []
        }
    
    def generate_dataset(self, num_samples, save_format='hdf5'):
        """
        데이터셋 생성
        
        Args:
            num_samples (int): 생성할 샘플 수
            save_format (str): 저장 형식 ('hdf5', 'npz', 'pickle')
        
        Returns:
            str: 생성된 데이터셋 파일 경로
        """
        print(f"합성 데이터셋 생성 시작: {num_samples} 샘플")
        
        if save_format == 'hdf5':
            return self._generate_hdf5_dataset(num_samples)
        elif save_format == 'npz':
            return self._generate_npz_dataset(num_samples)
        elif save_format == 'pickle':
            return self._generate_pickle_dataset(num_samples)
        else:
            raise ValueError(f"지원하지 않는 저장 형식: {save_format}")
    
    def _generate_hdf5_dataset(self, num_samples):
        """HDF5 형식으로 데이터셋 생성"""
        output_file = os.path.join(self.output_dir, f'radar_jamming_dataset_{num_samples}.h5')
        
        with h5py.File(output_file, 'w') as f:
            # 첫 번째 샘플로 크기 결정
            clean_signal, target_params = self.radar_sim.generate_clean_signal()
            jammed_signal, jammer_params = self.radar_sim.generate_jammed_signal(clean_signal)
            
            # 스펙트로그램 생성
            _, _, clean_spec = self.spec_gen.generate_spectrogram(
                clean_signal, self.radar_sim.config['sampling_rate']
            )
            _, _, jammed_spec = self.spec_gen.generate_spectrogram(
                jammed_signal, self.radar_sim.config['sampling_rate']
            )
            
            # 정규화
            clean_spec_norm = self.spec_gen.normalize_spectrogram(clean_spec)
            jammed_spec_norm = self.spec_gen.normalize_spectrogram(jammed_spec)
            
            # 데이터셋 생성
            f.create_dataset('clean_spectrograms', 
                           (num_samples,) + clean_spec_norm.shape, 
                           dtype=np.float32)
            f.create_dataset('jammed_spectrograms', 
                           (num_samples,) + jammed_spec_norm.shape, 
                           dtype=np.float32)
            f.create_dataset('clean_signals', 
                           (num_samples, len(clean_signal)), 
                           dtype=np.complex64)
            f.create_dataset('jammed_signals', 
                           (num_samples, len(jammed_signal)), 
                           dtype=np.complex64)
            
            # 첫 번째 샘플 저장
            f['clean_spectrograms'][0] = clean_spec_norm
            f['jammed_spectrograms'][0] = jammed_spec_norm
            f['clean_signals'][0] = clean_signal
            f['jammed_signals'][0] = jammed_signal
            
            # 메타데이터 업데이트
            sample_metadata = {
                'sample_id': 0,
                'target_params': target_params,
                'jammer_params': jammer_params
            }
            self.metadata['samples'].append(sample_metadata)
            
            # 나머지 샘플 생성
            for i in tqdm(range(1, num_samples), desc="데이터셋 생성 중"):
                # 신호 생성
                clean_sig, target_p = self.radar_sim.generate_clean_signal()
                jammed_sig, jammer_p = self.radar_sim.generate_jammed_signal(clean_sig)
                
                # 스펙트로그램 생성 및 정규화
                _, _, clean_sp = self.spec_gen.generate_spectrogram(
                    clean_sig, self.radar_sim.config['sampling_rate']
                )
                _, _, jammed_sp = self.spec_gen.generate_spectrogram(
                    jammed_sig, self.radar_sim.config['sampling_rate']
                )
                
                clean_sp_norm = self.spec_gen.normalize_spectrogram(clean_sp)
                jammed_sp_norm = self.spec_gen.normalize_spectrogram(jammed_sp)
                
                # 데이터 저장
                f['clean_spectrograms'][i] = clean_sp_norm
                f['jammed_spectrograms'][i] = jammed_sp_norm
                f['clean_signals'][i] = clean_sig
                f['jammed_signals'][i] = jammed_sig
                
                # 메타데이터 업데이트
                sample_metadata = {
                    'sample_id': i,
                    'target_params': target_p,
                    'jammer_params': jammer_p
                }
                self.metadata['samples'].append(sample_metadata)
        
        # 메타데이터 별도 저장
        metadata_file = os.path.join(self.output_dir, f'metadata_{num_samples}.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"데이터셋 생성 완료: {output_file}")
        print(f"메타데이터 저장: {metadata_file}")
        
        return output_file
    
    def _generate_npz_dataset(self, num_samples):
        """NPZ 형식으로 데이터셋 생성 (메모리 효율성 고려)"""
        output_file = os.path.join(self.output_dir, f'radar_jamming_dataset_{num_samples}.npz')
        
        clean_spectrograms = []
        jammed_spectrograms = []
        clean_signals = []
        jammed_signals = []
        
        for i in tqdm(range(num_samples), desc="데이터셋 생성 중"):
            # 신호 생성
            clean_signal, target_params = self.radar_sim.generate_clean_signal()
            jammed_signal, jammer_params = self.radar_sim.generate_jammed_signal(clean_signal)
            
            # 스펙트로그램 생성
            _, _, clean_spec = self.spec_gen.generate_spectrogram(
                clean_signal, self.radar_sim.config['sampling_rate']
            )
            _, _, jammed_spec = self.spec_gen.generate_spectrogram(
                jammed_signal, self.radar_sim.config['sampling_rate']
            )
            
            # 정규화
            clean_spec_norm = self.spec_gen.normalize_spectrogram(clean_spec)
            jammed_spec_norm = self.spec_gen.normalize_spectrogram(jammed_spec)
            
            # 리스트에 추가
            clean_spectrograms.append(clean_spec_norm)
            jammed_spectrograms.append(jammed_spec_norm)
            clean_signals.append(clean_signal)
            jammed_signals.append(jammed_signal)
            
            # 메타데이터 업데이트
            sample_metadata = {
                'sample_id': i,
                'target_params': target_params,
                'jammer_params': jammer_params
            }
            self.metadata['samples'].append(sample_metadata)
        
        # 배열로 변환하여 저장
        np.savez_compressed(
            output_file,
            clean_spectrograms=np.array(clean_spectrograms),
            jammed_spectrograms=np.array(jammed_spectrograms),
            clean_signals=np.array(clean_signals),
            jammed_signals=np.array(jammed_signals)
        )
        
        # 메타데이터 별도 저장
        metadata_file = os.path.join(self.output_dir, f'metadata_{num_samples}.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"데이터셋 생성 완료: {output_file}")
        return output_file
    
    def visualize_samples(self, num_visualize=5):
        """생성된 샘플들 시각화"""
        fig, axes = plt.subplots(2, num_visualize, figsize=(15, 6))
        fig.suptitle('합성 레이더 데이터셋 샘플', fontsize=16)
        
        for i in range(num_visualize):
            # 신호 생성
            clean_signal, target_params = self.radar_sim.generate_clean_signal()
            jammed_signal, jammer_params = self.radar_sim.generate_jammed_signal(clean_signal)
            
            # 스펙트로그램 생성
            f, t, clean_spec = self.spec_gen.generate_spectrogram(
                clean_signal, self.radar_sim.config['sampling_rate']
            )
            _, _, jammed_spec = self.spec_gen.generate_spectrogram(
                jammed_signal, self.radar_sim.config['sampling_rate']
            )
            
            # 정규화
            clean_spec_norm = self.spec_gen.normalize_spectrogram(clean_spec)
            jammed_spec_norm = self.spec_gen.normalize_spectrogram(jammed_spec)
            
            # 플롯
            im1 = axes[0, i].imshow(clean_spec_norm, aspect='auto', origin='lower')
            axes[0, i].set_title(f'Clean #{i+1}')
            axes[0, i].set_ylabel('Frequency')
            
            im2 = axes[1, i].imshow(jammed_spec_norm, aspect='auto', origin='lower')
            axes[1, i].set_title(f'Jammed #{i+1}')
            axes[1, i].set_xlabel('Time')
            axes[1, i].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # 저장
        viz_path = os.path.join(self.output_dir, 'sample_visualization.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"시각화 저장: {viz_path}")


def main():
    """메인 실행 함수"""
    # 출력 디렉토리 설정
    output_dir = "synthetic_dataset"
    
    # 레이더 시뮬레이터 초기화 - 1m 이내 실내 실험 최적화
    radar_config = {
        'center_freq': 8.748e9,      # 8.748 GHz (X4M06 중심 주파수)
        'bandwidth': 1.4e9,          # 1.4 GHz (거리 분해능 ~10.7cm)
        'chirp_duration': 8e-6,      # 8μs (PRI 10μs의 80%)
        'prf': 100000,               # 100 kHz (최대 1.5km, 실용적)
        'sampling_rate': 2e6,        # 2 MHz (해상도 향상)
        'target_range': [0.15, 0.9], # 15-90cm (실내 실험 적합)
        'target_velocity': [-2, 2],  # ±2 m/s (실내 이동 속도)
        'target_rcs': [0.01, 1.0],   # 작은 물체 대응 (책, 의자 등)
        'num_jammers': [1, 3],       # 실내 환경 맞춤
        'jammer_power_ratio': [0.5, 2.0],
        'freq_offset_range': [-0.05e9, 0.05e9],  # 범위 축소
        'time_offset_range': [0, 8e-6],     # 8μs 이내 (처프 지속시간과 동일)
        'snr_db': [10, 20],          # 실내 환경 맞춤
    }
    
    radar_sim = FMCWRadarSimulator(radar_config)
    
    # 스펙트로그램 생성기 초기화 - 1m 이내 정밀 분석
    stft_params = {
        'nperseg': 128,              # 더 세밀한 시간 분해능
        'noverlap': 64,              # 50% 중첩
        'nfft': 256,                 # FFT 포인트 (주파수 분해능)
        'window': 'hann',
    }
    
    spec_gen = SpectrogramGenerator(stft_params)
    
    # 데이터셋 생성기 초기화
    dataset_gen = DatasetGenerator(radar_sim, spec_gen, output_dir)
    
    # 샘플 시각화
    print("샘플 시각화 생성 중...")
    dataset_gen.visualize_samples(num_visualize=5)
    
    # 데이터셋 생성
    num_samples = 1000  # 테스트용 작은 크기
    dataset_file = dataset_gen.generate_dataset(num_samples, save_format='hdf5')
    
    print(f"\n데이터셋 생성 완료!")
    print(f"데이터셋 파일: {dataset_file}")
    print(f"메타데이터: {os.path.join(output_dir, f'metadata_{num_samples}.json')}")


if __name__ == "__main__":
    main()