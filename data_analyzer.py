#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
레이더 데이터 분석 및 전처리 유틸리티
합성 및 실제 레이더 데이터의 분석, 시각화, 전처리 기능 제공

연구목적: 딥러닝 모델 학습을 위한 데이터 전처리 파이프라인 구축
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
import h5py
import json
import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import cv2
from tqdm import tqdm

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


class RadarDataAnalyzer:
    """레이더 데이터 분석 클래스"""
    
    def __init__(self, output_dir="analysis_results"):
        """
        초기화
        Args:
            output_dir (str): 분석 결과 출력 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 분석 결과 저장용
        self.analysis_results = {
            'signal_statistics': {},
            'frequency_analysis': {},
            'jamming_analysis': {},
            'data_quality': {}
        }
    
    def load_dataset(self, dataset_path, dataset_type='hdf5'):
        """
        데이터셋 로드
        
        Args:
            dataset_path (str): 데이터셋 파일 경로
            dataset_type (str): 데이터셋 타입 ('hdf5', 'npz')
        
        Returns:
            dict: 로드된 데이터
        """
        print(f"데이터셋 로드 중: {dataset_path}")
        
        if dataset_type == 'hdf5':
            return self._load_hdf5_dataset(dataset_path)
        elif dataset_type == 'npz':
            return self._load_npz_dataset(dataset_path)
        else:
            raise ValueError(f"지원하지 않는 데이터셋 타입: {dataset_type}")
    
    def _load_hdf5_dataset(self, dataset_path):
        """HDF5 데이터셋 로드"""
        data = {}
        with h5py.File(dataset_path, 'r') as f:
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data[key] = f[key][:]
                    print(f"로드된 데이터: {key} - Shape: {data[key].shape}")
        return data
    
    def _load_npz_dataset(self, dataset_path):
        """NPZ 데이터셋 로드"""
        data = np.load(dataset_path)
        result = {}
        for key in data.files:
            result[key] = data[key]
            print(f"로드된 데이터: {key} - Shape: {result[key].shape}")
        return result
    
    def analyze_signal_statistics(self, signals, label=""):
        """
        신호 통계 분석
        
        Args:
            signals (np.ndarray): 신호 데이터 (N, samples)
            label (str): 분석 라벨
        
        Returns:
            dict: 통계 분석 결과
        """
        print(f"신호 통계 분석 시작: {label}")
        
        # 복소수 신호인 경우 크기 계산
        if np.iscomplexobj(signals):
            signal_magnitude = np.abs(signals)
            signal_phase = np.angle(signals)
        else:
            signal_magnitude = signals
            signal_phase = None
        
        stats = {
            'mean': np.mean(signal_magnitude, axis=0),
            'std': np.std(signal_magnitude, axis=0),
            'min': np.min(signal_magnitude, axis=0),
            'max': np.max(signal_magnitude, axis=0),
            'median': np.median(signal_magnitude, axis=0),
            'snr_estimate': self._estimate_snr(signal_magnitude),
            'dynamic_range': np.max(signal_magnitude) / (np.min(signal_magnitude) + 1e-12)
        }
        
        if signal_phase is not None:
            stats['phase_std'] = np.std(signal_phase, axis=0)
            stats['phase_unwrapped'] = np.unwrap(np.mean(signal_phase, axis=0))
        
        # 결과 저장
        self.analysis_results['signal_statistics'][label] = stats
        
        # 시각화
        self._visualize_signal_statistics(stats, label)
        
        return stats
    
    def _estimate_snr(self, signals):
        """SNR 추정"""
        # 간단한 SNR 추정: 신호 전력 대비 잡음 전력
        signal_power = np.mean(signals ** 2, axis=1)
        noise_power = np.var(signals, axis=1)
        
        snr_linear = signal_power / (noise_power + 1e-12)
        snr_db = 10 * np.log10(snr_linear + 1e-12)
        
        return {
            'mean_snr_db': np.mean(snr_db),
            'std_snr_db': np.std(snr_db),
            'min_snr_db': np.min(snr_db),
            'max_snr_db': np.max(snr_db)
        }
    
    def _visualize_signal_statistics(self, stats, label):
        """신호 통계 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'신호 통계 분석: {label}', fontsize=16)
        
        # 평균 및 표준편차
        axes[0, 0].plot(stats['mean'], label='평균', alpha=0.7)
        axes[0, 0].fill_between(
            range(len(stats['mean'])),
            stats['mean'] - stats['std'],
            stats['mean'] + stats['std'],
            alpha=0.3, label='±1σ'
        )
        axes[0, 0].set_title('평균 신호 크기 및 변동성')
        axes[0, 0].set_xlabel('샘플')
        axes[0, 0].set_ylabel('크기')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 최솟값, 최댓값
        axes[0, 1].plot(stats['min'], label='최솟값', alpha=0.7)
        axes[0, 1].plot(stats['max'], label='최댓값', alpha=0.7)
        axes[0, 1].plot(stats['median'], label='중앙값', alpha=0.7)
        axes[0, 1].set_title('신호 범위')
        axes[0, 1].set_xlabel('샘플')
        axes[0, 1].set_ylabel('크기')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # SNR 분포 히스토그램
        if 'snr_estimate' in stats and stats['snr_estimate']:
            snr_info = stats['snr_estimate']
            axes[1, 0].text(0.1, 0.8, f"평균 SNR: {snr_info['mean_snr_db']:.2f} dB", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].text(0.1, 0.7, f"표준편차: {snr_info['std_snr_db']:.2f} dB", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].text(0.1, 0.6, f"범위: {snr_info['min_snr_db']:.2f} ~ {snr_info['max_snr_db']:.2f} dB", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('SNR 정보')
            axes[1, 0].axis('off')
        
        # 동적 범위 및 기타 정보
        axes[1, 1].text(0.1, 0.8, f"동적 범위: {stats['dynamic_range']:.2f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.7, f"전체 평균: {np.mean(stats['mean']):.6f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f"전체 표준편차: {np.mean(stats['std']):.6f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('데이터 품질 지표')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # 저장
        save_path = self.output_dir / f'signal_statistics_{label}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"통계 분석 결과 저장: {save_path}")
    
    def analyze_frequency_spectrum(self, signals, sampling_rate, label=""):
        """
        주파수 스펙트럼 분석
        
        Args:
            signals (np.ndarray): 신호 데이터
            sampling_rate (float): 샘플링 주파수
            label (str): 분석 라벨
        
        Returns:
            dict: 주파수 분석 결과
        """
        print(f"주파수 스펙트럼 분석 시작: {label}")
        
        # 평균 스펙트럼 계산
        freq_spectrum = []
        for signal in signals[:100]:  # 처음 100개 샘플만 분석 (속도 향상)
            fft_result = fft(signal)
            freq_spectrum.append(np.abs(fft_result))
        
        freq_spectrum = np.array(freq_spectrum)
        freq_vector = fftfreq(len(signals[0]), 1/sampling_rate)
        
        analysis_result = {
            'freq_vector': freq_vector,
            'mean_spectrum': np.mean(freq_spectrum, axis=0),
            'std_spectrum': np.std(freq_spectrum, axis=0),
            'peak_frequencies': self._find_peak_frequencies(
                np.mean(freq_spectrum, axis=0), freq_vector
            ),
            'bandwidth_3db': self._calculate_3db_bandwidth(
                np.mean(freq_spectrum, axis=0), freq_vector
            )
        }
        
        # 결과 저장
        self.analysis_results['frequency_analysis'][label] = analysis_result
        
        # 시각화
        self._visualize_frequency_spectrum(analysis_result, label)
        
        return analysis_result
    
    def _find_peak_frequencies(self, spectrum, freq_vector, num_peaks=5):
        """주요 피크 주파수 찾기"""
        # 양의 주파수만 고려
        positive_freq_mask = freq_vector >= 0
        positive_freqs = freq_vector[positive_freq_mask]
        positive_spectrum = spectrum[positive_freq_mask]
        
        # 피크 찾기
        peaks, properties = signal.find_peaks(
            positive_spectrum, 
            height=np.max(positive_spectrum) * 0.1,  # 최댓값의 10% 이상
            distance=len(positive_spectrum) // 20   # 최소 거리
        )
        
        # 상위 피크들 선택
        if len(peaks) > num_peaks:
            peak_heights = positive_spectrum[peaks]
            top_peaks_idx = np.argsort(peak_heights)[-num_peaks:]
            peaks = peaks[top_peaks_idx]
        
        peak_info = [
            {'frequency': positive_freqs[peak], 'magnitude': positive_spectrum[peak]}
            for peak in peaks
        ]
        
        return sorted(peak_info, key=lambda x: x['magnitude'], reverse=True)
    
    def _calculate_3db_bandwidth(self, spectrum, freq_vector):
        """3dB 대역폭 계산"""
        positive_freq_mask = freq_vector >= 0
        positive_freqs = freq_vector[positive_freq_mask]
        positive_spectrum = spectrum[positive_freq_mask]
        
        # 최댓값의 70.7% (-3dB) 지점 찾기
        max_val = np.max(positive_spectrum)
        threshold_3db = max_val / np.sqrt(2)
        
        # 3dB 지점들 찾기
        above_threshold = positive_spectrum >= threshold_3db
        if np.any(above_threshold):
            freq_indices = np.where(above_threshold)[0]
            bandwidth = positive_freqs[freq_indices[-1]] - positive_freqs[freq_indices[0]]
            return float(bandwidth)
        else:
            return 0.0
    
    def _visualize_frequency_spectrum(self, analysis_result, label):
        """주파수 스펙트럼 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'주파수 스펙트럼 분석: {label}', fontsize=16)
        
        freq_vector = analysis_result['freq_vector']
        mean_spectrum = analysis_result['mean_spectrum']
        std_spectrum = analysis_result['std_spectrum']
        
        # 양의 주파수만 표시
        positive_mask = freq_vector >= 0
        pos_freq = freq_vector[positive_mask]
        pos_spectrum = mean_spectrum[positive_mask]
        pos_std = std_spectrum[positive_mask]
        
        # 스펙트럼 플롯
        axes[0].plot(pos_freq / 1e6, 20 * np.log10(pos_spectrum + 1e-12), 
                    alpha=0.8, linewidth=1)
        axes[0].fill_between(
            pos_freq / 1e6,
            20 * np.log10(pos_spectrum - pos_std + 1e-12),
            20 * np.log10(pos_spectrum + pos_std + 1e-12),
            alpha=0.3
        )
        axes[0].set_xlabel('주파수 (MHz)')
        axes[0].set_ylabel('크기 (dB)')
        axes[0].set_title('평균 주파수 스펙트럼')
        axes[0].grid(True, alpha=0.3)
        
        # 피크 주파수 표시
        peak_freqs = analysis_result['peak_frequencies']
        if peak_freqs:
            for i, peak in enumerate(peak_freqs[:3]):  # 상위 3개만 표시
                axes[0].axvline(peak['frequency'] / 1e6, color='red', 
                               linestyle='--', alpha=0.7, 
                               label=f'Peak {i+1}: {peak["frequency"]/1e6:.2f} MHz')
            axes[0].legend()
        
        # 스펙트로그램 (시간-주파수)
        sample_signal = mean_spectrum  # 평균 스펙트럼을 대표로 사용
        f_stft, t_stft, Zxx = signal.stft(sample_signal, nperseg=min(256, len(sample_signal)//4))
        
        im = axes[1].imshow(
            20 * np.log10(np.abs(Zxx) + 1e-12),
            aspect='auto',
            origin='lower',
            extent=[t_stft[0], t_stft[-1], f_stft[0]/1e6, f_stft[-1]/1e6]
        )
        axes[1].set_xlabel('시간 (s)')
        axes[1].set_ylabel('주파수 (MHz)')
        axes[1].set_title('스펙트로그램')
        plt.colorbar(im, ax=axes[1], label='크기 (dB)')
        
        plt.tight_layout()
        
        # 저장
        save_path = self.output_dir / f'frequency_analysis_{label}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"주파수 분석 결과 저장: {save_path}")
    
    def compare_clean_vs_jammed(self, clean_signals, jammed_signals, label=""):
        """
        깨끗한 신호 vs 재밍된 신호 비교 분석
        
        Args:
            clean_signals (np.ndarray): 깨끗한 신호
            jammed_signals (np.ndarray): 재밍된 신호
            label (str): 분석 라벨
        
        Returns:
            dict: 비교 분석 결과
        """
        print(f"깨끗한 신호 vs 재밍 신호 비교 분석: {label}")
        
        # 신호 크기 계산
        clean_magnitude = np.abs(clean_signals) if np.iscomplexobj(clean_signals) else clean_signals
        jammed_magnitude = np.abs(jammed_signals) if np.iscomplexobj(jammed_signals) else jammed_signals
        
        # 비교 지표 계산
        comparison_result = {
            'power_ratio': np.mean(jammed_magnitude**2) / np.mean(clean_magnitude**2),
            'correlation': self._calculate_correlation(clean_magnitude, jammed_magnitude),
            'distortion_metrics': self._calculate_distortion_metrics(clean_magnitude, jammed_magnitude),
            'frequency_difference': self._analyze_frequency_difference(clean_signals, jammed_signals)
        }
        
        # 결과 저장
        self.analysis_results['jamming_analysis'][label] = comparison_result
        
        # 시각화
        self._visualize_clean_vs_jammed(clean_signals, jammed_signals, comparison_result, label)
        
        return comparison_result
    
    def _calculate_correlation(self, clean_signals, jammed_signals):
        """신호 간 상관계수 계산"""
        correlations = []
        for clean, jammed in zip(clean_signals[:100], jammed_signals[:100]):
            corr = np.corrcoef(clean, jammed)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        return {
            'mean': np.mean(correlations),
            'std': np.std(correlations),
            'min': np.min(correlations),
            'max': np.max(correlations)
        }
    
    def _calculate_distortion_metrics(self, clean_signals, jammed_signals):
        """왜곡 지표 계산"""
        # MSE, MAE, SNR 등
        mse_values = []
        mae_values = []
        
        for clean, jammed in zip(clean_signals[:100], jammed_signals[:100]):
            mse = np.mean((clean - jammed)**2)
            mae = np.mean(np.abs(clean - jammed))
            mse_values.append(mse)
            mae_values.append(mae)
        
        return {
            'mse': {'mean': np.mean(mse_values), 'std': np.std(mse_values)},
            'mae': {'mean': np.mean(mae_values), 'std': np.std(mae_values)},
        }
    
    def _analyze_frequency_difference(self, clean_signals, jammed_signals):
        """주파수 영역에서의 차이 분석"""
        # 간단한 FFT 기반 차이 분석
        clean_fft = np.abs(fft(clean_signals[0]))
        jammed_fft = np.abs(fft(jammed_signals[0]))
        
        frequency_diff = np.mean(np.abs(jammed_fft - clean_fft))
        
        return {
            'mean_frequency_difference': frequency_diff,
            'frequency_correlation': np.corrcoef(clean_fft, jammed_fft)[0, 1]
        }
    
    def _visualize_clean_vs_jammed(self, clean_signals, jammed_signals, comparison_result, label):
        """깨끗한 vs 재밍 신호 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'깨끗한 신호 vs 재밍 신호 비교: {label}', fontsize=16)
        
        # 신호 크기 계산
        clean_mag = np.abs(clean_signals) if np.iscomplexobj(clean_signals) else clean_signals
        jammed_mag = np.abs(jammed_signals) if np.iscomplexobj(jammed_signals) else jammed_signals
        
        # 첫 번째 샘플 비교
        axes[0, 0].plot(clean_mag[0], label='깨끗한 신호', alpha=0.7)
        axes[0, 0].plot(jammed_mag[0], label='재밍 신호', alpha=0.7)
        axes[0, 0].set_title('첫 번째 샘플 비교')
        axes[0, 0].set_xlabel('샘플')
        axes[0, 0].set_ylabel('크기')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 평균 신호 비교
        axes[0, 1].plot(np.mean(clean_mag, axis=0), label='깨끗한 신호 평균', alpha=0.7)
        axes[0, 1].plot(np.mean(jammed_mag, axis=0), label='재밍 신호 평균', alpha=0.7)
        axes[0, 1].set_title('평균 신호 비교')
        axes[0, 1].set_xlabel('샘플')
        axes[0, 1].set_ylabel('크기')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 차이 신호
        diff_signal = np.mean(jammed_mag - clean_mag, axis=0)
        axes[0, 2].plot(diff_signal)
        axes[0, 2].set_title('차이 신호 (재밍 - 깨끗함)')
        axes[0, 2].set_xlabel('샘플')
        axes[0, 2].set_ylabel('크기 차이')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 상관계수 히스토그램
        if 'correlation' in comparison_result:
            corr_data = [comparison_result['correlation']['mean']]
            axes[1, 0].bar(['상관계수'], corr_data)
            axes[1, 0].set_title(f'상관계수: {comparison_result["correlation"]["mean"]:.3f}')
            axes[1, 0].set_ylim([0, 1])
        
        # 전력비 및 왜곡 지표
        metrics_text = f"""
        전력비: {comparison_result['power_ratio']:.3f}
        MSE: {comparison_result['distortion_metrics']['mse']['mean']:.6f}
        MAE: {comparison_result['distortion_metrics']['mae']['mean']:.6f}
        주파수 상관계수: {comparison_result['frequency_difference']['frequency_correlation']:.3f}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='center')
        axes[1, 1].set_title('왜곡 지표')
        axes[1, 1].axis('off')
        
        # 스펙트럼 비교
        clean_spectrum = np.abs(fft(clean_signals[0]))
        jammed_spectrum = np.abs(fft(jammed_signals[0]))
        freq_vector = fftfreq(len(clean_signals[0]))
        
        positive_mask = freq_vector >= 0
        axes[1, 2].semilogy(freq_vector[positive_mask], clean_spectrum[positive_mask], 
                           label='깨끗한 신호', alpha=0.7)
        axes[1, 2].semilogy(freq_vector[positive_mask], jammed_spectrum[positive_mask], 
                           label='재밍 신호', alpha=0.7)
        axes[1, 2].set_title('주파수 스펙트럼 비교')
        axes[1, 2].set_xlabel('정규화된 주파수')
        axes[1, 2].set_ylabel('크기 (로그 스케일)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        save_path = self.output_dir / f'clean_vs_jammed_{label}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"비교 분석 결과 저장: {save_path}")
    
    def generate_analysis_report(self):
        """종합 분석 보고서 생성"""
        report_path = self.output_dir / 'analysis_report.json'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"분석 보고서 저장: {report_path}")
        
        # 텍스트 보고서도 생성
        self._generate_text_report()
    
    def _generate_text_report(self):
        """텍스트 형식 분석 보고서 생성"""
        report_path = self.output_dir / 'analysis_summary.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 레이더 데이터 분석 보고서 ===\n\n")
            
            # 신호 통계 요약
            if 'signal_statistics' in self.analysis_results:
                f.write("1. 신호 통계 분석\n")
                f.write("-" * 30 + "\n")
                for label, stats in self.analysis_results['signal_statistics'].items():
                    f.write(f"- {label}:\n")
                    if 'snr_estimate' in stats and stats['snr_estimate']:
                        f.write(f"  평균 SNR: {stats['snr_estimate']['mean_snr_db']:.2f} dB\n")
                    f.write(f"  동적 범위: {stats['dynamic_range']:.2f}\n")
                    f.write(f"  평균 신호 크기: {np.mean(stats['mean']):.6f}\n\n")
            
            # 주파수 분석 요약
            if 'frequency_analysis' in self.analysis_results:
                f.write("2. 주파수 분석\n")
                f.write("-" * 30 + "\n")
                for label, analysis in self.analysis_results['frequency_analysis'].items():
                    f.write(f"- {label}:\n")
                    f.write(f"  3dB 대역폭: {analysis['bandwidth_3db']/1e6:.2f} MHz\n")
                    if analysis['peak_frequencies']:
                        f.write("  주요 피크 주파수:\n")
                        for i, peak in enumerate(analysis['peak_frequencies'][:3]):
                            f.write(f"    {i+1}. {peak['frequency']/1e6:.2f} MHz\n")
                    f.write("\n")
            
            # 재밍 분석 요약
            if 'jamming_analysis' in self.analysis_results:
                f.write("3. 재밍 분석\n")
                f.write("-" * 30 + "\n")
                for label, analysis in self.analysis_results['jamming_analysis'].items():
                    f.write(f"- {label}:\n")
                    f.write(f"  전력비: {analysis['power_ratio']:.3f}\n")
                    f.write(f"  상관계수: {analysis['correlation']['mean']:.3f}\n")
                    f.write(f"  MSE: {analysis['distortion_metrics']['mse']['mean']:.6f}\n")
                    f.write(f"  주파수 영역 상관계수: {analysis['frequency_difference']['frequency_correlation']:.3f}\n\n")
        
        print(f"요약 보고서 저장: {report_path}")


class DataPreprocessor:
    """딥러닝 모델용 데이터 전처리 클래스"""
    
    def __init__(self, config=None):
        """
        초기화
        Args:
            config (dict): 전처리 설정
        """
        self.config = config if config else self.get_default_config()
        self.scaler = None
        self.is_fitted = False
    
    def get_default_config(self):
        """기본 전처리 설정"""
        return {
            'normalization_method': 'minmax',  # 'minmax', 'standard', 'robust'
            'spectrogram_params': {
                'nperseg': 256,
                'noverlap': 128,
                'nfft': 512,
                'window': 'hann'
            },
            'image_size': (256, 256),  # 스펙트로그램 이미지 크기
            'data_augmentation': {
                'noise_level': 0.01,
                'time_shift_samples': 10,
                'frequency_mask_width': 5
            }
        }
    
    def preprocess_for_training(self, clean_signals, jammed_signals, sampling_rate):
        """
        딥러닝 모델 학습용 데이터 전처리
        
        Args:
            clean_signals (np.ndarray): 깨끗한 신호
            jammed_signals (np.ndarray): 재밍된 신호
            sampling_rate (float): 샘플링 주파수
        
        Returns:
            tuple: (입력 스펙트로그램, 타겟 스펙트로그램, 전처리 정보)
        """
        print("딥러닝 학습용 데이터 전처리 시작...")
        
        # 스펙트로그램 생성
        input_spectrograms = []
        target_spectrograms = []
        
        for i, (clean, jammed) in enumerate(tqdm(zip(clean_signals, jammed_signals), 
                                                   desc="스펙트로그램 생성", 
                                                   total=len(clean_signals))):
            # STFT 계산
            _, _, clean_spec = signal.stft(clean, fs=sampling_rate, **self.config['spectrogram_params'])
            _, _, jammed_spec = signal.stft(jammed, fs=sampling_rate, **self.config['spectrogram_params'])
            
            # 크기 스펙트로그램 (dB 스케일)
            clean_spec_db = 20 * np.log10(np.abs(clean_spec) + 1e-12)
            jammed_spec_db = 20 * np.log10(np.abs(jammed_spec) + 1e-12)
            
            # 이미지 크기로 리사이즈
            clean_resized = cv2.resize(clean_spec_db, self.config['image_size'])
            jammed_resized = cv2.resize(jammed_spec_db, self.config['image_size'])
            
            target_spectrograms.append(clean_resized)
            input_spectrograms.append(jammed_resized)
        
        input_spectrograms = np.array(input_spectrograms)
        target_spectrograms = np.array(target_spectrograms)
        
        # 정규화
        input_normalized, target_normalized = self._normalize_spectrograms(
            input_spectrograms, target_spectrograms
        )
        
        # 채널 차원 추가 (grayscale image)
        if len(input_normalized.shape) == 3:
            input_normalized = input_normalized[..., np.newaxis]
            target_normalized = target_normalized[..., np.newaxis]
        
        preprocessing_info = {
            'input_shape': input_normalized.shape,
            'target_shape': target_normalized.shape,
            'normalization_method': self.config['normalization_method'],
            'spectrogram_params': self.config['spectrogram_params'],
            'image_size': self.config['image_size']
        }
        
        print(f"전처리 완료 - 입력: {input_normalized.shape}, 타겟: {target_normalized.shape}")
        
        return input_normalized, target_normalized, preprocessing_info
    
    def _normalize_spectrograms(self, input_spectrograms, target_spectrograms):
        """스펙트로그램 정규화"""
        if self.config['normalization_method'] == 'minmax':
            # Min-Max 정규화 (각 스펙트로그램 개별적으로)
            input_normalized = np.zeros_like(input_spectrograms)
            target_normalized = np.zeros_like(target_spectrograms)
            
            for i in range(len(input_spectrograms)):
                # 입력 스펙트로그램 정규화
                inp_min, inp_max = input_spectrograms[i].min(), input_spectrograms[i].max()
                if inp_max > inp_min:
                    input_normalized[i] = (input_spectrograms[i] - inp_min) / (inp_max - inp_min)
                else:
                    input_normalized[i] = np.zeros_like(input_spectrograms[i])
                
                # 타겟 스펙트로그램 정규화
                tgt_min, tgt_max = target_spectrograms[i].min(), target_spectrograms[i].max()
                if tgt_max > tgt_min:
                    target_normalized[i] = (target_spectrograms[i] - tgt_min) / (tgt_max - tgt_min)
                else:
                    target_normalized[i] = np.zeros_like(target_spectrograms[i])
        
        elif self.config['normalization_method'] == 'standard':
            # 표준화
            input_normalized = (input_spectrograms - np.mean(input_spectrograms)) / np.std(input_spectrograms)
            target_normalized = (target_spectrograms - np.mean(target_spectrograms)) / np.std(target_spectrograms)
        
        else:
            # 정규화 없음
            input_normalized = input_spectrograms.copy()
            target_normalized = target_spectrograms.copy()
        
        return input_normalized, target_normalized
    
    def apply_data_augmentation(self, spectrograms, labels=None):
        """
        데이터 증강 적용
        
        Args:
            spectrograms (np.ndarray): 스펙트로그램 데이터
            labels (np.ndarray): 라벨 (있는 경우)
        
        Returns:
            tuple: 증강된 데이터
        """
        print("데이터 증강 적용 중...")
        
        augmented_spectrograms = []
        augmented_labels = [] if labels is not None else None
        
        for i, spec in enumerate(tqdm(spectrograms, desc="데이터 증강")):
            # 원본 데이터 추가
            augmented_spectrograms.append(spec)
            if labels is not None:
                augmented_labels.append(labels[i])
            
            # 노이즈 추가
            noisy_spec = spec + np.random.normal(0, self.config['data_augmentation']['noise_level'], spec.shape)
            augmented_spectrograms.append(noisy_spec)
            if labels is not None:
                augmented_labels.append(labels[i])
            
            # 시간 이동 (순환 이동)
            shift_samples = self.config['data_augmentation']['time_shift_samples']
            shifted_spec = np.roll(spec, shift_samples, axis=1)  # 시간 축으로 이동
            augmented_spectrograms.append(shifted_spec)
            if labels is not None:
                augmented_labels.append(labels[i])
            
            # 주파수 마스킹
            masked_spec = spec.copy()
            freq_mask_width = self.config['data_augmentation']['frequency_mask_width']
            start_freq = np.random.randint(0, spec.shape[0] - freq_mask_width)
            masked_spec[start_freq:start_freq + freq_mask_width, :] = 0
            augmented_spectrograms.append(masked_spec)
            if labels is not None:
                augmented_labels.append(labels[i])
        
        result = [np.array(augmented_spectrograms)]
        if augmented_labels is not None:
            result.append(np.array(augmented_labels))
        
        print(f"데이터 증강 완료: {len(spectrograms)} -> {len(augmented_spectrograms)}")
        
        return tuple(result) if len(result) > 1 else result[0]


def main():
    """메인 실행 함수"""
    print("레이더 데이터 분석 및 전처리 시작")
    print("=" * 50)
    
    # 분석기 초기화
    analyzer = RadarDataAnalyzer()
    
    # 데이터셋 로드 (예시)
    dataset_path = "synthetic_dataset/radar_jamming_dataset_1000.h5"
    
    if os.path.exists(dataset_path):
        print(f"데이터셋 로드: {dataset_path}")
        data = analyzer.load_dataset(dataset_path)
        
        # 신호 분석
        if 'clean_signals' in data and 'jammed_signals' in data:
            # 통계 분석
            clean_stats = analyzer.analyze_signal_statistics(data['clean_signals'], "Clean")
            jammed_stats = analyzer.analyze_signal_statistics(data['jammed_signals'], "Jammed")
            
            # 주파수 분석
            sampling_rate = 1e6  # 1 MHz (설정에 따라 조정)
            clean_freq = analyzer.analyze_frequency_spectrum(
                data['clean_signals'], sampling_rate, "Clean"
            )
            jammed_freq = analyzer.analyze_frequency_spectrum(
                data['jammed_signals'], sampling_rate, "Jammed"
            )
            
            # 비교 분석
            comparison = analyzer.compare_clean_vs_jammed(
                data['clean_signals'], data['jammed_signals'], "Comparison"
            )
            
            # 보고서 생성
            analyzer.generate_analysis_report()
        
        # 전처리 테스트
        if 'clean_signals' in data and 'jammed_signals' in data:
            preprocessor = DataPreprocessor()
            
            # 작은 샘플로 테스트
            test_size = min(100, len(data['clean_signals']))
            clean_test = data['clean_signals'][:test_size]
            jammed_test = data['jammed_signals'][:test_size]
            
            input_data, target_data, preprocess_info = preprocessor.preprocess_for_training(
                clean_test, jammed_test, sampling_rate
            )
            
            print(f"전처리된 데이터 형태: {input_data.shape} -> {target_data.shape}")
            
            # 데이터 증강 테스트
            augmented_input = preprocessor.apply_data_augmentation(input_data[:10])  # 10개 샘플만
            print(f"데이터 증강 결과: {input_data[:10].shape} -> {augmented_input.shape}")
    
    else:
        print(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
        print("먼저 jamming_simulator.py를 실행하여 데이터셋을 생성하세요.")


if __name__ == "__main__":
    main()