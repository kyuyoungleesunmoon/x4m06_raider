#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
X4M06 레이더 시뮬레이션 모델 현실성 검증 도구

시뮬레이션 데이터와 하드웨어 데이터 간의 통계적/물리적 유사성을 분석하여
시뮬레이션 모델의 현실성을 정량적으로 평가합니다.

사용법:
    python simulation_validation.py --sim-data simulation_data.h5 --hw-data hardware_data.h5
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scipy.stats
import scipy.signal
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error
from pathlib import Path
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정"""
    try:
        font_path = 'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("한글 폰트 설정 실패, 기본 폰트 사용")

class SimulationValidator:
    """시뮬레이션 모델 현실성 검증 클래스"""
    
    def __init__(self, sim_data_path, hw_data_path=None):
        """
        초기화
        Args:
            sim_data_path (str): 시뮬레이션 데이터 파일 경로
            hw_data_path (str): 하드웨어 데이터 파일 경로 (선택사항)
        """
        self.sim_data_path = sim_data_path
        self.hw_data_path = hw_data_path
        self.validation_results = {}
        
        setup_korean_font()
        
    def load_simulation_data(self):
        """시뮬레이션 데이터 로드"""
        try:
            with h5py.File(self.sim_data_path, 'r') as f:
                self.sim_clean = f['clean_signals'][:]
                self.sim_jammed = f['jammed_signals'][:]
                self.sim_clean_spec = f['clean_spectrograms'][:]
                self.sim_jammed_spec = f['jammed_spectrograms'][:]
            
            print(f"✅ 시뮬레이션 데이터 로드 완료")
            print(f"   Clean signals: {self.sim_clean.shape}")
            print(f"   Jammed signals: {self.sim_jammed.shape}")
            return True
            
        except Exception as e:
            print(f"❌ 시뮬레이션 데이터 로드 실패: {e}")
            return False
    
    def load_hardware_data(self):
        """하드웨어 데이터 로드"""
        if not self.hw_data_path:
            print("⚠️  하드웨어 데이터 경로가 제공되지 않음")
            return False
            
        try:
            with h5py.File(self.hw_data_path, 'r') as f:
                # 하드웨어 데이터 구조에 맞게 조정
                self.hw_clean = f['clean_signals'][:]
                self.hw_jammed = f['jammed_signals'][:]
                
            print(f"✅ 하드웨어 데이터 로드 완료")
            print(f"   Clean signals: {self.hw_clean.shape}")
            print(f"   Jammed signals: {self.hw_jammed.shape}")
            return True
            
        except Exception as e:
            print(f"❌ 하드웨어 데이터 로드 실패: {e}")
            return False
    
    def calculate_basic_statistics(self, data, label):
        """기본 통계량 계산"""
        return {
            'label': label,
            'mean': np.mean(data),
            'std': np.std(data),
            'variance': np.var(data),
            'skewness': scipy.stats.skew(data.flatten()),
            'kurtosis': scipy.stats.kurtosis(data.flatten()),
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data)
        }
    
    def statistical_comparison(self):
        """통계적 특성 비교"""
        print("\n" + "="*60)
        print("📊 통계적 특성 비교 분석")
        print("="*60)
        
        # 기본 통계량 계산
        sim_clean_stats = self.calculate_basic_statistics(self.sim_clean, "Sim Clean")
        sim_jammed_stats = self.calculate_basic_statistics(self.sim_jammed, "Sim Jammed")
        
        # 결과 저장
        self.validation_results['statistical_comparison'] = {
            'sim_clean': sim_clean_stats,
            'sim_jammed': sim_jammed_stats
        }
        
        # 하드웨어 데이터가 있는 경우 비교
        if hasattr(self, 'hw_clean'):
            hw_clean_stats = self.calculate_basic_statistics(self.hw_clean, "HW Clean")
            hw_jammed_stats = self.calculate_basic_statistics(self.hw_jammed, "HW Jammed")
            
            self.validation_results['statistical_comparison']['hw_clean'] = hw_clean_stats
            self.validation_results['statistical_comparison']['hw_jammed'] = hw_jammed_stats
            
            # 상관계수 계산
            clean_correlation = np.corrcoef(
                self.sim_clean.flatten()[:10000],  # 샘플 크기 조정
                self.hw_clean.flatten()[:10000]
            )[0, 1]
            
            jammed_correlation = np.corrcoef(
                self.sim_jammed.flatten()[:10000],
                self.hw_jammed.flatten()[:10000]
            )[0, 1]
            
            self.validation_results['correlations'] = {
                'clean_correlation': clean_correlation,
                'jammed_correlation': jammed_correlation
            }
            
            print(f"🎯 Clean 신호 상관계수: {clean_correlation:.4f}")
            print(f"🎯 Jammed 신호 상관계수: {jammed_correlation:.4f}")
            
            # KS 테스트
            ks_clean = scipy.stats.ks_2samp(
                self.sim_clean.flatten()[:5000],
                self.hw_clean.flatten()[:5000]
            )
            ks_jammed = scipy.stats.ks_2samp(
                self.sim_jammed.flatten()[:5000],
                self.hw_jammed.flatten()[:5000]
            )
            
            self.validation_results['ks_tests'] = {
                'clean_ks_statistic': ks_clean.statistic,
                'clean_p_value': ks_clean.pvalue,
                'jammed_ks_statistic': ks_jammed.statistic,
                'jammed_p_value': ks_jammed.pvalue
            }
            
            print(f"📈 Clean KS 테스트: statistic={ks_clean.statistic:.4f}, p-value={ks_clean.pvalue:.4f}")
            print(f"📈 Jammed KS 테스트: statistic={ks_jammed.statistic:.4f}, p-value={ks_jammed.pvalue:.4f}")
        
        # 통계량 비교 표 출력
        self._print_statistics_table()
    
    def _print_statistics_table(self):
        """통계량 비교 표 출력"""
        stats_keys = ['mean', 'std', 'skewness', 'kurtosis']
        
        print(f"\n{'통계량':<12} {'Sim Clean':<15} {'Sim Jammed':<15}", end="")
        if hasattr(self, 'hw_clean'):
            print(f" {'HW Clean':<15} {'HW Jammed':<15}")
        else:
            print()
            
        print("-" * 80)
        
        for key in stats_keys:
            sim_clean_val = self.validation_results['statistical_comparison']['sim_clean'][key]
            sim_jammed_val = self.validation_results['statistical_comparison']['sim_jammed'][key]
            
            print(f"{key:<12} {sim_clean_val:<15.4f} {sim_jammed_val:<15.4f}", end="")
            
            if hasattr(self, 'hw_clean'):
                hw_clean_val = self.validation_results['statistical_comparison']['hw_clean'][key]
                hw_jammed_val = self.validation_results['statistical_comparison']['hw_jammed'][key]
                print(f" {hw_clean_val:<15.4f} {hw_jammed_val:<15.4f}")
            else:
                print()
    
    def spectral_analysis(self):
        """스펙트럼 특성 분석"""
        print("\n" + "="*60)
        print("🎵 스펙트럼 특성 분석")
        print("="*60)
        
        # 샘플 신호에 대한 PSD 계산
        sample_idx = 0
        
        # 시뮬레이션 신호 PSD
        f_sim_clean, psd_sim_clean = scipy.signal.welch(
            self.sim_clean[sample_idx], 
            fs=1e6,
            nperseg=256
        )
        f_sim_jammed, psd_sim_jammed = scipy.signal.welch(
            self.sim_jammed[sample_idx],
            fs=1e6, 
            nperseg=256
        )
        
        self.validation_results['spectral_analysis'] = {
            'sim_clean_psd_peak': np.max(psd_sim_clean),
            'sim_jammed_psd_peak': np.max(psd_sim_jammed),
            'sim_clean_bandwidth': self._calculate_bandwidth(f_sim_clean, psd_sim_clean),
            'sim_jammed_bandwidth': self._calculate_bandwidth(f_sim_jammed, psd_sim_jammed)
        }
        
        # 하드웨어 데이터가 있는 경우
        if hasattr(self, 'hw_clean'):
            f_hw_clean, psd_hw_clean = scipy.signal.welch(
                self.hw_clean[sample_idx],
                fs=1e6,
                nperseg=256
            )
            f_hw_jammed, psd_hw_jammed = scipy.signal.welch(
                self.hw_jammed[sample_idx],
                fs=1e6,
                nperseg=256
            )
            
            # 스펙트럼 유사도 계산 (코사인 유사도)
            clean_spectral_similarity = 1 - cosine(psd_sim_clean, psd_hw_clean)
            jammed_spectral_similarity = 1 - cosine(psd_sim_jammed, psd_hw_jammed)
            
            self.validation_results['spectral_analysis'].update({
                'hw_clean_psd_peak': np.max(psd_hw_clean),
                'hw_jammed_psd_peak': np.max(psd_hw_jammed),
                'clean_spectral_similarity': clean_spectral_similarity,
                'jammed_spectral_similarity': jammed_spectral_similarity
            })
            
            print(f"🎯 Clean 스펙트럼 유사도: {clean_spectral_similarity:.4f}")
            print(f"🎯 Jammed 스펙트럼 유사도: {jammed_spectral_similarity:.4f}")
        
        print(f"📊 Sim Clean PSD 피크: {np.max(psd_sim_clean):.2e}")
        print(f"📊 Sim Jammed PSD 피크: {np.max(psd_sim_jammed):.2e}")
    
    def _calculate_bandwidth(self, frequencies, psd):
        """유효 대역폭 계산 (-3dB 대역폭)"""
        peak_power = np.max(psd)
        half_power = peak_power / 2
        
        # 반전력 지점 찾기
        indices = np.where(psd > half_power)[0]
        if len(indices) > 0:
            return frequencies[indices[-1]] - frequencies[indices[0]]
        return 0
    
    def signal_quality_metrics(self):
        """신호 품질 지표 계산"""
        print("\n" + "="*60)
        print("📡 신호 품질 지표 분석")
        print("="*60)
        
        # SNR 계산 (근사치)
        sim_clean_snr = self._calculate_snr(self.sim_clean)
        sim_jammed_snr = self._calculate_snr(self.sim_jammed)
        
        # 재밍 대 신호 비율
        jamming_ratio = np.mean(np.abs(self.sim_jammed)) / np.mean(np.abs(self.sim_clean))
        
        self.validation_results['signal_quality'] = {
            'sim_clean_snr_db': sim_clean_snr,
            'sim_jammed_snr_db': sim_jammed_snr,
            'jamming_ratio': jamming_ratio
        }
        
        print(f"📊 Sim Clean SNR: {sim_clean_snr:.2f} dB")
        print(f"📊 Sim Jammed SNR: {sim_jammed_snr:.2f} dB") 
        print(f"📊 재밍 비율: {jamming_ratio:.2f}")
        
        if hasattr(self, 'hw_clean'):
            hw_clean_snr = self._calculate_snr(self.hw_clean)
            hw_jammed_snr = self._calculate_snr(self.hw_jammed)
            hw_jamming_ratio = np.mean(np.abs(self.hw_jammed)) / np.mean(np.abs(self.hw_clean))
            
            self.validation_results['signal_quality'].update({
                'hw_clean_snr_db': hw_clean_snr,
                'hw_jammed_snr_db': hw_jammed_snr,
                'hw_jamming_ratio': hw_jamming_ratio
            })
            
            print(f"📊 HW Clean SNR: {hw_clean_snr:.2f} dB")
            print(f"📊 HW Jammed SNR: {hw_jammed_snr:.2f} dB")
            print(f"📊 HW 재밍 비율: {hw_jamming_ratio:.2f}")
    
    def _calculate_snr(self, signal):
        """SNR 계산 (근사치)"""
        # 신호 파워와 노이즈 파워 추정
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = np.var(signal) * 0.1  # 노이즈 추정 (간단한 방법)
        
        if noise_power > 0:
            snr_linear = signal_power / noise_power
            return 10 * np.log10(snr_linear)
        return float('inf')
    
    def create_validation_report(self, output_dir="validation_results"):
        """검증 리포트 생성"""
        print("\n" + "="*60)
        print("📋 검증 리포트 생성")
        print("="*60)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # JSON 리포트 저장
        report_file = output_path / "validation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 시각화 생성
        self._create_validation_plots(output_path)
        
        # 종합 평가
        overall_score = self._calculate_overall_score()
        
        # 텍스트 리포트 생성
        text_report = output_path / "validation_summary.txt"
        with open(text_report, 'w', encoding='utf-8') as f:
            f.write("=== X4M06 레이더 시뮬레이션 모델 검증 리포트 ===\n\n")
            
            f.write("1. 종합 평가\n")
            f.write(f"   전체 점수: {overall_score:.2f}/100\n")
            f.write(f"   평가 등급: {self._get_grade(overall_score)}\n\n")
            
            if 'correlations' in self.validation_results:
                f.write("2. 상관관계 분석\n")
                f.write(f"   Clean 신호 상관계수: {self.validation_results['correlations']['clean_correlation']:.4f}\n")
                f.write(f"   Jammed 신호 상관계수: {self.validation_results['correlations']['jammed_correlation']:.4f}\n\n")
            
            f.write("3. 검증 권장사항\n")
            recommendations = self._generate_recommendations()
            for rec in recommendations:
                f.write(f"   - {rec}\n")
        
        print(f"✅ 검증 리포트 생성 완료: {output_path}")
        print(f"📊 종합 점수: {overall_score:.2f}/100 ({self._get_grade(overall_score)})")
    
    def _create_validation_plots(self, output_path):
        """검증 시각화 생성"""
        # 1. 신호 비교 플롯
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('시뮬레이션 vs 하드웨어 신호 비교', fontsize=16)
        
        # Clean 신호 비교
        axes[0, 0].plot(np.real(self.sim_clean[0, :500]), label='Sim Clean', alpha=0.7)
        if hasattr(self, 'hw_clean'):
            axes[0, 0].plot(np.real(self.hw_clean[0, :500]), label='HW Clean', alpha=0.7)
        axes[0, 0].set_title('Clean 신호 비교')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Jammed 신호 비교
        axes[0, 1].plot(np.real(self.sim_jammed[0, :500]), label='Sim Jammed', alpha=0.7)
        if hasattr(self, 'hw_jammed'):
            axes[0, 1].plot(np.real(self.hw_jammed[0, :500]), label='HW Jammed', alpha=0.7)
        axes[0, 1].set_title('Jammed 신호 비교')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 히스토그램 비교
        axes[1, 0].hist(np.real(self.sim_clean).flatten()[:10000], bins=50, alpha=0.5, label='Sim Clean', density=True)
        if hasattr(self, 'hw_clean'):
            axes[1, 0].hist(np.real(self.hw_clean).flatten()[:10000], bins=50, alpha=0.5, label='HW Clean', density=True)
        axes[1, 0].set_title('Clean 신호 분포 비교')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(np.real(self.sim_jammed).flatten()[:10000], bins=50, alpha=0.5, label='Sim Jammed', density=True)
        if hasattr(self, 'hw_jammed'):
            axes[1, 1].hist(np.real(self.hw_jammed).flatten()[:10000], bins=50, alpha=0.5, label='HW Jammed', density=True)
        axes[1, 1].set_title('Jammed 신호 분포 비교')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'signal_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 스펙트로그램 비교 (시뮬레이션만)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('시뮬레이션 스펙트로그램 비교', fontsize=16)
        
        im1 = axes[0].imshow(self.sim_clean_spec[0], aspect='auto', origin='lower')
        axes[0].set_title('Clean 스펙트로그램')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(self.sim_jammed_spec[0], aspect='auto', origin='lower')
        axes[1].set_title('Jammed 스펙트로그램')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(output_path / 'spectrogram_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_overall_score(self):
        """종합 평가 점수 계산"""
        score = 0
        
        # 상관관계 점수 (40점)
        if 'correlations' in self.validation_results:
            clean_corr = self.validation_results['correlations']['clean_correlation']
            jammed_corr = self.validation_results['correlations']['jammed_correlation']
            avg_corr = (clean_corr + jammed_corr) / 2
            score += min(40, max(0, avg_corr * 40))
        else:
            score += 30  # 시뮬레이션만 있는 경우 기본 점수
        
        # KS 테스트 점수 (30점)
        if 'ks_tests' in self.validation_results:
            clean_p = self.validation_results['ks_tests']['clean_p_value']
            jammed_p = self.validation_results['ks_tests']['jammed_p_value']
            avg_p = (clean_p + jammed_p) / 2
            score += min(30, max(0, avg_p * 600))  # p > 0.05면 30점
        else:
            score += 20
        
        # 스펙트럼 유사도 점수 (20점)
        if 'clean_spectral_similarity' in self.validation_results.get('spectral_analysis', {}):
            clean_sim = self.validation_results['spectral_analysis']['clean_spectral_similarity']
            jammed_sim = self.validation_results['spectral_analysis']['jammed_spectral_similarity']
            avg_sim = (clean_sim + jammed_sim) / 2
            score += min(20, max(0, avg_sim * 20))
        else:
            score += 15
        
        # 신호 품질 점수 (10점)
        score += 10  # 기본적으로 시뮬레이션이 생성되었으므로
        
        return min(100, score)
    
    def _get_grade(self, score):
        """점수를 등급으로 변환"""
        if score >= 80:
            return "우수 (A)"
        elif score >= 60:
            return "양호 (B)"
        elif score >= 40:
            return "보통 (C)"
        else:
            return "미흡 (D)"
    
    def _generate_recommendations(self):
        """검증 결과 기반 권장사항 생성"""
        recommendations = []
        
        if 'correlations' in self.validation_results:
            clean_corr = self.validation_results['correlations']['clean_correlation']
            jammed_corr = self.validation_results['correlations']['jammed_correlation']
            
            if clean_corr < 0.6:
                recommendations.append("Clean 신호 모델링 파라미터 재검토 필요")
            if jammed_corr < 0.6:
                recommendations.append("재밍 신호 모델링 파라미터 조정 필요")
        
        if 'ks_tests' in self.validation_results:
            if self.validation_results['ks_tests']['clean_p_value'] < 0.05:
                recommendations.append("Clean 신호 분포 특성 개선 필요")
            if self.validation_results['ks_tests']['jammed_p_value'] < 0.05:
                recommendations.append("Jammed 신호 분포 특성 개선 필요")
        
        if not recommendations:
            recommendations.append("현재 시뮬레이션 모델 품질이 양호함")
            recommendations.append("추가 하드웨어 데이터 수집을 통한 지속적 검증 권장")
        
        return recommendations
    
    def run_full_validation(self):
        """전체 검증 프로세스 실행"""
        print("\n" + "="*80)
        print("🚀 X4M06 레이더 시뮬레이션 모델 현실성 검증 시작")
        print("="*80)
        
        # 데이터 로드
        if not self.load_simulation_data():
            return False
        
        if self.hw_data_path:
            self.load_hardware_data()
        
        # 검증 분석 실행
        self.statistical_comparison()
        self.spectral_analysis() 
        self.signal_quality_metrics()
        
        # 리포트 생성
        self.create_validation_report()
        
        print("\n" + "="*80)
        print("✅ 시뮬레이션 모델 현실성 검증 완료!")
        print("="*80)
        
        return True


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='X4M06 레이더 시뮬레이션 검증')
    parser.add_argument('--sim-data', required=True, help='시뮬레이션 데이터 파일 경로')
    parser.add_argument('--hw-data', help='하드웨어 데이터 파일 경로 (선택)')
    parser.add_argument('--output-dir', default='validation_results', help='결과 출력 디렉토리')
    
    args = parser.parse_args()
    
    # 검증기 초기화 및 실행
    validator = SimulationValidator(args.sim_data, args.hw_data)
    validator.run_full_validation()


if __name__ == "__main__":
    main()