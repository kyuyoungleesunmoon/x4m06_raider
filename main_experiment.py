#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
X4M06 레이더 재밍 신호 복원 연구 - 메인 실험 스크립트
차량 레이더 재밍 환경 시뮬레이션 및 데이터셋 구축 통합 실행

사용법:
    python main_experiment.py --mode simulation --samples 1000
    python main_experiment.py --mode hardware --device COM3
    python main_experiment.py --mode analysis --dataset synthetic_dataset/radar_jamming_dataset_1000.h5
"""

import argparse
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np

# 로컬 모듈 임포트
from jamming_simulator import FMCWRadarSimulator, SpectrogramGenerator, DatasetGenerator
from x4m06_data_collector import X4M06DataCollector, ExperimentController
from data_analyzer import RadarDataAnalyzer, DataPreprocessor


class ExperimentManager:
    """실험 통합 관리 클래스"""
    
    def __init__(self, base_output_dir="experiment_results"):
        """
        초기화
        Args:
            base_output_dir (str): 기본 출력 디렉토리
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True, parents=True)
        
        # 실험 세션 정보
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_output_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        # 실험 로그
        self.experiment_log = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'experiments': []
        }
        
        print(f"실험 세션 시작: {self.session_id}")
        print(f"결과 저장 경로: {self.session_dir}")
    
    def run_simulation_experiment(self, config=None):
        """
        시뮬레이션 기반 실험 실행
        
        Args:
            config (dict): 실험 설정
        
        Returns:
            dict: 실험 결과 정보
        """
        print("\n" + "="*50)
        print("시뮬레이션 기반 실험 시작")
        print("="*50)
        
        if config is None:
            config = self._get_default_simulation_config()
        
        # 출력 디렉토리 설정
        sim_output_dir = self.session_dir / "simulation"
        sim_output_dir.mkdir(exist_ok=True)
        
        try:
            # 1. 레이더 시뮬레이터 초기화
            print("\n1. 레이더 시뮬레이터 초기화...")
            radar_sim = FMCWRadarSimulator(config['radar_config'])
            spec_gen = SpectrogramGenerator(config['stft_params'])
            dataset_gen = DatasetGenerator(radar_sim, spec_gen, str(sim_output_dir))
            
            # 2. 샘플 시각화
            print("\n2. 샘플 시각화 생성...")
            dataset_gen.visualize_samples(num_visualize=config.get('num_visualize', 5))
            
            # 3. 데이터셋 생성
            print(f"\n3. 합성 데이터셋 생성 ({config['num_samples']}개 샘플)...")
            dataset_file = dataset_gen.generate_dataset(
                config['num_samples'], 
                save_format=config.get('save_format', 'hdf5')
            )
            
            # 4. 실험 결과 기록
            experiment_result = {
                'type': 'simulation',
                'config': config,
                'dataset_file': str(dataset_file),
                'output_dir': str(sim_output_dir),
                'status': 'success',
                'end_time': datetime.now().isoformat()
            }
            
            self.experiment_log['experiments'].append(experiment_result)
            
            print(f"\n✅ 시뮬레이션 실험 완료!")
            print(f"데이터셋: {dataset_file}")
            
            return experiment_result
            
        except Exception as e:
            error_result = {
                'type': 'simulation',
                'config': config,
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            }
            self.experiment_log['experiments'].append(error_result)
            
            print(f"\n❌ 시뮬레이션 실험 실패: {e}")
            return error_result
    
    def run_hardware_experiment(self, device_name, config=None):
        """
        하드웨어 기반 실험 실행
        
        Args:
            device_name (str): 하드웨어 장치 이름 (COM 포트)
            config (dict): 실험 설정
        
        Returns:
            dict: 실험 결과 정보
        """
        print("\n" + "="*50)
        print(f"하드웨어 기반 실험 시작: {device_name}")
        print("="*50)
        
        if config is None:
            config = self._get_default_hardware_config()
        
        # 출력 디렉토리 설정
        hw_output_dir = self.session_dir / "hardware"
        hw_output_dir.mkdir(exist_ok=True)
        
        try:
            # 1. 데이터 수집기 초기화
            print(f"\n1. X4M06 데이터 수집기 초기화: {device_name}")
            collector = X4M06DataCollector(device_name, config.get('radar_config'))
            experiment_ctrl = ExperimentController(str(hw_output_dir))
            
            # 2. 베이스라인 실험
            print(f"\n2. 베이스라인 실험 실행...")
            baseline_file = experiment_ctrl.run_baseline_experiment(
                collector, 
                num_frames=config.get('baseline_frames', 1000)
            )
            
            # 3. 시나리오 기반 실험
            if config.get('run_scenarios', True):
                print(f"\n3. 시나리오 기반 실험 실행...")
                scenarios = config.get('scenarios', [
                    {'name': 'close_range', 'description': '근거리 탐지'},
                    {'name': 'medium_range', 'description': '중거리 탐지'},
                    {'name': 'long_range', 'description': '원거리 탐지'}
                ])
                
                scenario_results = experiment_ctrl.run_multi_radar_simulation(
                    collector, scenarios
                )
            else:
                scenario_results = []
            
            # 4. 메타데이터 저장
            experiment_ctrl.save_metadata()
            
            # 5. 실험 결과 기록
            experiment_result = {
                'type': 'hardware',
                'device_name': device_name,
                'config': config,
                'baseline_file': baseline_file,
                'scenario_files': scenario_results,
                'output_dir': str(hw_output_dir),
                'status': 'success',
                'end_time': datetime.now().isoformat()
            }
            
            self.experiment_log['experiments'].append(experiment_result)
            
            print(f"\n✅ 하드웨어 실험 완료!")
            print(f"베이스라인 데이터: {baseline_file}")
            print(f"시나리오 데이터: {len(scenario_results)}개 파일")
            
            return experiment_result
            
        except Exception as e:
            error_result = {
                'type': 'hardware',
                'device_name': device_name,
                'config': config,
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            }
            self.experiment_log['experiments'].append(error_result)
            
            print(f"\n❌ 하드웨어 실험 실패: {e}")
            return error_result
    
    def run_analysis_experiment(self, dataset_path, config=None):
        """
        데이터 분석 실험 실행
        
        Args:
            dataset_path (str): 분석할 데이터셋 경로
            config (dict): 분석 설정
        
        Returns:
            dict: 분석 결과 정보
        """
        print("\n" + "="*50)
        print(f"데이터 분석 실험 시작: {dataset_path}")
        print("="*50)
        
        if config is None:
            config = self._get_default_analysis_config()
        
        # 출력 디렉토리 설정
        analysis_output_dir = self.session_dir / "analysis"
        analysis_output_dir.mkdir(exist_ok=True)
        
        try:
            # 1. 데이터 분석기 초기화
            print("\n1. 데이터 분석기 초기화...")
            analyzer = RadarDataAnalyzer(str(analysis_output_dir))
            
            # 2. 데이터셋 로드
            print("\n2. 데이터셋 로드...")
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
            
            data = analyzer.load_dataset(dataset_path, config.get('dataset_type', 'hdf5'))
            
            # 3. 신호 통계 분석
            print("\n3. 신호 통계 분석...")
            if 'clean_signals' in data:
                clean_stats = analyzer.analyze_signal_statistics(data['clean_signals'], "Clean")
            
            if 'jammed_signals' in data:
                jammed_stats = analyzer.analyze_signal_statistics(data['jammed_signals'], "Jammed")
            
            # 4. 주파수 분석
            print("\n4. 주파수 스펙트럼 분석...")
            sampling_rate = config.get('sampling_rate', 1e6)
            
            if 'clean_signals' in data:
                clean_freq = analyzer.analyze_frequency_spectrum(
                    data['clean_signals'], sampling_rate, "Clean"
                )
            
            if 'jammed_signals' in data:
                jammed_freq = analyzer.analyze_frequency_spectrum(
                    data['jammed_signals'], sampling_rate, "Jammed"
                )
            
            # 5. 비교 분석
            if 'clean_signals' in data and 'jammed_signals' in data:
                print("\n5. 깨끗한 vs 재밍 신호 비교 분석...")
                comparison = analyzer.compare_clean_vs_jammed(
                    data['clean_signals'], data['jammed_signals'], "Comparison"
                )
            
            # 6. 전처리 테스트 (선택적)
            if config.get('test_preprocessing', True):
                print("\n6. 전처리 파이프라인 테스트...")
                preprocessor = DataPreprocessor(config.get('preprocess_config'))
                
                # 작은 샘플로 테스트
                test_size = min(config.get('preprocess_test_size', 100), len(data['clean_signals']))
                clean_test = data['clean_signals'][:test_size]
                jammed_test = data['jammed_signals'][:test_size]
                
                input_data, target_data, preprocess_info = preprocessor.preprocess_for_training(
                    clean_test, jammed_test, sampling_rate
                )
                
                print(f"전처리된 데이터 형태: {input_data.shape} -> {target_data.shape}")
                
                # 전처리된 데이터 샘플 저장
                preprocessed_file = analysis_output_dir / "preprocessed_sample.npz"
                np.savez_compressed(
                    preprocessed_file,
                    input_data=input_data[:10],  # 10개 샘플만 저장
                    target_data=target_data[:10],
                    preprocess_info=json.dumps(preprocess_info)
                )
                print(f"전처리 샘플 저장: {preprocessed_file}")
            
            # 7. 종합 보고서 생성
            print("\n7. 분석 보고서 생성...")
            analyzer.generate_analysis_report()
            
            # 8. 실험 결과 기록
            experiment_result = {
                'type': 'analysis',
                'dataset_path': dataset_path,
                'config': config,
                'output_dir': str(analysis_output_dir),
                'data_shape': {k: list(v.shape) if hasattr(v, 'shape') else str(type(v)) 
                              for k, v in data.items()},
                'status': 'success',
                'end_time': datetime.now().isoformat()
            }
            
            self.experiment_log['experiments'].append(experiment_result)
            
            print(f"\n✅ 데이터 분석 완료!")
            print(f"분석 결과: {analysis_output_dir}")
            
            return experiment_result
            
        except Exception as e:
            error_result = {
                'type': 'analysis',
                'dataset_path': dataset_path,
                'config': config,
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            }
            self.experiment_log['experiments'].append(error_result)
            
            print(f"\n❌ 데이터 분석 실패: {e}")
            return error_result
    
    def save_experiment_log(self):
        """실험 로그 저장"""
        self.experiment_log['end_time'] = datetime.now().isoformat()
        
        log_file = self.session_dir / "experiment_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n실험 로그 저장: {log_file}")
        
        # 요약 리포트도 생성
        self._generate_summary_report()
    
    def _generate_summary_report(self):
        """실험 요약 리포트 생성"""
        summary_file = self.session_dir / "experiment_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"=== 실험 세션 요약 리포트 ===\n")
            f.write(f"세션 ID: {self.session_id}\n")
            f.write(f"시작 시간: {self.experiment_log['start_time']}\n")
            f.write(f"종료 시간: {self.experiment_log.get('end_time', 'N/A')}\n")
            f.write(f"총 실험 수: {len(self.experiment_log['experiments'])}\n\n")
            
            # 실험별 요약
            for i, exp in enumerate(self.experiment_log['experiments'], 1):
                f.write(f"{i}. {exp['type'].upper()} 실험\n")
                f.write(f"   상태: {exp['status']}\n")
                
                if exp['status'] == 'success':
                    if exp['type'] == 'simulation':
                        f.write(f"   데이터셋: {exp.get('dataset_file', 'N/A')}\n")
                        f.write(f"   샘플 수: {exp.get('config', {}).get('num_samples', 'N/A')}\n")
                    elif exp['type'] == 'hardware':
                        f.write(f"   장치: {exp.get('device_name', 'N/A')}\n")
                        f.write(f"   베이스라인: {exp.get('baseline_file', 'N/A')}\n")
                    elif exp['type'] == 'analysis':
                        f.write(f"   데이터셋: {exp.get('dataset_path', 'N/A')}\n")
                        f.write(f"   출력 디렉토리: {exp.get('output_dir', 'N/A')}\n")
                else:
                    f.write(f"   오류: {exp.get('error', 'Unknown error')}\n")
                
                f.write(f"   종료 시간: {exp.get('end_time', 'N/A')}\n\n")
        
        print(f"실험 요약 리포트 저장: {summary_file}")
    
    def _get_default_simulation_config(self):
        """기본 시뮬레이션 설정 - 1m 이내 실내 실험 최적화"""
        return {
            'num_samples': 1000,
            'num_visualize': 5,
            'save_format': 'hdf5',
            'radar_config': {
                'center_freq': 8.748e9,      # 8.748 GHz (X4M06 중심 주파수)
                'bandwidth': 1.4e9,          # 1.4 GHz (거리 분해능 ~10.7cm)
                'chirp_duration': 50e-6,     # 50μs (현실적 지속시간)
                'prf': 1000,                # 1 kHz (업데이트 레이트)
                'sampling_rate': 10e6,       # 10 MHz (충분한 해상도)
                'target_range': [0.2, 2.0],  # 20cm-2m (X4M06 최소거리 고려)
                'target_velocity': [-2, 2],  # ±2 m/s (실내 이동 속도)
                'target_rcs': [0.01, 1.0],   # 작은 물체 대응 (책, 의자 등)
                'num_jammers': [1, 3],       # 실내 환경 맞춤
                'jammer_power_ratio': [0.5, 2.0],
                'freq_offset_range': [-0.05e9, 0.05e9],  # 범위 축소
                'time_offset_range': [0, 40e-6],    # 40μs 이내 (처프 지속시간 내)
                'snr_db': [10, 20],          # 실내 환경 맞춤
            },
            'stft_params': {
                'nperseg': 128,              # 더 세밀한 시간 분해능
                'noverlap': 64,              # 50% 중첩
                'nfft': 256,                 # FFT 포인트 (주파수 분해능)
                'window': 'hann',
            }
        }
    
    def _get_default_hardware_config(self):
        """기본 하드웨어 설정 - 1m 이내 실내 실험"""
        return {
            'baseline_frames': 1000,
            'run_scenarios': True,
            'scenarios': [
                {'name': 'close_range_20cm', 'description': '초근거리 (15-25cm)'},
                {'name': 'close_range_40cm', 'description': '근거리 (35-45cm)'},
                {'name': 'close_range_60cm', 'description': '중근거리 (55-65cm)'},
                {'name': 'close_range_80cm', 'description': '원근거리 (75-85cm)'},
            ],
            'radar_config': {
                'dac_min': 900,
                'dac_max': 1150,
                'iterations': 16,
                'pulses_per_step': 26,
                'frame_area_start': 0.5,
                'frame_area_end': 5.0,
                'frame_area_offset': 0.18,
                'fps': 20,
                'tx_power': 2,
                'center_frequency': 3,
                'prf_div': 16,
                'downconversion': 1,
            }
        }
    
    def _get_default_analysis_config(self):
        """기본 분석 설정"""
        return {
            'dataset_type': 'hdf5',
            'sampling_rate': 1e6,
            'test_preprocessing': True,
            'preprocess_test_size': 100,
            'preprocess_config': {
                'normalization_method': 'minmax',
                'spectrogram_params': {
                    'nperseg': 256,
                    'noverlap': 128,
                    'nfft': 512,
                    'window': 'hann'
                },
                'image_size': (256, 256),
            }
        }


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="X4M06 레이더 재밍 신호 복원 연구 - 통합 실험 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 시뮬레이션 기반 데이터셋 생성 (1000개 샘플)
  python main_experiment.py --mode simulation --samples 1000
  
  # X4M06 하드웨어를 이용한 실제 데이터 수집
  python main_experiment.py --mode hardware --device COM3
  
  # 생성된 데이터셋 분석
  python main_experiment.py --mode analysis --dataset experiment_results/session_*/simulation/radar_jamming_dataset_*.h5
  
  # 모든 실험을 순차적으로 실행
  python main_experiment.py --mode all --samples 5000 --device COM3
        """
    )
    
    # 기본 인자
    parser.add_argument(
        '--mode', 
        choices=['simulation', 'hardware', 'analysis', 'all'],
        required=True,
        help='실험 모드 선택'
    )
    
    parser.add_argument(
        '--output-dir',
        default='experiment_results',
        help='결과 출력 디렉토리 (기본값: experiment_results)'
    )
    
    # 시뮬레이션 관련 인자
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='생성할 합성 데이터 샘플 수 (기본값: 1000)'
    )
    
    # 하드웨어 관련 인자
    parser.add_argument(
        '--device',
        default='COM3',
        help='X4M06 장치 이름/COM 포트 (기본값: COM3)'
    )
    
    parser.add_argument(
        '--frames',
        type=int,
        default=1000,
        help='하드웨어에서 수집할 프레임 수 (기본값: 1000)'
    )
    
    # 분석 관련 인자
    parser.add_argument(
        '--dataset',
        help='분석할 데이터셋 파일 경로'
    )
    
    parser.add_argument(
        '--config',
        help='사용자 정의 설정 파일 (JSON 형식)'
    )
    
    args = parser.parse_args()
    
    print("X4M06 레이더 재밍 신호 복원 연구")
    print("=" * 60)
    print(f"실행 모드: {args.mode}")
    print(f"출력 디렉토리: {args.output_dir}")
    
    # 실험 매니저 초기화
    experiment_manager = ExperimentManager(args.output_dir)
    
    try:
        # 사용자 정의 설정 로드
        custom_config = None
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r', encoding='utf-8') as f:
                custom_config = json.load(f)
            print(f"사용자 정의 설정 로드: {args.config}")
        
        if args.mode == 'simulation':
            # 시뮬레이션 실험
            config = custom_config if custom_config else {}
            
            # 기본 config와 병합
            default_config = experiment_manager._get_default_simulation_config()
            if not custom_config:
                config = default_config
            config['num_samples'] = args.samples
            
            result = experiment_manager.run_simulation_experiment(config)
            
        elif args.mode == 'hardware':
            # 하드웨어 실험
            config = custom_config or {}
            config.update({'baseline_frames': args.frames})
            
            result = experiment_manager.run_hardware_experiment(args.device, config)
            
        elif args.mode == 'analysis':
            # 데이터 분석 실험
            if not args.dataset:
                print("❌ 분석 모드에서는 --dataset 인자가 필요합니다.")
                sys.exit(1)
            
            config = custom_config or {}
            result = experiment_manager.run_analysis_experiment(args.dataset, config)
            
        elif args.mode == 'all':
            # 모든 실험 순차 실행
            print("\n🚀 전체 실험 파이프라인 시작!")
            
            # 1. 시뮬레이션 실험
            sim_config = custom_config.get('simulation', {}) if custom_config else {}
            sim_config.update({'num_samples': args.samples})
            
            sim_result = experiment_manager.run_simulation_experiment(sim_config)
            
            # 2. 하드웨어 실험 (선택적)
            hw_result = None
            try:
                hw_config = custom_config.get('hardware', {}) if custom_config else {}
                hw_config.update({'baseline_frames': args.frames})
                
                hw_result = experiment_manager.run_hardware_experiment(args.device, hw_config)
            except Exception as e:
                print(f"⚠️  하드웨어 실험 건너뜀: {e}")
            
            # 3. 분석 실험 (시뮬레이션 데이터 사용)
            if sim_result['status'] == 'success':
                analysis_config = custom_config.get('analysis', {}) if custom_config else {}
                analysis_result = experiment_manager.run_analysis_experiment(
                    sim_result['dataset_file'], analysis_config
                )
            
            print("\n🎉 전체 실험 파이프라인 완료!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  사용자에 의해 실험이 중단되었습니다.")
    
    except Exception as e:
        print(f"\n❌ 실험 중 예기치 않은 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 실험 로그 저장
        experiment_manager.save_experiment_log()
        print(f"\n📁 실험 결과가 다음 위치에 저장되었습니다:")
        print(f"   {experiment_manager.session_dir}")


if __name__ == "__main__":
    main()