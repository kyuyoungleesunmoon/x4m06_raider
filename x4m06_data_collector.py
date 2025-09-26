#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
X4M06 레이더 실험 데이터 수집기
실제 X4M06 하드웨어를 이용한 레이더 신호 수집 및 재밍 실험

연구목적: 합성 데이터와 실제 데이터 간의 검증 및 실환경 재밍 패턴 분석
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py
import json
from datetime import datetime
from time import sleep
import threading
import queue
from scipy import signal
from tqdm import tqdm

# 현재 디렉토리에서 pymoduleconnector 모듈 검색
sys.path.append(os.path.join(os.path.dirname(__file__), 
                            '..', 'Users', 'User', 'Downloads', 'X4M06_Package', 
                            'ModuleConnector', 'ModuleConnector-win32_win64-1', 'python36-win64'))

try:
    from pymoduleconnector import ModuleConnector, DataType
    from pymoduleconnector.ids import *
    PYMODULECONNECTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pymoduleconnector not available: {e}")
    print("시뮬레이션 모드로만 동작 가능합니다.")
    PYMODULECONNECTOR_AVAILABLE = False
    # 더미 상수 정의
    XTID_SSIC_ITEMNUMBER = 0x00
    XTID_SSIC_ORDERCODE = 0x01
    XTID_SSIC_FIRMWAREID = 0x02
    XTID_SSIC_VERSION = 0x03
    XTID_SSIC_BUILD = 0x04
    XTID_SSIC_SERIALNUMBER = 0x05


class X4M06DataCollector:
    """X4M06 레이더 데이터 수집 클래스"""
    
    def __init__(self, device_name="COM3", config=None):
        """
        초기화
        Args:
            device_name (str): 장치 이름 (COM 포트)
            config (dict): 레이더 설정
        """
        self.device_name = device_name
        self.config = config if config else self.get_default_config()
        self.mc = None
        self.xep = None
        self.is_connected = False
        self.is_streaming = False
        
        # 데이터 수집용 큐
        self.data_queue = queue.Queue(maxsize=1000)
        self.collection_thread = None
        
    def get_default_config(self):
        """기본 레이더 설정"""
        return {
            'dac_min': 900,
            'dac_max': 1150,
            'iterations': 16,
            'pulses_per_step': 26,
            'frame_area_start': 0.5,    # 미터
            'frame_area_end': 5.0,      # 미터
            'frame_area_offset': 0.18,
            'fps': 20,
            'tx_power': 2,
            'center_frequency': 3,
            'prf_div': 16,
            'downconversion': 1,        # 베이스밴드 모드
        }
    
    def connect(self):
        """레이더 모듈 연결"""
        if not PYMODULECONNECTOR_AVAILABLE:
            print("Error: pymoduleconnector가 사용할 수 없습니다.")
            return False
        
        try:
            print(f"X4M06 연결 시도: {self.device_name}")
            
            # 모듈 연결
            self.mc = ModuleConnector(self.device_name)
            
            # X4M300 모드로 설정 시도 (X4M06과 호환)
            try:
                app = self.mc.get_x4m300()
                app.set_sensor_mode(0x13, 0)  # 프로파일 정지
                app.set_sensor_mode(0x12, 0)  # 수동 모드
            except RuntimeError:
                print("X4M300 모드 설정 실패, XEP 모드로 진행")
            
            # XEP 인터페이스 획득
            self.xep = self.mc.get_xep()
            
            # 통신 테스트
            pong = self.xep.ping()
            print(f"Ping 응답: {hex(pong)}")
            
            # 시스템 정보 출력
            self._print_system_info()
            
            # 레이더 초기화 및 설정
            self._initialize_radar()
            
            self.is_connected = True
            print("X4M06 연결 성공!")
            return True
            
        except Exception as e:
            print(f"연결 실패: {e}")
            return False
    
    def disconnect(self):
        """레이더 모듈 연결 해제"""
        try:
            if self.is_streaming:
                self.stop_streaming()
            
            if self.xep:
                self.xep.x4driver_set_fps(0)  # 스트리밍 정지
                self.xep.module_reset()
            
            if self.mc:
                self.mc.close()
            
            self.is_connected = False
            print("X4M06 연결 해제 완료")
            
        except Exception as e:
            print(f"연결 해제 중 오류: {e}")
    
    def _print_system_info(self):
        """시스템 정보 출력"""
        try:
            print("\n=== X4M06 시스템 정보 ===")
            print(f"Item Number: {self.xep.get_system_info(XTID_SSIC_ITEMNUMBER)}")
            print(f"Order Code: {self.xep.get_system_info(XTID_SSIC_ORDERCODE)}")
            print(f"Firmware ID: {self.xep.get_system_info(XTID_SSIC_FIRMWAREID)}")
            print(f"Version: {self.xep.get_system_info(XTID_SSIC_VERSION)}")
            print(f"Build: {self.xep.get_system_info(XTID_SSIC_BUILD)}")
            print(f"Serial Number: {self.xep.get_system_info(XTID_SSIC_SERIALNUMBER)}")
            print("========================\n")
        except Exception as e:
            print(f"시스템 정보 읽기 실패: {e}")
    
    def _initialize_radar(self):
        """레이더 초기화 및 파라미터 설정"""
        try:
            # 드라이버 초기화
            self.xep.x4driver_init()
            
            # 기본 파라미터 설정
            self.xep.x4driver_set_enable(1)
            self.xep.x4driver_set_dac_min(self.config['dac_min'])
            self.xep.x4driver_set_dac_max(self.config['dac_max'])
            self.xep.x4driver_set_iterations(self.config['iterations'])
            self.xep.x4driver_set_pulses_per_step(self.config['pulses_per_step'])
            
            # 프레임 영역 설정
            self.xep.x4driver_set_frame_area_offset(self.config['frame_area_offset'])
            self.xep.x4driver_set_frame_area(
                self.config['frame_area_start'], 
                self.config['frame_area_end']
            )
            
            # RF 파라미터 설정
            self.xep.x4driver_set_tx_power(self.config['tx_power'])
            self.xep.x4driver_set_tx_center_frequency(self.config['center_frequency'])
            self.xep.x4driver_set_prf_div(self.config['prf_div'])
            self.xep.x4driver_set_downconversion(self.config['downconversion'])
            
            print("레이더 초기화 완료")
            
            # 설정값 확인
            frame_area = self.xep.x4driver_get_frame_area()
            print(f"설정된 프레임 영역: {frame_area.start}m - {frame_area.end}m")
            print(f"FPS: {self.xep.x4driver_get_fps()}")
            
        except Exception as e:
            print(f"레이더 초기화 실패: {e}")
            raise
    
    def start_streaming(self, fps=None):
        """데이터 스트리밍 시작"""
        if not self.is_connected:
            print("레이더가 연결되지 않았습니다.")
            return False
        
        try:
            fps = fps if fps else self.config['fps']
            self.xep.x4driver_set_fps(fps)
            self.is_streaming = True
            
            # 버퍼 클리어
            self._clear_buffer()
            
            print(f"데이터 스트리밍 시작 (FPS: {fps})")
            return True
            
        except Exception as e:
            print(f"스트리밍 시작 실패: {e}")
            return False
    
    def stop_streaming(self):
        """데이터 스트리밍 정지"""
        try:
            self.xep.x4driver_set_fps(0)
            self.is_streaming = False
            
            if self.collection_thread and self.collection_thread.is_alive():
                self.collection_thread.join(timeout=2.0)
            
            print("데이터 스트리밍 정지")
            
        except Exception as e:
            print(f"스트리밍 정지 실패: {e}")
    
    def _clear_buffer(self):
        """프레임 버퍼 클리어"""
        try:
            while self.xep.peek_message_data_float():
                self.xep.read_message_data_float()
        except Exception as e:
            print(f"버퍼 클리어 실패: {e}")
    
    def read_frame(self):
        """단일 프레임 데이터 읽기"""
        if not self.is_streaming:
            print("스트리밍이 시작되지 않았습니다.")
            return None
        
        try:
            if self.xep.peek_message_data_float():
                d = self.xep.read_message_data_float()
                frame = np.array(d.data, dtype=np.float32)
                
                # 베이스밴드 모드인 경우 복소수 변환
                if self.config['downconversion']:
                    n = len(frame)
                    if n % 2 == 0:
                        frame = frame[:n//2] + 1j * frame[n//2:]
                    else:
                        print("Warning: 홀수 길이 프레임, 복소수 변환 실패")
                
                return frame
            else:
                return None
                
        except Exception as e:
            print(f"프레임 읽기 실패: {e}")
            return None
    
    def collect_data_batch(self, num_frames, timeout=30):
        """배치 데이터 수집"""
        if not self.is_streaming:
            print("스트리밍을 먼저 시작하세요.")
            return None
        
        collected_frames = []
        start_time = datetime.now()
        
        print(f"{num_frames}개 프레임 수집 시작...")
        
        with tqdm(total=num_frames, desc="데이터 수집") as pbar:
            while len(collected_frames) < num_frames:
                frame = self.read_frame()
                if frame is not None:
                    collected_frames.append(frame)
                    pbar.update(1)
                else:
                    sleep(0.001)  # 짧은 대기
                
                # 타임아웃 체크
                if (datetime.now() - start_time).total_seconds() > timeout:
                    print(f"타임아웃: {len(collected_frames)}개 프레임만 수집됨")
                    break
        
        if collected_frames:
            return np.array(collected_frames)
        else:
            return None


class ExperimentController:
    """실험 제어 및 데이터 관리 클래스"""
    
    def __init__(self, output_dir="experiment_data"):
        """
        초기화
        Args:
            output_dir (str): 실험 데이터 출력 디렉토리
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 실험 메타데이터
        self.experiment_metadata = {
            'creation_date': datetime.now().isoformat(),
            'experiments': []
        }
    
    def run_baseline_experiment(self, collector, num_frames=1000):
        """
        베이스라인 실험: 재밍이 없는 환경에서의 데이터 수집
        
        Args:
            collector (X4M06DataCollector): 데이터 수집기
            num_frames (int): 수집할 프레임 수
        
        Returns:
            str: 저장된 데이터 파일 경로
        """
        print("\n=== 베이스라인 실험 시작 ===")
        
        if not collector.connect():
            print("레이더 연결 실패")
            return None
        
        try:
            # 스트리밍 시작
            if not collector.start_streaming():
                print("스트리밍 시작 실패")
                return None
            
            # 데이터 수집
            data = collector.collect_data_batch(num_frames)
            
            if data is None:
                print("데이터 수집 실패")
                return None
            
            # 데이터 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f'baseline_data_{timestamp}.h5')
            
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('radar_data', data=data)
                f.create_dataset('config', data=json.dumps(collector.config))
                f.attrs['experiment_type'] = 'baseline'
                f.attrs['num_frames'] = num_frames
                f.attrs['timestamp'] = timestamp
            
            # 메타데이터 업데이트
            exp_metadata = {
                'experiment_type': 'baseline',
                'timestamp': timestamp,
                'file_path': output_file,
                'num_frames': num_frames,
                'config': collector.config
            }
            self.experiment_metadata['experiments'].append(exp_metadata)
            
            print(f"베이스라인 데이터 저장: {output_file}")
            
            # 간단한 분석
            self._analyze_baseline_data(data, output_file.replace('.h5', '_analysis.png'))
            
            return output_file
            
        finally:
            collector.disconnect()
    
    def run_multi_radar_simulation(self, collector, scenarios=None):
        """
        다중 레이더 시뮬레이션 실험
        
        Args:
            collector (X4M06DataCollector): 데이터 수집기
            scenarios (list): 테스트 시나리오 리스트
        """
        if scenarios is None:
            scenarios = [
                {'name': 'single_target', 'description': '단일 목표물'},
                {'name': 'multiple_targets', 'description': '다중 목표물'},
                {'name': 'moving_target', 'description': '이동 목표물'},
            ]
        
        print("\n=== 다중 레이더 시뮬레이션 실험 ===")
        
        results = []
        
        for scenario in scenarios:
            print(f"\n시나리오: {scenario['name']} ({scenario['description']})")
            
            # 각 시나리오별 데이터 수집 로직 구현
            # (실제 환경에서는 물리적 목표물 배치 변경 필요)
            
            result = self._run_scenario(collector, scenario)
            if result:
                results.append(result)
        
        return results
    
    def _run_scenario(self, collector, scenario):
        """개별 시나리오 실행"""
        try:
            if not collector.connect():
                return None
            
            collector.start_streaming()
            
            # 시나리오별 특별 설정 (예: 프레임 영역 변경 등)
            if scenario['name'] == 'moving_target':
                # 더 넓은 범위 설정
                collector.xep.x4driver_set_frame_area(0.5, 8.0)
            
            # 데이터 수집
            data = collector.collect_data_batch(500)  # 시나리오별로 적은 프레임
            
            if data is not None:
                # 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(
                    self.output_dir, 
                    f'scenario_{scenario["name"]}_{timestamp}.h5'
                )
                
                with h5py.File(output_file, 'w') as f:
                    f.create_dataset('radar_data', data=data)
                    f.attrs['scenario_name'] = scenario['name']
                    f.attrs['scenario_description'] = scenario['description']
                    f.attrs['timestamp'] = timestamp
                
                print(f"시나리오 데이터 저장: {output_file}")
                return output_file
            
        except Exception as e:
            print(f"시나리오 실행 실패: {e}")
        
        finally:
            collector.disconnect()
        
        return None
    
    def _analyze_baseline_data(self, data, output_path):
        """베이스라인 데이터 분석 및 시각화"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('베이스라인 데이터 분석', fontsize=16)
            
            # 시간 영역 신호
            axes[0, 0].plot(np.abs(data[0]))
            axes[0, 0].set_title('첫 번째 프레임 (크기)')
            axes[0, 0].set_xlabel('샘플')
            axes[0, 0].set_ylabel('크기')
            
            # 위상 정보
            if np.iscomplexobj(data):
                axes[0, 1].plot(np.angle(data[0]))
                axes[0, 1].set_title('첫 번째 프레임 (위상)')
                axes[0, 1].set_xlabel('샘플')
                axes[0, 1].set_ylabel('위상 (라디안)')
            
            # 평균 신호
            mean_signal = np.mean(np.abs(data), axis=0)
            axes[1, 0].plot(mean_signal)
            axes[1, 0].set_title('평균 신호 크기')
            axes[1, 0].set_xlabel('샘플')
            axes[1, 0].set_ylabel('평균 크기')
            
            # 신호 변화량
            signal_variation = np.std(np.abs(data), axis=0)
            axes[1, 1].plot(signal_variation)
            axes[1, 1].set_title('신호 변화량 (표준편차)')
            axes[1, 1].set_xlabel('샘플')
            axes[1, 1].set_ylabel('표준편차')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"분석 결과 저장: {output_path}")
            
        except Exception as e:
            print(f"데이터 분석 실패: {e}")
    
    def save_metadata(self):
        """실험 메타데이터 저장"""
        metadata_file = os.path.join(self.output_dir, 'experiment_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"실험 메타데이터 저장: {metadata_file}")


def main():
    """메인 실행 함수"""
    print("X4M06 레이더 실험 시작")
    print("=" * 50)
    
    # 사용 가능한 COM 포트 확인 (Windows)
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    print("사용 가능한 COM 포트:")
    for port in ports:
        print(f"  {port.device}: {port.description}")
    
    if not ports:
        print("사용 가능한 COM 포트가 없습니다.")
        print("시뮬레이션 모드로 진행합니다.")
        return
    
    # 데이터 수집기 초기화
    device_name = "COM3"  # 실제 연결된 포트로 변경
    collector = X4M06DataCollector(device_name)
    
    # 실험 컨트롤러 초기화
    experiment_controller = ExperimentController()
    
    try:
        # 베이스라인 실험 실행
        baseline_file = experiment_controller.run_baseline_experiment(
            collector, num_frames=1000
        )
        
        if baseline_file:
            print(f"\n베이스라인 실험 완료: {baseline_file}")
        
        # 다중 레이더 시뮬레이션 실험
        scenarios = [
            {'name': 'close_range', 'description': '근거리 탐지 (0.5-2m)'},
            {'name': 'medium_range', 'description': '중거리 탐지 (2-5m)'},
            {'name': 'long_range', 'description': '원거리 탐지 (5-10m)'},
        ]
        
        scenario_results = experiment_controller.run_multi_radar_simulation(
            collector, scenarios
        )
        
        print(f"\n시나리오 실험 완료: {len(scenario_results)}개 시나리오")
        
    except Exception as e:
        print(f"실험 중 오류 발생: {e}")
    
    finally:
        # 메타데이터 저장
        experiment_controller.save_metadata()
        print("\n실험 완료!")


if __name__ == "__main__":
    main()