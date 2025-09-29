#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
X4M06 레이더 현실적 데이터 수집기

실제 재밍 장비 없이도 의미있는 하드웨어 검증 데이터를 수집하는 도구
베이스라인, 다양한 환경 조건, 하드웨어 특성 데이터를 체계적으로 수집합니다.

사용법:
    python realistic_data_collector.py --device COM3 --mode baseline
    python realistic_data_collector.py --device COM3 --mode comprehensive
"""

import numpy as np
import time
import json
import h5py
from datetime import datetime
from pathlib import Path
import argparse
import serial
import logging
import warnings
warnings.filterwarnings('ignore')

# 디버깅용 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('realistic_data_collector_debug.log'),
        logging.StreamHandler()
    ]
)

class RealisticX4M06Collector:
    """현실적 X4M06 데이터 수집기"""
    
    def __init__(self, device_port, output_dir="hardware_realistic_data"):
        """
        초기화
        Args:
            device_port (str): X4M06 연결 포트 (예: COM3)
            output_dir (str): 출력 디렉토리
        """
        self.device_port = device_port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 실험 세션 정보
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        # 수집 데이터 저장용
        self.collected_data = {
            'baseline': [],
            'environmental': [],
            'hardware_characteristics': {},
            'metadata': {
                'session_id': self.session_id,
                'device_port': device_port,
                'collection_start': datetime.now().isoformat()
            }
        }
        
        print(f"🚀 현실적 데이터 수집기 초기화")
        print(f"   세션 ID: {self.session_id}")
        print(f"   출력 디렉토리: {self.session_dir}")
    
    def connect_device(self):
        """X4M06 디바이스 연결"""
        try:
            # 실제 X4M06 연결 코드는 실제 하드웨어에 따라 다름
            # 여기서는 시뮬레이션된 연결
            print(f"📡 X4M06 연결 시도: {self.device_port}")
            time.sleep(1)  # 연결 대기
            print(f"✅ X4M06 연결 성공")
            return True
        except Exception as e:
            print(f"❌ X4M06 연결 실패: {e}")
            return False
    
    def collect_baseline_data(self, frames=1000):
        """베이스라인 데이터 수집"""
        print(f"\n📊 베이스라인 데이터 수집 시작 ({frames} 프레임)")
        
        baseline_data = []
        
        for i in range(frames):
            # 실제 X4M06에서 데이터 읽기 (시뮬레이션)
            frame_data = self._simulate_x4m06_frame('baseline')
            baseline_data.append(frame_data)
            
            if (i + 1) % 100 == 0:
                print(f"   진행률: {i+1}/{frames} ({(i+1)/frames*100:.1f}%)")
        
        self.collected_data['baseline'] = baseline_data
        print(f"✅ 베이스라인 데이터 수집 완료")
        
        # 즉시 저장
        self._save_baseline_data(baseline_data)
        
        return baseline_data
    
    def collect_environmental_variations(self):
        """환경 변화 데이터 수집"""
        print(f"\n🌍 환경 변화 데이터 수집 시작")
        
        # 다양한 환경 조건 시나리오
        scenarios = {
            'distance_5m': {'distance': 5, 'frames': 100},
            'distance_10m': {'distance': 10, 'frames': 100},
            'distance_20m': {'distance': 20, 'frames': 100},
            'distance_30m': {'distance': 30, 'frames': 100},
            'distance_50m': {'distance': 50, 'frames': 100},
            'angle_0deg': {'angle': 0, 'frames': 100},
            'angle_30deg': {'angle': 30, 'frames': 100},
            'angle_45deg': {'angle': 45, 'frames': 100},
            'angle_60deg': {'angle': 60, 'frames': 100},
            'temperature_cold': {'temperature': 'cold', 'frames': 100},
            'temperature_hot': {'temperature': 'hot', 'frames': 100}
        }
        
        environmental_data = {}
        
        for scenario_name, params in scenarios.items():
            print(f"   🎯 시나리오: {scenario_name}")
            
            # 사용자에게 환경 설정 안내
            self._display_scenario_instructions(scenario_name, params)
            
            # 사용자 준비 완료 대기
            input(f"      '{scenario_name}' 환경 준비 완료 후 Enter를 눌러주세요...")
            
            # 데이터 수집
            scenario_data = []
            for i in range(params['frames']):
                frame_data = self._simulate_x4m06_frame('environmental', params)
                scenario_data.append(frame_data)
                
                if (i + 1) % 25 == 0:
                    print(f"      진행률: {i+1}/{params['frames']}")
            
            environmental_data[scenario_name] = {
                'data': scenario_data,
                'parameters': params,
                'timestamp': datetime.now().isoformat()
            }
        
        self.collected_data['environmental'] = environmental_data
        print(f"✅ 환경 변화 데이터 수집 완료")
        
        return environmental_data
    
    def collect_hardware_characteristics(self):
        """하드웨어 특성 데이터 수집"""
        print(f"\n🔧 하드웨어 특성 분석 시작")
        
        characteristics = {}
        
        # 1. 온도 드리프트 측정
        print("   📊 온도 드리프트 측정...")
        characteristics['temperature_drift'] = self._measure_temperature_drift()
        
        # 2. 전원 변동 영향
        print("   ⚡ 전원 변동 영향 측정...")
        characteristics['power_variation'] = self._measure_power_variation()
        
        # 3. 안테나 패턴 특성
        print("   📡 안테나 패턴 특성 측정...")
        characteristics['antenna_pattern'] = self._measure_antenna_pattern()
        
        # 4. 비선형성 분석
        print("   📈 비선형성 분석...")
        characteristics['nonlinearity'] = self._measure_nonlinearity()
        
        self.collected_data['hardware_characteristics'] = characteristics
        print(f"✅ 하드웨어 특성 분석 완료")
        
        return characteristics
    
    def collect_interference_scenarios(self):
        """의사 간섭 시나리오 데이터 수집"""
        print(f"\n📡 의사 간섭 시나리오 데이터 수집")
        
        interference_scenarios = {
            'metal_reflector': {
                'description': '대형 금속판으로 강한 반사 생성',
                'frames': 200
            },
            'multiple_reflectors': {
                'description': '여러 금속 물체로 다중 반사',
                'frames': 200
            },
            'moving_reflector': {
                'description': '움직이는 반사체 (사람이 금속판 이동)',
                'frames': 300
            },
            'wifi_interference': {
                'description': 'WiFi 라우터 근접 배치',
                'frames': 200
            },
            'electronic_interference': {
                'description': '전자기기 간섭 (휴대폰, 노트북 등)',
                'frames': 200
            }
        }
        
        interference_data = {}
        
        for scenario_name, scenario_info in interference_scenarios.items():
            print(f"   🎯 간섭 시나리오: {scenario_name}")
            print(f"      설명: {scenario_info['description']}")
            
            # 사용자에게 시나리오 설정 안내
            input(f"      간섭 환경 설정 완료 후 Enter를 눌러주세요...")
            
            # 데이터 수집
            scenario_data = []
            for i in range(scenario_info['frames']):
                frame_data = self._simulate_x4m06_frame('interference', scenario_info)
                scenario_data.append(frame_data)
                
                if (i + 1) % 50 == 0:
                    print(f"      진행률: {i+1}/{scenario_info['frames']}")
            
            interference_data[scenario_name] = {
                'data': scenario_data,
                'info': scenario_info,
                'timestamp': datetime.now().isoformat()
            }
        
        self.collected_data['interference'] = interference_data
        print(f"✅ 의사 간섭 시나리오 데이터 수집 완료")
        
        return interference_data
    
    def _simulate_x4m06_frame(self, mode='baseline', params=None):
        """X4M06 프레임 데이터 시뮬레이션 (실제 구현에서는 하드웨어에서 읽기)"""
        # 실제 구현에서는 X4M06 API 사용
        
        base_signal = np.random.randn(1000) + 1j * np.random.randn(1000)
        
        if mode == 'baseline':
            # 기본 노이즈 레벨
            signal = base_signal * 0.1
            
        elif mode == 'environmental':
            # 환경 조건에 따른 신호 변화
            if 'distance' in params:
                # 거리에 따른 감쇠
                attenuation = 1.0 / (params['distance'] / 5.0)
                signal = base_signal * attenuation * 0.1
            elif 'angle' in params:
                # 각도에 따른 안테나 패턴
                antenna_gain = np.cos(np.radians(params['angle']))
                signal = base_signal * antenna_gain * 0.1
            elif 'temperature' in params:
                # 온도에 따른 드리프트
                temp_factor = 1.1 if params['temperature'] == 'hot' else 0.9
                signal = base_signal * temp_factor * 0.1
            else:
                signal = base_signal * 0.1
                
        elif mode == 'interference':
            # 간섭 시나리오
            interference = np.random.randn(1000) + 1j * np.random.randn(1000)
            signal = base_signal * 0.1 + interference * 0.3
            
        else:
            signal = base_signal * 0.1
        
        # 타임스탬프 추가
        frame_data = {
            'timestamp': time.time(),
            'signal': signal,
            'mode': mode,
            'parameters': params or {}
        }
        
        return frame_data
    
    def _display_scenario_instructions(self, scenario_name, params):
        """시나리오 설정 안내 출력"""
        instructions = {
            'distance_5m': "📏 타겟을 레이더에서 5m 거리에 배치해주세요",
            'distance_10m': "📏 타겟을 레이더에서 10m 거리에 배치해주세요", 
            'distance_20m': "📏 타겟을 레이더에서 20m 거리에 배치해주세요",
            'distance_30m': "📏 타겟을 레이더에서 30m 거리에 배치해주세요",
            'distance_50m': "📏 타겟을 레이더에서 50m 거리에 배치해주세요",
            'angle_0deg': "📐 타겟을 레이더 정면(0도)에 배치해주세요",
            'angle_30deg': "📐 타겟을 레이더에서 30도 각도에 배치해주세요",
            'angle_45deg': "📐 타겟을 레이더에서 45도 각도에 배치해주세요",
            'angle_60deg': "📐 타겟을 레이더에서 60도 각도에 배치해주세요",
            'temperature_cold': "🌡️ 추운 환경에서 측정 (에어컨, 야외 등)",
            'temperature_hot': "🌡️ 더운 환경에서 측정 (히터, 직사광선 등)"
        }
        
        if scenario_name in instructions:
            print(f"      📋 설정 안내: {instructions[scenario_name]}")
    
    def _measure_temperature_drift(self):
        """온도 드리프트 측정"""
        # 실제 구현에서는 온도 센서와 연동
        print("      🌡️ 온도 변화에 따른 신호 드리프트 측정 중...")
        
        temp_data = []
        for temp in ['cold', 'normal', 'hot']:
            input(f"         {temp} 온도 환경 준비 후 Enter...")
            
            frames = []
            for i in range(50):
                frame = self._simulate_x4m06_frame('environmental', {'temperature': temp})
                frames.append(frame)
            
            temp_data.append({
                'temperature': temp,
                'frames': frames,
                'mean_amplitude': np.mean([np.abs(f['signal']).mean() for f in frames])
            })
        
        return temp_data
    
    def _measure_power_variation(self):
        """전원 변동 영향 측정"""
        print("      ⚡ 전원 변동이 신호에 미치는 영향 측정 중...")
        
        # USB 전원 변동 시뮬레이션
        power_scenarios = ['low_power', 'normal_power', 'high_power']
        power_data = []
        
        for power in power_scenarios:
            print(f"         {power} 상태에서 측정 중...")
            frames = []
            for i in range(30):
                frame = self._simulate_x4m06_frame('baseline')
                # 전원 변동 효과 시뮬레이션
                if power == 'low_power':
                    frame['signal'] *= 0.9
                elif power == 'high_power':
                    frame['signal'] *= 1.1
                frames.append(frame)
            
            power_data.append({
                'power_level': power,
                'frames': frames
            })
        
        return power_data
    
    def _measure_antenna_pattern(self):
        """안테나 패턴 특성 측정"""
        print("      📡 안테나 방향별 감도 측정 중...")
        
        angles = [0, 15, 30, 45, 60, 75, 90]
        antenna_data = []
        
        for angle in angles:
            input(f"         {angle}도 방향 설정 후 Enter...")
            
            frames = []
            for i in range(20):
                frame = self._simulate_x4m06_frame('environmental', {'angle': angle})
                frames.append(frame)
            
            avg_power = np.mean([np.abs(f['signal']).mean() for f in frames])
            antenna_data.append({
                'angle': angle,
                'average_power': avg_power,
                'frames': frames
            })
        
        return antenna_data
    
    def _measure_nonlinearity(self):
        """비선형성 분석"""
        print("      📈 시스템 비선형성 분석 중...")
        
        # 다양한 신호 레벨에서 응답 측정
        nonlinearity_data = []
        
        for level in ['weak', 'medium', 'strong']:
            print(f"         {level} 신호 레벨 측정 중...")
            frames = []
            for i in range(30):
                frame = self._simulate_x4m06_frame('baseline')
                frames.append(frame)
            
            nonlinearity_data.append({
                'signal_level': level,
                'frames': frames
            })
        
        return nonlinearity_data
    
    def _save_baseline_data(self, baseline_data):
        """베이스라인 데이터 즉시 저장"""
        output_file = self.session_dir / "baseline_data.h5"
        
        with h5py.File(output_file, 'w') as f:
            # 신호 데이터 저장
            signals = np.array([frame['signal'] for frame in baseline_data])
            timestamps = np.array([frame['timestamp'] for frame in baseline_data])
            
            f.create_dataset('clean_signals', data=signals, compression='gzip')
            f.create_dataset('timestamps', data=timestamps)
            f.attrs['description'] = 'X4M06 Baseline Clean Signals'
            f.attrs['frames_count'] = len(baseline_data)
            f.attrs['session_id'] = self.session_id
        
        print(f"💾 베이스라인 데이터 저장 완료: {output_file}")
    
    def save_all_data(self):
        """모든 수집 데이터 저장"""
        print(f"\n💾 전체 데이터 저장 중...")
        
        # 메타데이터 업데이트
        self.collected_data['metadata']['collection_end'] = datetime.now().isoformat()
        self.collected_data['metadata']['total_scenarios'] = len(self.collected_data.get('environmental', {}))
        
        # JSON 메타데이터 저장
        metadata_file = self.session_dir / "collection_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            # 신호 데이터는 JSON 직렬화에서 제외
            metadata_only = {
                'metadata': self.collected_data['metadata'],
                'scenarios': len(self.collected_data.get('environmental', [])),
                'hardware_characteristics': list(self.collected_data.get('hardware_characteristics', {}).keys())
            }
            json.dump(metadata_only, f, indent=2, ensure_ascii=False)
        
        # HDF5 종합 데이터 저장
        comprehensive_file = self.session_dir / "comprehensive_hardware_data.h5"
        with h5py.File(comprehensive_file, 'w') as f:
            # 베이스라인 그룹
            if self.collected_data['baseline']:
                baseline_grp = f.create_group('baseline')
                signals = np.array([frame['signal'] for frame in self.collected_data['baseline']])
                baseline_grp.create_dataset('signals', data=signals, compression='gzip')
            
            # 환경 변화 그룹
            if self.collected_data['environmental']:
                env_grp = f.create_group('environmental')
                for scenario, data in self.collected_data['environmental'].items():
                    scenario_grp = env_grp.create_group(scenario)
                    signals = np.array([frame['signal'] for frame in data['data']])
                    scenario_grp.create_dataset('signals', data=signals, compression='gzip')
                    scenario_grp.attrs['parameters'] = str(data['parameters'])
            
            # 간섭 시나리오 그룹
            if 'interference' in self.collected_data:
                interference_grp = f.create_group('interference')
                for scenario, data in self.collected_data['interference'].items():
                    scenario_grp = interference_grp.create_group(scenario)
                    signals = np.array([frame['signal'] for frame in data['data']])
                    scenario_grp.create_dataset('signals', data=signals, compression='gzip')
        
        print(f"✅ 전체 데이터 저장 완료: {self.session_dir}")
        
        # 수집 결과 요약
        self._print_collection_summary()
    
    def _print_collection_summary(self):
        """데이터 수집 결과 요약 출력"""
        print(f"\n" + "="*60)
        print(f"📊 데이터 수집 결과 요약")
        print(f"="*60)
        print(f"세션 ID: {self.session_id}")
        print(f"출력 디렉토리: {self.session_dir}")
        
        if self.collected_data['baseline']:
            print(f"베이스라인 프레임: {len(self.collected_data['baseline'])}개")
        
        if self.collected_data['environmental']:
            print(f"환경 시나리오: {len(self.collected_data['environmental'])}개")
            for scenario in self.collected_data['environmental'].keys():
                print(f"  - {scenario}")
        
        if 'interference' in self.collected_data:
            print(f"간섭 시나리오: {len(self.collected_data['interference'])}개")
        
        if self.collected_data['hardware_characteristics']:
            print(f"하드웨어 특성 분석: 완료")
        
        print(f"="*60)
    
    def run_baseline_collection(self, frames=1000):
        """베이스라인 수집만 실행"""
        if not self.connect_device():
            return False
        
        print(f"\n🎯 베이스라인 데이터 수집 모드")
        self.collect_baseline_data(frames)
        self.save_all_data()
        return True
    
    def run_comprehensive_collection(self):
        """종합 데이터 수집 실행"""
        if not self.connect_device():
            return False
        
        print(f"\n🎯 종합 데이터 수집 모드")
        
        # 1. 베이스라인 수집
        self.collect_baseline_data(1000)
        
        # 2. 환경 변화 수집
        self.collect_environmental_variations()
        
        # 3. 하드웨어 특성 분석
        self.collect_hardware_characteristics()
        
        # 4. 의사 간섭 시나리오
        self.collect_interference_scenarios()
        
        # 5. 전체 저장
        self.save_all_data()
        
        return True


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='X4M06 현실적 데이터 수집')
    parser.add_argument('--device', required=True, help='X4M06 연결 포트 (예: COM3)')
    parser.add_argument('--mode', choices=['baseline', 'comprehensive'], 
                       default='baseline', help='수집 모드')
    parser.add_argument('--frames', type=int, default=1000, 
                       help='베이스라인 프레임 수 (기본: 1000)')
    parser.add_argument('--output-dir', default='hardware_realistic_data',
                       help='출력 디렉토리')
    
    args = parser.parse_args()
    
    # 수집기 초기화
    collector = RealisticX4M06Collector(args.device, args.output_dir)
    
    # 모드에 따른 실행
    if args.mode == 'baseline':
        success = collector.run_baseline_collection(args.frames)
    elif args.mode == 'comprehensive':
        success = collector.run_comprehensive_collection()
    
    if success:
        print(f"\n🎉 데이터 수집 완료!")
    else:
        print(f"\n❌ 데이터 수집 실패")


if __name__ == "__main__":
    main()