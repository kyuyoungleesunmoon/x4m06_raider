#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X4M06 레이더 하드웨어 데이터 저장 프로그램
실제 하드웨어에서 데이터를 수집하여 파일로 저장합니다.
"""

import sys
import os
import numpy as np
import time
import json
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')  # 인터랙티브 백엔드 강제 설정
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.font_manager as fm
import threading

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'  # 기본 폰트로 설정
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# pymoduleconnector 경로 설정
current_dir = os.path.dirname(__file__)
x4m06_path = r"C:\Users\User\Downloads\X4M06_Package\ModuleConnector\ModuleConnector-win32_win64-1\python36-win64"
sys.path.insert(0, x4m06_path)

from pymoduleconnector import ModuleConnector

class X4M06DataSaver:
    """X4M06 데이터 수집 및 저장 클래스"""
    
    def __init__(self, device_name="COM3"):
        self.device_name = device_name
        self.mc = None
        self.xep = None
        self.save_dir = "collected_data"
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 데이터 저장용 변수
        self.frame_data = []
        self.frame_counter = 0
        
        # 실시간 플롯용 변수
        self.current_frame = None
        self.plot_enabled = False
        self.fig = None
        self.ax = None
        self.line = None
        self.animation = None
        
        # 저장 디렉토리 생성
        os.makedirs(self.save_dir, exist_ok=True)
        
    def reset_x4m06(self):
        """X4M06 모듈 리셋"""
        print("🔄 X4M06 모듈 리셋 중...")
        try:
            mc_reset = ModuleConnector(self.device_name, log_level=0)
            xep_reset = mc_reset.get_xep()
            xep_reset.module_reset()
            mc_reset.close()
            time.sleep(3)
            print("✓ 모듈 리셋 완료")
            return True
        except Exception as e:
            print(f"❌ 리셋 실패: {e}")
            return False
    
    def connect(self):
        """X4M06 연결"""
        try:
            print(f"📡 {self.device_name} 연결 중...")
            self.mc = ModuleConnector(self.device_name, log_level=0)
            self.xep = self.mc.get_xep()
            print("✓ 연결 성공")
            return True
        except Exception as e:
            print(f"❌ 연결 실패: {e}")
            return False
    
    def setup_5m_detection(self):
        """5m 이내 탐지 설정"""
        try:
            print("🎯 5m 이내 탐지 설정 적용 중...")
            
            # DAC 범위: 900-1400 (약 5m) - 제조사 예제 기반
            self.xep.x4driver_set_dac_min(900)
            self.xep.x4driver_set_dac_max(1400)
            
            # 반복 및 펄스 설정 - 제조사 예제와 동일
            self.xep.x4driver_set_iterations(16)
            self.xep.x4driver_set_pulses_per_step(26)
            
            print("✓ DAC 범위: 900 ~ 1400")
            print("✓ 반복: 16, PPS: 26")
            print("✓ 제조사 설정 적용됨")
            return True
            
        except Exception as e:
            print(f"❌ 설정 실패: {e}")
            return False
    
    def start_streaming(self, fps=20):
        """데이터 스트리밍 시작"""
        try:
            print(f"📊 {fps} FPS로 데이터 스트리밍 시작...")
            
            # 데이터 출력 제어 설정
            self.xep.x4driver_set_fps(fps)
            
            # 기존 버퍼 클리어
            while self.xep.peek_message_data_float():
                self.xep.read_message_data_float()
            
            # 스트리밍 시작
            self.xep.x4driver_set_enable(1)
            print("✓ 스트리밍 시작됨")
            return True
            
        except Exception as e:
            print(f"❌ 스트리밍 시작 실패: {e}")
            return False
    
    def setup_realtime_plot(self):
        """실시간 플롯 설정"""
        try:
            print("📈 실시간 데이터 플롯 설정 중...")
            
            # 인터랙티브 모드 활성화
            plt.ion()
            
            # 플롯 초기화
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
            self.ax.set_title(f'X4M06 Realtime Radar Data (5m Detection) - Session: {self.session_id}', fontsize=14)
            self.ax.set_xlabel('Sample Index', fontsize=12)
            self.ax.set_ylabel('Signal Intensity', fontsize=12)
            self.ax.grid(True, alpha=0.3)
            
            # 초기 빈 라인 생성
            dummy_data = np.zeros(1488)  # 예상 프레임 크기
            self.line, = self.ax.plot(dummy_data, 'b-', linewidth=1.5, alpha=0.8)
            
            # Y축 범위 설정 (예상 범위)
            self.ax.set_ylim(-0.2, 0.2)
            self.ax.set_xlim(0, 1488)
            
            # 상태 표시 텍스트
            self.status_text = self.ax.text(0.02, 0.95, 'Waiting for data...', 
                                          transform=self.ax.transAxes, 
                                          fontsize=11, 
                                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            
            # 플롯 창을 앞으로 가져오기
            self.fig.canvas.manager.window.wm_attributes('-topmost', 1)
            self.fig.canvas.manager.window.wm_attributes('-topmost', 0)
            
            # 플롯 표시
            plt.show(block=False)
            plt.draw()
            plt.pause(0.5)  # 창이 표시될 시간 확보
            
            self.plot_enabled = True
            print("✓ 실시간 플롯 준비 완료 (플롯 창 확인)")
            return True
            
        except Exception as e:
            print(f"❌ 플롯 설정 실패: {e}")
            return False
    
    def update_plot_simple(self):
        """간단한 실시간 플롯 업데이트"""
        if self.current_frame is not None and self.plot_enabled and self.fig is not None:
            try:
                # 플롯 창이 살아있는지 확인
                if plt.fignum_exists(self.fig.number):
                    # 데이터 업데이트
                    self.line.set_ydata(self.current_frame)
                    
                    # X축 범위 설정 (프레임 길이에 맞게)
                    if len(self.current_frame) != len(self.line.get_xdata()):
                        self.line.set_xdata(range(len(self.current_frame)))
                        self.ax.set_xlim(0, len(self.current_frame))
                    
                    # Y축 자동 조정
                    y_min, y_max = np.min(self.current_frame), np.max(self.current_frame)
                    y_range = y_max - y_min
                    if y_range > 0:
                        self.ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.1)
                    
                    # 상태 정보 업데이트
                    status_msg = f'Frame: {self.frame_counter} | Range: [{y_min:.3f}, {y_max:.3f}] | Mean: {np.mean(self.current_frame):.3f}'
                    self.status_text.set_text(status_msg)
                    
                    # 플롯 업데이트
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()
                else:
                    # 플롯 창이 닫혔으면 플롯 비활성화
                    self.plot_enabled = False
                    print("   📊 플롯 창이 닫혔습니다 - 데이터 수집만 계속됩니다")
                
            except Exception as e:
                # 오류가 발생하면 플롯 비활성화
                self.plot_enabled = False
        
        return True
    
    def start_realtime_plot(self):
        """실시간 플롯 시작"""
        if self.plot_enabled and self.fig is not None:
            print("📊 실시간 플롯 활성화됨 (플롯 창 확인)")
            print("   💡 데이터 수집과 함께 플롯이 실시간으로 업데이트됩니다")
            return True
        return False
    
    def stop_realtime_plot(self):
        """실시간 플롯 중지"""
        self.plot_enabled = False
        try:
            if self.fig:
                plt.ioff()  # 인터랙티브 모드 비활성화
                # 5초 후에 자동으로 창 닫기
                print("   💡 플롯 창을 5초 후 자동 닫습니다 (수동으로 닫으셔도 됩니다)")
                plt.pause(5)
                plt.close(self.fig)
        except:
            pass
    
    def collect_and_save_data(self, duration_seconds=60, max_frames=1000, show_realtime=True):
        """데이터 수집 및 저장 (실시간 플롯 포함)"""
        print(f"\n📈 데이터 수집 시작 (최대 {duration_seconds}초 또는 {max_frames}프레임)")
        print("   Ctrl+C로 중단 가능")
        
        # 실시간 플롯 설정
        if show_realtime:
            if not self.setup_realtime_plot():
                show_realtime = False
            else:
                self.start_realtime_plot()
        
        start_time = time.time()
        self.frame_data = []
        self.frame_counter = 0
        
        try:
            print(f"🎬 {'실시간 플롯과 함께 ' if show_realtime else ''}데이터 수집 중...")
            
            while True:
                # 시간 및 프레임 수 체크
                elapsed_time = time.time() - start_time
                if elapsed_time > duration_seconds or self.frame_counter >= max_frames:
                    break
                
                # 데이터 읽기 - 제조사 예제 방식
                if self.xep.peek_message_data_float():
                    data_float = self.xep.read_message_data_float()
                    frame_data = np.array(data_float.data)
                    
                    if frame_data is not None and len(frame_data) > 0:
                        # 실시간 플롯용 현재 프레임 업데이트
                        self.current_frame = frame_data.copy()
                        
                        # 실시간 플롯 업데이트
                        if show_realtime and self.frame_counter % 2 == 0:  # 2프레임마다 업데이트
                            self.update_plot_simple()
                        
                        # 프레임 정보 저장
                        frame_info = {
                            'frame_number': self.frame_counter,
                            'timestamp': time.time(),
                            'data': frame_data.tolist(),
                            'length': len(frame_data),
                            'max_value': float(np.max(frame_data)),
                            'min_value': float(np.min(frame_data)),
                            'mean_value': float(np.mean(frame_data)),
                            'std_value': float(np.std(frame_data))
                        }
                        
                        self.frame_data.append(frame_info)
                        self.frame_counter += 1
                        
                        # 진행 상황 표시 (실시간 플롯이 있을 때는 덜 자주)
                        display_interval = 50 if show_realtime else 20
                        if self.frame_counter % display_interval == 0:
                            print(f"   프레임 {self.frame_counter}: 길이={len(frame_data)}, "
                                  f"범위=[{np.min(frame_data):.3f}, {np.max(frame_data):.3f}], "
                                  f"경과시간={elapsed_time:.1f}초")
                
                # 실시간 플롯을 위한 짧은 대기
                if show_realtime:
                    plt.pause(0.01)  # 플롯 업데이트를 위한 대기
                else:
                    time.sleep(0.01)  # CPU 사용량 조절
                
        except KeyboardInterrupt:
            print("\n⚠️  사용자에 의해 중단됨")
        
        finally:
            # 실시간 플롯 정리
            if show_realtime:
                print("🎬 실시간 플롯 중지 중...")
                self.stop_realtime_plot()
        
        print(f"\n📊 수집 완료: {self.frame_counter}개 프레임, {elapsed_time:.1f}초")
        return self.frame_counter > 0
    
    def save_data(self, data_type="json"):
        """수집된 데이터를 파일로 저장"""
        if not self.frame_data:
            print("❌ 저장할 데이터가 없습니다")
            return False
        
        # 파일명 생성
        base_filename = f"x4m06_data_{self.session_id}"
        
        try:
            if data_type == "json":
                # JSON 형식으로 저장 (메타데이터 포함)
                json_filename = os.path.join(self.save_dir, f"{base_filename}.json")
                
                save_data = {
                    'session_info': {
                        'session_id': self.session_id,
                        'device': self.device_name,
                        'collection_time': datetime.now().isoformat(),
                        'total_frames': len(self.frame_data),
                        'settings': {
                            'dac_min': 900,
                            'dac_max': 1400,
                            'iterations': 16,
                            'pps': 26,
                            'prf_div': 16,
                            'frame_area': '0-5m'
                        }
                    },
                    'frames': self.frame_data
                }
                
                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
                
                print(f"✓ JSON 데이터 저장: {json_filename}")
            
            # NumPy 배열로도 저장 (빠른 분석용)
            np_filename = os.path.join(self.save_dir, f"{base_filename}.npz")
            
            # 모든 프레임 데이터를 2D 배열로 변환
            frame_arrays = []
            timestamps = []
            frame_numbers = []
            
            for frame_info in self.frame_data:
                frame_arrays.append(frame_info['data'])
                timestamps.append(frame_info['timestamp'])
                frame_numbers.append(frame_info['frame_number'])
            
            # 가장 긴 프레임 길이로 패딩
            max_length = max(len(frame) for frame in frame_arrays)
            padded_frames = np.zeros((len(frame_arrays), max_length))
            
            for i, frame in enumerate(frame_arrays):
                padded_frames[i, :len(frame)] = frame
            
            np.savez_compressed(np_filename,
                              frames=padded_frames,
                              timestamps=np.array(timestamps),
                              frame_numbers=np.array(frame_numbers),
                              session_id=self.session_id)
            
            print(f"✓ NumPy 데이터 저장: {np_filename}")
            
            # 요약 통계 저장
            self.save_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터 저장 실패: {e}")
            return False
    
    def save_summary(self):
        """수집 요약 정보 저장"""
        try:
            summary_filename = os.path.join(self.save_dir, f"summary_{self.session_id}.txt")
            
            # 통계 계산
            all_max_values = [frame['max_value'] for frame in self.frame_data]
            all_min_values = [frame['min_value'] for frame in self.frame_data]
            all_mean_values = [frame['mean_value'] for frame in self.frame_data]
            frame_lengths = [frame['length'] for frame in self.frame_data]
            
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write(f"X4M06 데이터 수집 요약\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"세션 ID: {self.session_id}\n")
                f.write(f"장치: {self.device_name}\n")
                f.write(f"수집 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"총 프레임 수: {len(self.frame_data)}\n\n")
                
                f.write(f"프레임 길이 통계:\n")
                f.write(f"  평균: {np.mean(frame_lengths):.1f}\n")
                f.write(f"  최소: {np.min(frame_lengths)}\n")
                f.write(f"  최대: {np.max(frame_lengths)}\n\n")
                
                f.write(f"신호 크기 통계:\n")
                f.write(f"  전체 최대값: {np.max(all_max_values):.6f}\n")
                f.write(f"  전체 최소값: {np.min(all_min_values):.6f}\n")
                f.write(f"  평균값 평균: {np.mean(all_mean_values):.6f}\n")
                f.write(f"  평균값 표준편차: {np.std(all_mean_values):.6f}\n\n")
                
                f.write(f"레이더 설정:\n")
                f.write(f"  DAC 범위: 900 - 1400 (약 5m)\n")
                f.write(f"  반복: 16\n")
                f.write(f"  PPS: 26\n")
                f.write(f"  PRF 분주비: 16\n")
            
            print(f"✓ 요약 정보 저장: {summary_filename}")
            
        except Exception as e:
            print(f"❌ 요약 저장 실패: {e}")
    
    def disconnect(self):
        """연결 해제"""
        try:
            if self.xep:
                self.xep.x4driver_set_enable(0)
            if self.mc:
                self.mc.close()
            print("✓ 연결 해제 완료")
        except Exception as e:
            print(f"⚠️  연결 해제 중 오류: {e}")

def main():
    """메인 실행 함수"""
    print("🎯 X4M06 하드웨어 데이터 저장 프로그램")
    print("   포트: COM3")
    print("   범위: 5m 이내 탐지")
    print("   저장 형식: JSON + NumPy")
    
    # 데이터 수집기 초기화
    data_saver = X4M06DataSaver("COM3")
    
    try:
        # 1. 모듈 리셋
        if not data_saver.reset_x4m06():
            return
        
        # 2. 연결
        if not data_saver.connect():
            return
        
        # 3. 5m 탐지 설정
        if not data_saver.setup_5m_detection():
            return
        
        # 4. 스트리밍 시작
        if not data_saver.start_streaming(fps=20):
            return
        
        # 5. 사용자 입력으로 수집 시간 설정
        print("\n📝 수집 설정:")
        try:
            duration = int(input("   수집 시간(초, 기본값 60): ") or "60")
            max_frames = int(input("   최대 프레임 수(기본값 1000): ") or "1000")
            show_plot = input("   실시간 플롯 표시? (Y/n, 기본값 Y): ").strip().lower()
            show_realtime = show_plot != 'n'
        except ValueError:
            duration = 60
            max_frames = 1000
            show_realtime = True
            print("   기본값 사용: 60초, 1000프레임, 실시간 플롯 ON")
        
        if show_realtime:
            print("📊 실시간 레이더 데이터 플롯이 함께 표시됩니다!")
        
        # 6. 데이터 수집
        if data_saver.collect_and_save_data(duration, max_frames, show_realtime):
            # 7. 데이터 저장
            data_saver.save_data("json")
            
            print(f"\n🎉 === 데이터 저장 완료! ===")
            print(f"📁 저장 위치: {data_saver.save_dir}/")
            print(f"📊 세션 ID: {data_saver.session_id}")
            print(f"📈 총 프레임: {len(data_saver.frame_data)}개")
        else:
            print("❌ 데이터 수집 실패")
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    
    finally:
        # 8. 정리
        print("\n🔄 정리 중...")
        data_saver.disconnect()

if __name__ == "__main__":
    main()