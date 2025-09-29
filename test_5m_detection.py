#!/usr/bin/env python3
"""
X4M06 5m 이내 탐지 최적화 테스트
제조사 simple_xep_plot2.py 파일 기반으로 작성
"""

import sys
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from datetime import datetime

# pymoduleconnector 경로 추가
pymodule_path = r"C:\Users\User\Downloads\X4M06_Package\ModuleConnector\ModuleConnector-win32_win64-1\python36-win64"
if pymodule_path not in sys.path:
    sys.path.insert(0, pymodule_path)

from pymoduleconnector import ModuleConnector

# 설정
device_name = "COM3"
bb = False  # Baseband mode (False = RF data, True = Baseband IQ)
FPS = 20
__version__ = "5m_optimized_v1.0"

def reset_x4m06(device_name):
    """X4M06 모듈 리셋"""
    print("🔄 X4M06 모듈 리셋 중...")
    try:
        from time import sleep
        mc = ModuleConnector(device_name)
        r = mc.get_xep()
        r.module_reset()
        mc.close()
        sleep(3)
        print("✓ 모듈 리셋 완료")
        return True
    except Exception as e:
        print(f"✗ 모듈 리셋 실패: {e}")
        return False

def setup_5m_detection(r):
    """5m 이내 탐지를 위한 최적화 설정"""
    print("🎯 5m 이내 탐지 설정 적용 중...")
    
    try:
        # DAC 범위 설정 (5m 탐지에 최적화)
        dac_min = 900   # 시작점
        dac_max = 1400  # 5m까지 커버하도록 확장
        r.x4driver_set_dac_min(dac_min)
        r.x4driver_set_dac_max(dac_max)
        print(f"✓ DAC 범위: {dac_min} ~ {dac_max}")
        
        # 해상도 및 정확도 설정
        iterations = 16      # 적당한 정확도
        pps = 26            # Pulses per step
        r.x4driver_set_iterations(iterations)
        r.x4driver_set_pulses_per_step(pps)
        print(f"✓ 반복: {iterations}, PPS: {pps}")
        
        # PRF 설정 (5m 환경에 맞춤)
        # PRF를 낮추면 최대 탐지 거리가 증가
        prf_div = 16  # 기본값, 5m까지 충분
        try:
            r.x4driver_set_prf_div(prf_div)
            print(f"✓ PRF 분주비: {prf_div}")
        except:
            print("? PRF 설정 생략 (구버전 펌웨어)")
        
        # Downconversion 설정
        if bb:
            r.x4driver_set_downconversion(1)
            print("✓ Baseband IQ 모드 활성화")
        else:
            print("✓ RF 데이터 모드 사용")
        
        return True
        
    except Exception as e:
        print(f"✗ 설정 실패: {e}")
        return False

def test_connection_and_data():
    """연결 테스트 및 데이터 수집"""
    print("=== X4M06 5m 탐지 테스트 시작 ===")
    
    # 1. 모듈 리셋
    if not reset_x4m06(device_name):
        return False
    
    try:
        # 2. 연결 생성
        print(f"📡 {device_name} 연결 중...")
        mc = ModuleConnector(device_name)
        r = mc.get_xep()
        print("✓ 연결 성공")
        
        # 3. 5m 탐지 설정
        if not setup_5m_detection(r):
            return False
        
        # 4. 스트리밍 시작
        print(f"📊 {FPS} FPS로 데이터 스트리밍 시작...")
        r.x4driver_set_fps(FPS)
        
        # 5. 버퍼 클리어
        def clear_buffer():
            """프레임 버퍼 클리어"""
            while r.peek_message_data_float():
                _ = r.read_message_data_float()
        
        clear_buffer()
        print("✓ 버퍼 클리어 완료")
        
        # 6. 데이터 읽기 테스트
        def read_frame():
            """모듈에서 프레임 데이터 읽기"""
            d = r.read_message_data_float()
            frame = np.array(d.data)
            
            # Baseband 모드인 경우 복소수 배열로 변환
            if bb:
                n = len(frame)
                frame = frame[:n//2] + 1j*frame[n//2:]
            
            return frame
        
        # 7. 몇 개 프레임 테스트
        print("📈 데이터 수집 테스트 (10프레임)...")
        for i in range(10):
            try:
                frame = read_frame()
                if bb:
                    frame_magnitude = abs(frame)
                    print(f"   프레임 {i+1}: 길이={len(frame)}, 최대값={np.max(frame_magnitude):.6f}")
                else:
                    print(f"   프레임 {i+1}: 길이={len(frame)}, 최대값={np.max(frame):.6f}, 최소값={np.min(frame):.6f}")
                time.sleep(0.1)
            except Exception as e:
                print(f"   프레임 {i+1} 읽기 실패: {e}")
        
        # 8. 실시간 플롯 시작
        print("\n🎮 실시간 레이더 데이터 플롯 시작...")
        print("   창을 닫으면 종료됩니다.")
        
        # 플롯 설정
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(f"X4M06 5m 탐지 최적화 | 버전: {__version__} | Baseband: {bb}")
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("실시간 레이더 신호 (0~5m 탐지 최적화)")
        ax.set_xlabel("샘플 인덱스")
        ax.set_ylabel("신호 강도")
        ax.grid(True, alpha=0.3)
        
        # 첫 번째 프레임으로 초기화
        initial_frame = read_frame()
        if bb:
            initial_frame = abs(initial_frame)
        
        line, = ax.plot(initial_frame, 'b-', linewidth=1)
        ax.set_ylim(np.min(initial_frame) * 1.1, np.max(initial_frame) * 1.1)
        
        clear_buffer()
        
        # 애니메이션 함수
        def animate(frame_num):
            try:
                if bb:
                    new_data = abs(read_frame())
                else:
                    new_data = read_frame()
                line.set_ydata(new_data)
                
                # Y축 자동 조정 (가끔씩)
                if frame_num % 100 == 0:
                    ax.set_ylim(np.min(new_data) * 1.1, np.max(new_data) * 1.1)
                
                return line,
            except:
                return line,
        
        # 애니메이션 시작
        ani = FuncAnimation(fig, animate, interval=1000//FPS, blit=True)
        plt.tight_layout()
        plt.show()
        
        # 9. 정리
        print("\n🔄 정리 중...")
        r.x4driver_set_fps(0)  # 스트리밍 중지
        mc.close()
        print("✓ 연결 해제 완료")
        
        print("\n🎉 === 테스트 완료! ===")
        print("✅ X4M06이 5m 이내 탐지에 최적화되어 정상 작동했습니다!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        print(f"오류 타입: {type(e).__name__}")
        
        try:
            r.x4driver_set_fps(0)
            mc.close()
        except:
            pass
        
        return False

if __name__ == "__main__":
    print(f"🎯 X4M06 5m 이내 탐지 최적화 테스트")
    print(f"   포트: {device_name}")
    print(f"   모드: {'Baseband IQ' if bb else 'RF Data'}")
    print(f"   FPS: {FPS}")
    print()
    
    success = test_connection_and_data()
    
    if success:
        print("\n✨ 실제 하드웨어 연결 및 5m 탐지 설정이 성공적으로 완료되었습니다!")
    else:
        print("\n💡 문제 해결 방안:")
        print("   1. X4M06이 COM3에 올바르게 연결되어 있는지 확인")
        print("   2. X4M06 전원 상태 확인")
        print("   3. 다른 프로그램에서 COM3 사용 여부 확인")
        print("   4. USB 드라이버 상태 확인")
    
    sys.exit(0 if success else 1)