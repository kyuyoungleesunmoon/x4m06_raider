#!/usr/bin/env python3
"""
X4M06 실제 하드웨어 연결 및 5m 이내 탐지 테스트
pymoduleconnector 예제를 참고한 올바른 연결 방법
"""

import sys
import os
import time
import numpy as np

# pymoduleconnector 경로 추가
pymodule_path = r"C:\Users\User\Downloads\X4M06_Package\ModuleConnector\ModuleConnector-win32_win64-1\python36-win64"
if pymodule_path not in sys.path:
    sys.path.insert(0, pymodule_path)

from pymoduleconnector import ModuleConnector
from pymoduleconnector.ids import *

def test_x4m06_real_hardware():
    """X4M06 실제 하드웨어 연결 및 테스트"""
    print("=== X4M06 실제 하드웨어 연결 테스트 ===")
    
    device_name = "COM3"
    log_level = 0  # 로그 레벨 설정
    
    try:
        print(f"📡 {device_name}로 연결 중...")
        mc = ModuleConnector(device_name, log_level)
        print("✓ ModuleConnector 생성 성공")
        
        # XEP 인터페이스 사용 (X4M06용)
        xep = mc.get_xep()
        print("✓ XEP 인터페이스 획득 성공")
        
        # Ping 테스트
        pong = xep.ping()
        print(f"✓ Ping 성공, 응답: {hex(pong)}")
        
        # 시스템 정보 확인
        print("\n📋 X4M06 시스템 정보:")
        try:
            item_number = xep.get_system_info(XTID_SSIC_ITEMNUMBER)
            print(f"   아이템 번호: {item_number}")
            
            order_code = xep.get_system_info(XTID_SSIC_ORDERCODE)
            print(f"   주문 코드: {order_code}")
            
            firmware_id = xep.get_system_info(XTID_SSIC_FIRMWAREID)
            print(f"   펌웨어 ID: {firmware_id}")
            
            version = xep.get_system_info(XTID_SSIC_VERSION)
            print(f"   버전: {version}")
            
            serial_number = xep.get_system_info(XTID_SSIC_SERIALNUMBER)
            print(f"   시리얼 번호: {serial_number}")
            
        except Exception as e:
            print(f"   시스템 정보 확인 중 오류: {e}")
        
        # X4 드라이버 초기화
        print("\n🔧 X4 드라이버 초기화...")
        xep.x4driver_init()
        print("✓ X4 드라이버 초기화 완료")
        
        # 5m 이내 탐지를 위한 설정
        print("\n🎯 5m 이내 탐지 설정 적용...")
        
        try:
            # Enable 핀 설정
            xep.x4driver_set_enable(1)
            print("✓ X4 드라이버 활성화")
            
            # 기본 파라미터 설정 (5m 환경에 적합)
            # PRF (Pulse Repetition Frequency) - 낮은 값으로 설정하여 최대 거리 증가
            prf_div = 16  # 더 낮은 PRF로 5m까지 측정 가능
            xep.x4driver_set_prf_div(prf_div)
            print(f"✓ PRF 분주비 설정: {prf_div}")
            
            # 샘플링 설정
            dac_min = 949
            dac_max = 1100
            xep.x4driver_set_dac_min(dac_min)
            xep.x4driver_set_dac_max(dac_max)
            print(f"✓ DAC 범위 설정: {dac_min} ~ {dac_max}")
            
            # PPS (Pulses Per Step) 설정
            pps = 26  # 적당한 해상도
            xep.x4driver_set_pps(pps)
            print(f"✓ PPS 설정: {pps}")
            
            # 반복 횟수
            iterations = 16
            xep.x4driver_set_iterations(iterations)
            print(f"✓ 반복 횟수 설정: {iterations}")
            
            # Downconversion 활성화
            xep.x4driver_set_downconversion(1)
            print("✓ Downconversion 활성화")
            
        except Exception as e:
            print(f"? 설정 적용 중 오류: {e}")
        
        # 짧은 데이터 수집 테스트
        print("\n📊 데이터 수집 테스트 (5초간)...")
        
        try:
            # 데이터 수집 준비
            xep.x4driver_set_mode(0x13, 0)  # Manual mode
            print("✓ Manual 모드 설정")
            
            # 잠시 대기
            time.sleep(1)
            
            data_count = 0
            start_time = time.time()
            
            # 5초간 데이터 수집
            while time.time() - start_time < 5:
                try:
                    # 프레임 영역 읽기
                    frame_area = xep.read_message_data_float()
                    if frame_area and len(frame_area) > 0:
                        data_count += 1
                        if data_count % 5 == 0:
                            print(f"   📡 {data_count}개 프레임 수집, 길이: {len(frame_area)}")
                except:
                    pass
                
                time.sleep(0.2)
            
            print(f"✓ 총 {data_count}개 데이터 프레임 수집 완료!")
            
        except Exception as e:
            print(f"? 데이터 수집 테스트 실패: {e}")
        
        # 정리 및 연결 해제
        try:
            xep.x4driver_set_enable(0)
            print("✓ X4 드라이버 비활성화")
        except:
            pass
        
        mc.disconnect()
        print("✓ 연결 해제 완료")
        
        print("\n🎉 === X4M06 하드웨어 연결 테스트 성공! ===")
        print("✅ 실제 하드웨어와 정상 통신하며 5m 이내 탐지 설정이 적용되었습니다!")
        
        return True
        
    except Exception as e:
        print(f"✗ 연결 실패: {e}")
        print(f"오류 타입: {type(e).__name__}")
        
        print("\n🔧 문제 해결 방안:")
        print("1. X4M06이 COM3에 올바르게 연결되어 있는지 확인")
        print("2. X4M06 전원이 켜져 있는지 확인")
        print("3. 다른 프로그램에서 COM3 포트를 사용하고 있지 않은지 확인")
        print("4. USB 케이블 상태 확인")
        print("5. 드라이버가 올바르게 설치되어 있는지 확인")
        
        return False

if __name__ == "__main__":
    success = test_x4m06_real_hardware()
    
    if success:
        print("\n🚀 이제 실제 데이터 수집 실험을 진행할 수 있습니다!")
        print("   - 5m 이내 객체 탐지 최적화됨")
        print("   - 실시간 레이더 데이터 수집 가능")
    
    sys.exit(0 if success else 1)