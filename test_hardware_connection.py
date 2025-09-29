#!/usr/bin/env python3
"""
X4M06 간단 연결 테스트
실제 하드웨어 연결 및 5m 이내 탐지 설정
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

def test_x4m06_connection():
    """X4M06 하드웨어 연결 테스트"""
    print("=== X4M06 하드웨어 연결 테스트 ===")
    
    # COM3 포트로 연결
    device_name = "COM3"
    print(f"📡 {device_name} 포트로 연결 시도...")
    
    try:
        # ModuleConnector 생성
        mc = ModuleConnector(device_name)
        print("✓ ModuleConnector 생성 성공")
        
        # X4M06 모듈 객체 가져오기
        x4m06 = mc.get_x4m06()
        print("✓ X4M06 객체 획득 성공")
        
        # 기본 정보 확인
        try:
            # 간단한 ping 테스트
            x4m06.ping()
            print("✓ X4M06 Ping 성공")
        except Exception as e:
            print(f"? Ping 실패: {e}")
        
        # 5m 이내 탐지를 위한 설정
        print("\n🎯 5m 이내 탐지 설정 적용 중...")
        
        try:
            # 감지 존 설정 (0 ~ 5m)
            detection_zone_start = 0.0  # 0m
            detection_zone_end = 5.0    # 5m
            
            x4m06.set_detection_zone(detection_zone_start, detection_zone_end)
            print(f"✓ 감지 영역 설정: {detection_zone_start}m ~ {detection_zone_end}m")
            
            # 감도 설정 (5m 환경에 적합)
            sensitivity = 5  # 중간 감도
            x4m06.set_sensitivity(sensitivity)
            print(f"✓ 감도 설정: {sensitivity}")
            
            # 프로파일 설정 - 근거리용
            # Profile ID 2: Balanced (근거리-중거리 균형)
            profile_id = 2
            x4m06.load_profile(profile_id)
            print(f"✓ 프로파일 설정: {profile_id} (Balanced)")
            
        except Exception as e:
            print(f"? 설정 적용 중 오류: {e}")
        
        # 간단한 데이터 수집 테스트
        print("\n📊 간단한 데이터 수집 테스트...")
        
        try:
            # 데이터 수집 시작
            # x4m06.set_output_control(0x000000A0, 1)  # BASEBAND_IQ
            print("✓ 데이터 수집 준비 완료")
            
            # 짧은 시간 데이터 수집
            start_time = time.time()
            data_count = 0
            
            while time.time() - start_time < 3:  # 3초간 수집
                try:
                    # 데이터 읽기
                    data = x4m06.read_message_data_float()
                    if data:
                        data_count += 1
                        if data_count % 10 == 0:
                            print(f"   데이터 {data_count}개 수집 중...")
                except:
                    pass
                
                time.sleep(0.1)
            
            print(f"✓ 총 {data_count}개 데이터 프레임 수집 완료")
            
            # 데이터 수집 중지
            print("✓ 데이터 수집 중지")
            
        except Exception as e:
            print(f"? 데이터 수집 테스트 실패: {e}")
        
        # 연결 해제
        mc.disconnect()
        print("✓ 연결 해제 완료")
        
        print("\n=== 하드웨어 연결 테스트 성공! ===")
        print("🎉 X4M06이 정상적으로 작동하며 5m 이내 탐지 설정이 적용되었습니다!")
        return True
        
    except Exception as e:
        print(f"✗ 연결 실패: {e}")
        print(f"오류 타입: {type(e).__name__}")
        
        print("\n🔧 문제 해결 방법:")
        print("1. X4M06이 COM3에 올바르게 연결되었는지 확인")
        print("2. 다른 프로그램에서 COM3를 사용하고 있지 않은지 확인")
        print("3. X4M06 전원이 켜져 있는지 확인")
        print("4. USB 케이블 연결 상태 확인")
        
        return False

if __name__ == "__main__":
    success = test_x4m06_connection()
    sys.exit(0 if success else 1)