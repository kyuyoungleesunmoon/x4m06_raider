#!/usr/bin/env python3
"""
X4M06 레이더 연결 테스트
pymoduleconnector 라이브러리 연결 상태 확인
"""

import sys
import os
import time

print("=== X4M06 연결 테스트 시작 ===")
print(f"Python 버전: {sys.version}")
print(f"작업 디렉토리: {os.getcwd()}")

# 1. pymoduleconnector 모듈 import 테스트
print("\n1. pymoduleconnector 모듈 로드 테스트...")
try:
    # PYTHONPATH 설정
    pymodule_path = r"C:\Users\User\Downloads\X4M06_Package\ModuleConnector\ModuleConnector-win32_win64-1\python36-win64"
    if pymodule_path not in sys.path:
        sys.path.insert(0, pymodule_path)
    
    print(f"Python 경로에 추가: {pymodule_path}")
    
    # 기본 import 테스트
    import pymoduleconnector
    print("✓ pymoduleconnector 모듈 로드 성공")
    
    # 세부 클래스 import 테스트
    from pymoduleconnector import ModuleConnector
    print("✓ ModuleConnector 클래스 로드 성공")
    
except ImportError as e:
    print(f"✗ pymoduleconnector 모듈 로드 실패: {e}")
    print("세부 오류 확인 중...")
    
    try:
        # 직접 moduleconnectorwrapper 테스트
        from pymoduleconnector.moduleconnectorwrapper import PythonModuleConnector
        print("✓ PythonModuleConnector 로드 성공")
    except ImportError as e2:
        print(f"✗ PythonModuleConnector 로드 실패: {e2}")
    
    print("시뮬레이션 모드로만 사용 가능합니다.")
    sys.exit(1)

# 2. COM 포트 확인
print("\n2. COM 포트 확인...")
import serial.tools.list_ports
ports = list(serial.tools.list_ports.comports())
print("사용 가능한 COM 포트:")
for port in ports:
    print(f"  {port.device}: {port.description}")

target_port = "COM3"
com3_exists = any(port.device == target_port for port in ports)
if com3_exists:
    print(f"✓ {target_port} 포트 확인됨")
else:
    print(f"✗ {target_port} 포트를 찾을 수 없습니다")

# 3. X4M06 연결 테스트
print(f"\n3. X4M06 {target_port} 연결 테스트...")
try:
    # ModuleConnector 인스턴스 생성
    device_name = target_port
    mc = ModuleConnector(device_name)
    print(f"✓ {target_port}에 ModuleConnector 생성 성공")
    
    # X4M06 모듈 객체 가져오기
    x4m06 = mc.get_x4m06()
    print("✓ X4M06 모듈 객체 획득 성공")
    
    # 모듈 정보 확인
    try:
        # 펌웨어 버전 확인
        fw_version = x4m06.get_firmware_version()
        print(f"✓ 펌웨어 버전: {fw_version}")
    except Exception as e:
        print(f"? 펌웨어 버전 확인 실패: {e}")
    
    # 연결 해제
    mc.disconnect()
    print("✓ 연결 해제 완료")
    
    print("\n=== 연결 테스트 완료: 성공 ===")
    print("X4M06 하드웨어 연결이 정상 작동합니다!")
    
except Exception as e:
    print(f"✗ X4M06 연결 실패: {e}")
    print(f"오류 타입: {type(e).__name__}")
    print("\n가능한 원인:")
    print("1. X4M06이 COM3에 연결되지 않음")
    print("2. 다른 프로그램에서 COM3 사용 중")
    print("3. 드라이버 문제")
    print("4. pymoduleconnector 라이브러리 호환성 문제")
    
    print("\n=== 연결 테스트 완료: 실패 ===")
    sys.exit(1)