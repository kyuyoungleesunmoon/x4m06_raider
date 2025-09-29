#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X4M06 간단 테스트용 스크립트
"""

import sys
import os

# pymoduleconnector 경로 설정
current_dir = os.path.dirname(__file__)
x4m06_path = r"C:\Users\User\Downloads\X4M06_Package\ModuleConnector\ModuleConnector-win32_win64-1\python36-win64"
sys.path.insert(0, x4m06_path)

from x4m06_data_saver import X4M06DataSaver

def quick_test():
    """빠른 테스트 (30초, 100프레임, 플롯 없음)"""
    print("🚀 빠른 테스트 시작 (30초, 100프레임, 플롯 없음)")
    
    data_saver = X4M06DataSaver("COM3")
    
    try:
        # 1. 리셋 및 연결
        if not data_saver.reset_x4m06():
            return
        if not data_saver.connect():
            return
        if not data_saver.setup_5m_detection():
            return
        if not data_saver.start_streaming(fps=20):
            return
        
        # 2. 데이터 수집 (플롯 없음)
        if data_saver.collect_and_save_data(30, 100, show_realtime=False):
            data_saver.save_data("json")
            print(f"✅ 테스트 완료! 세션: {data_saver.session_id}")
        else:
            print("❌ 데이터 수집 실패")
    
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
    
    finally:
        data_saver.disconnect()

if __name__ == "__main__":
    quick_test()