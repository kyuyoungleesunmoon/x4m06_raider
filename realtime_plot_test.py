#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X4M06 실시간 플롯 테스트 (짧은 시간)
"""

import sys
import os

# pymoduleconnector 경로 설정
current_dir = os.path.dirname(__file__)
x4m06_path = r"C:\Users\User\Downloads\X4M06_Package\ModuleConnector\ModuleConnector-win32_win64-1\python36-win64"
sys.path.insert(0, x4m06_path)

from x4m06_data_saver import X4M06DataSaver

def realtime_plot_test():
    """실시간 플롯 테스트 (15초, 50프레임, 플롯 있음)"""
    print("📊 실시간 플롯 테스트 시작 (15초, 50프레임, 플롯 포함)")
    print("   💡 플롯 창이 표시되면 실시간 데이터를 확인할 수 있습니다")
    
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
        
        # 2. 데이터 수집 (플롯 포함)
        if data_saver.collect_and_save_data(15, 50, show_realtime=True):
            data_saver.save_data("json")
            print(f"✅ 실시간 플롯 테스트 완료! 세션: {data_saver.session_id}")
        else:
            print("❌ 데이터 수집 실패")
    
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
    
    finally:
        data_saver.disconnect()

if __name__ == "__main__":
    realtime_plot_test()