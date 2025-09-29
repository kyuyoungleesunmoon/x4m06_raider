#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X4M06 간편 데이터 수집기
사용자 친화적인 메뉴 인터페이스로 데이터 수집
"""

import sys
import os
import time

# pymoduleconnector 경로 설정
current_dir = os.path.dirname(__file__)
x4m06_path = r"C:\Users\User\Downloads\X4M06_Package\ModuleConnector\ModuleConnector-win32_win64-1\python36-win64"
sys.path.insert(0, x4m06_path)

from x4m06_data_saver import X4M06DataSaver

class EasyDataCollector:
    """간편 데이터 수집기"""
    
    def __init__(self):
        self.com_port = "COM3"
        self.presets = {
            "1": {"name": "빠른 테스트", "duration": 30, "frames": 100, "plot": False},
            "2": {"name": "표준 수집", "duration": 120, "frames": 500, "plot": True},
            "3": {"name": "장시간 수집", "duration": 300, "frames": 1000, "plot": False},
            "4": {"name": "실시간 모니터링", "duration": 60, "frames": 300, "plot": True},
            "5": {"name": "사용자 정의", "duration": 0, "frames": 0, "plot": True}
        }
    
    def print_header(self):
        """헤더 출력"""
        print("=" * 60)
        print("🎯 X4M06 UWB 레이더 데이터 수집기")
        print("   Version 1.0 - 사용자 친화적 인터페이스")
        print("=" * 60)
        print(f"📡 연결 포트: {self.com_port}")
        print(f"🎯 탐지 범위: 5m 이내 (DAC 900-1400)")
        print(f"📊 샘플링: 20 FPS")
        print("=" * 60)
    
    def show_presets(self):
        """사전 설정 메뉴 표시"""
        print("\n📋 수집 모드 선택:")
        print("─" * 50)
        for key, preset in self.presets.items():
            if key != "5":
                plot_str = "실시간 플롯 ON" if preset["plot"] else "플롯 OFF"
                print(f"  {key}. {preset['name']:<12} | {preset['duration']:3d}초 | {preset['frames']:4d}프레임 | {plot_str}")
        print(f"  5. 사용자 정의        | 직접 설정")
        print("─" * 50)
        print("  0. 종료")
        print()
    
    def get_user_choice(self):
        """사용자 선택 입력"""
        while True:
            try:
                choice = input("선택하세요 (0-5): ").strip()
                if choice in ["0", "1", "2", "3", "4", "5"]:
                    return choice
                else:
                    print("❌ 잘못된 선택입니다. 0-5 중에서 선택해주세요.")
            except KeyboardInterrupt:
                print("\n\n👋 프로그램을 종료합니다.")
                sys.exit(0)
    
    def get_custom_settings(self):
        """사용자 정의 설정 입력"""
        print("\n⚙️ 사용자 정의 설정:")
        print("─" * 30)
        
        try:
            duration = int(input("수집 시간 (초, 10-600): ") or "60")
            duration = max(10, min(600, duration))
            
            frames = int(input("최대 프레임 수 (100-5000): ") or "500")
            frames = max(100, min(5000, frames))
            
            plot_choice = input("실시간 플롯 표시? (Y/n): ").strip().lower()
            show_plot = plot_choice != 'n'
            
            return {"duration": duration, "frames": frames, "plot": show_plot}
            
        except ValueError:
            print("⚠️ 잘못된 입력입니다. 기본값을 사용합니다.")
            return {"duration": 60, "frames": 500, "plot": True}
    
    def check_connection(self):
        """연결 상태 확인"""
        print(f"\n🔍 {self.com_port} 연결 확인 중...")
        
        try:
            # 간단한 연결 테스트
            from pymoduleconnector import ModuleConnector
            mc = ModuleConnector(self.com_port, log_level=0)
            mc.close()
            print("✅ 레이더 연결 확인됨")
            return True
        except Exception as e:
            print(f"❌ 연결 실패: {e}")
            print("\n💡 해결 방법:")
            print("  1. USB 케이블 확인")
            print("  2. COM 포트 번호 확인 (장치 관리자)")
            print("  3. 다른 프로그램에서 레이더 사용 중인지 확인")
            return False
    
    def collect_data(self, settings):
        """데이터 수집 실행"""
        print(f"\n🚀 데이터 수집 시작:")
        print(f"   ⏱️  수집 시간: {settings['duration']}초")
        print(f"   📊 최대 프레임: {settings['frames']}개")
        print(f"   📈 실시간 플롯: {'ON' if settings['plot'] else 'OFF'}")
        print()
        
        data_saver = X4M06DataSaver(self.com_port)
        
        try:
            # 1. 초기화
            if not data_saver.reset_x4m06():
                return False
            if not data_saver.connect():
                return False
            if not data_saver.setup_5m_detection():
                return False
            if not data_saver.start_streaming(fps=20):
                return False
            
            # 2. 데이터 수집
            if data_saver.collect_and_save_data(
                settings['duration'], 
                settings['frames'], 
                settings['plot']
            ):
                # 3. 데이터 저장
                data_saver.save_data("json")
                
                print(f"\n🎉 데이터 수집 완료!")
                print(f"📁 저장 위치: collected_data/")
                print(f"📊 세션 ID: {data_saver.session_id}")
                print(f"📈 수집된 프레임: {len(data_saver.frame_data)}개")
                
                # 4. 분석 옵션 제공
                self.offer_analysis(data_saver.session_id)
                return True
            else:
                print("❌ 데이터 수집 실패")
                return False
                
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            return False
        
        finally:
            data_saver.disconnect()
    
    def offer_analysis(self, session_id):
        """분석 옵션 제공"""
        print(f"\n📊 수집 완료! 다음 작업을 선택하세요:")
        print(f"  1. 데이터 분석 및 시각화")
        print(f"  2. 다른 수집 작업 계속")
        print(f"  3. 종료")
        
        choice = input("선택하세요 (1-3): ").strip()
        
        if choice == "1":
            print(f"\n🔍 데이터 분석기를 실행합니다...")
            try:
                # 분석기 실행
                os.system("python hardware_data_analyzer.py")
            except:
                print("❌ 분석기 실행 실패. 수동으로 'python hardware_data_analyzer.py' 실행해주세요.")
        elif choice == "2":
            return True  # 메인 루프 계속
        else:
            return False  # 종료
    
    def run(self):
        """메인 실행 루프"""
        self.print_header()
        
        # 연결 확인
        if not self.check_connection():
            input("\nEnter 키를 눌러 종료...")
            return
        
        while True:
            self.show_presets()
            choice = self.get_user_choice()
            
            if choice == "0":
                print("\n👋 프로그램을 종료합니다.")
                break
            
            # 설정 가져오기
            if choice == "5":
                settings = self.get_custom_settings()
            else:
                settings = self.presets[choice].copy()
                settings.pop("name")  # name 키 제거
                print(f"\n✅ '{self.presets[choice]['name']}' 모드 선택됨")
            
            # 데이터 수집 실행
            success = self.collect_data(settings)
            
            if success:
                # 계속할지 묻기
                continue_choice = input("\n계속 수집하시겠습니까? (Y/n): ").strip().lower()
                if continue_choice == 'n':
                    break
            else:
                retry = input("\n다시 시도하시겠습니까? (Y/n): ").strip().lower()
                if retry == 'n':
                    break
        
        print("\n✨ X4M06 데이터 수집기를 종료합니다.")

def main():
    """메인 함수"""
    try:
        collector = EasyDataCollector()
        collector.run()
    except KeyboardInterrupt:
        print("\n\n🛑 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        input("Enter 키를 눌러 종료...")

if __name__ == "__main__":
    main()