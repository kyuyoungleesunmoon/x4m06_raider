#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X4M06 시스템 론처
사용자를 위한 통합 실행 인터페이스
"""

import os
import sys

def print_banner():
    """배너 출력"""
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + " " * 12 + "🎯 X4M06 UWB 레이더 시스템" + " " * 13 + "║")
    print("║" + " " * 17 + "데이터 수집 및 분석 도구" + " " * 17 + "║")
    print("║" + " " * 58 + "║")
    print("║" + " " * 20 + "Version 1.0 Final" + " " * 21 + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")

def check_environment():
    """환경 확인"""
    print("\n🔍 시스템 환경 확인...")
    
    # Python 버전 확인
    if sys.version_info.major == 3 and sys.version_info.minor == 6:
        print("✅ Python 3.6 환경 확인됨")
    else:
        print(f"⚠️ Python 버전: {sys.version_info.major}.{sys.version_info.minor} (권장: 3.6)")
    
    # 필요한 파일들 확인
    required_files = [
        "easy_data_collector.py",
        "x4m06_data_saver.py", 
        "config_manager.py",
        "hardware_data_analyzer.py"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - 누락됨")
            missing_files.append(file)
    
    # collected_data 디렉토리 확인
    if not os.path.exists("collected_data"):
        os.makedirs("collected_data")
        print("📁 collected_data 디렉토리 생성됨")
    else:
        print("✅ collected_data 디렉토리 존재함")
    
    return len(missing_files) == 0

def show_main_menu():
    """메인 메뉴 표시"""
    print("\n" + "─" * 60)
    print("📋 메인 메뉴")
    print("─" * 60)
    print("  🚀 1. 간편 데이터 수집기 (권장)")
    print("     └─ 초보자용 메뉴 방식 데이터 수집")
    print()
    print("  🔧 2. 고급 데이터 수집기")
    print("     └─ 상세 설정 가능한 수집 도구")
    print()
    print("  ⚙️  3. 설정 관리자")
    print("     └─ 레이더 파라미터 및 시스템 설정")
    print()
    print("  📊 4. 데이터 분석기")
    print("     └─ 수집된 데이터 시각화 및 분석")
    print()
    print("  🔍 5. 빠른 연결 테스트")
    print("     └─ 레이더 연결 상태 확인")
    print()
    print("  📖 6. 사용 가이드")
    print("     └─ 상세한 사용 방법 안내")
    print("─" * 60)
    print("  0. 종료")
    print()

def run_quick_test():
    """빠른 연결 테스트"""
    print("\n🔍 레이더 연결 테스트 중...")
    try:
        os.system("python quick_test.py")
    except Exception as e:
        print(f"❌ 테스트 실행 실패: {e}")

def show_user_guide():
    """사용 가이드 표시"""
    guide = """
╔══════════════════════════════════════════════════════════════╗
║                        📖 사용 가이드                         ║
╚══════════════════════════════════════════════════════════════╝

🚀 처음 사용하시는 분:
  1. X4M06 레이더를 USB에 연결
  2. '1. 간편 데이터 수집기' 선택
  3. '빠른 테스트' 모드로 시작
  4. 수집 완료 후 데이터 분석기 실행

⚙️ 설정 변경이 필요한 경우:
  1. '3. 설정 관리자' 실행
  2. 탐지 범위 조정 (근거리/중거리/장거리)
  3. 수집 프리셋 커스터마이징
  4. COM 포트 변경 (필요시)

📊 수집된 데이터 분석:
  1. '4. 데이터 분석기' 실행
  2. 세션 선택 후 시각화
  3. 모션 감지 분석
  4. CSV 내보내기

🔧 고급 사용자:
  1. '2. 고급 데이터 수집기' 사용
  2. 실시간 플롯 활성화
  3. 커스텀 수집 시간/프레임 설정

💾 데이터 저장 위치:
  - collected_data/ 폴더에 자동 저장
  - 세션별로 JSON, NPZ, TXT 파일 생성
  - 백업 권장

⚠️ 문제 해결:
  - COM 포트 확인: 장치 관리자에서 포트 번호 확인
  - 연결 실패: USB 케이블 교체, 다른 프로그램 종료
  - 권한 문제: 관리자 권한으로 실행

📞 지원:
  - 로그 파일: logs/ 폴더 확인
  - 설정 백업: config_manager.py에서 내보내기
  - 샘플 데이터: collected_data/ 폴더 참조
"""
    print(guide)
    input("\n📖 가이드를 읽으셨으면 Enter 키를 눌러 메뉴로 돌아가세요...")

def main():
    """메인 함수"""
    print_banner()
    
    # 환경 확인
    if not check_environment():
        print("\n❌ 필수 파일이 누락되었습니다. 설치를 확인해주세요.")
        input("Enter 키를 눌러 종료...")
        return
    
    print("\n✅ 모든 시스템 구성 요소가 준비되었습니다!")
    
    while True:
        try:
            show_main_menu()
            choice = input("선택하세요 (0-6): ").strip()
            
            if choice == "0":
                print("\n👋 X4M06 시스템을 종료합니다.")
                print("수집하신 데이터는 collected_data/ 폴더에 저장되어 있습니다.")
                break
            
            elif choice == "1":
                print("\n🚀 간편 데이터 수집기를 실행합니다...")
                os.system("python easy_data_collector.py")
            
            elif choice == "2":
                print("\n🔧 고급 데이터 수집기를 실행합니다...")
                os.system("python x4m06_data_saver.py")
            
            elif choice == "3":
                print("\n⚙️ 설정 관리자를 실행합니다...")
                os.system("python config_manager.py")
            
            elif choice == "4":
                print("\n📊 데이터 분석기를 실행합니다...")
                os.system("python hardware_data_analyzer.py")
            
            elif choice == "5":
                run_quick_test()
            
            elif choice == "6":
                show_user_guide()
            
            else:
                print("❌ 잘못된 선택입니다. 0-6 중에서 선택해주세요.")
            
        except KeyboardInterrupt:
            print("\n\n🛑 사용자에 의해 중단되었습니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            input("Enter 키를 눌러 계속...")

if __name__ == "__main__":
    main()