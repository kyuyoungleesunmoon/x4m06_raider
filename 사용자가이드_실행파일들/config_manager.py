#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X4M06 설정 관리자
레이더 파라미터 및 시스템 설정 관리
"""

import json
import os
import sys

class ConfigManager:
    """설정 관리 클래스"""
    
    def __init__(self):
        self.config_file = "x4m06_config.json"
        self.default_config = {
            "hardware": {
                "com_port": "COM3",
                "device_name": "X4M06"
            },
            "radar_settings": {
                "dac_min": 900,
                "dac_max": 1400,
                "iterations": 16,
                "pulses_per_step": 26,
                "fps": 20,
                "detection_range": "5m"
            },
            "collection_presets": {
                "quick_test": {"duration": 30, "frames": 100, "plot": False},
                "standard": {"duration": 120, "frames": 500, "plot": True},
                "long_term": {"duration": 300, "frames": 1000, "plot": False},
                "monitoring": {"duration": 60, "frames": 300, "plot": True}
            },
            "data_settings": {
                "save_directory": "collected_data",
                "auto_backup": True,
                "compression": True
            }
        }
        self.config = self.load_config()
    
    def load_config(self):
        """설정 파일 로드"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 설정 파일 로드 실패: {e}")
                return self.default_config.copy()
        else:
            return self.default_config.copy()
    
    def save_config(self):
        """설정 파일 저장"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print("✅ 설정이 저장되었습니다.")
            return True
        except Exception as e:
            print(f"❌ 설정 저장 실패: {e}")
            return False
    
    def print_header(self):
        """헤더 출력"""
        print("=" * 60)
        print("⚙️ X4M06 레이더 설정 관리자")
        print("   레이더 파라미터 및 시스템 설정")
        print("=" * 60)
    
    def show_main_menu(self):
        """메인 메뉴 표시"""
        print("\n📋 설정 메뉴:")
        print("─" * 40)
        print("  1. 하드웨어 설정")
        print("  2. 레이더 파라미터 설정")
        print("  3. 수집 프리셋 관리")
        print("  4. 데이터 저장 설정")
        print("  5. 현재 설정 보기")
        print("  6. 기본값으로 복원")
        print("  7. 설정 내보내기/가져오기")
        print("─" * 40)
        print("  0. 종료")
        print()
    
    def hardware_settings(self):
        """하드웨어 설정"""
        print("\n🔧 하드웨어 설정:")
        print(f"현재 COM 포트: {self.config['hardware']['com_port']}")
        print(f"장치 이름: {self.config['hardware']['device_name']}")
        
        new_port = input(f"새 COM 포트 (현재: {self.config['hardware']['com_port']}): ").strip()
        if new_port:
            self.config['hardware']['com_port'] = new_port
            print(f"✅ COM 포트가 {new_port}로 변경되었습니다.")
    
    def radar_settings(self):
        """레이더 파라미터 설정"""
        print("\n🎯 레이더 파라미터 설정:")
        radar = self.config['radar_settings']
        
        print(f"현재 설정:")
        print(f"  DAC 범위: {radar['dac_min']} - {radar['dac_max']}")
        print(f"  반복 횟수: {radar['iterations']}")
        print(f"  펄스/스텝: {radar['pulses_per_step']}")
        print(f"  FPS: {radar['fps']}")
        print(f"  탐지 범위: {radar['detection_range']}")
        
        print("\n🎛️ 탐지 범위 프리셋:")
        print("  1. 근거리 (0-2m) - DAC 900-1000")
        print("  2. 중거리 (0-5m) - DAC 900-1400 [기본값]")
        print("  3. 장거리 (0-10m) - DAC 900-1800")
        print("  4. 사용자 정의")
        
        choice = input("선택하세요 (1-4, Enter=유지): ").strip()
        
        if choice == "1":
            radar['dac_min'] = 900
            radar['dac_max'] = 1000
            radar['detection_range'] = "2m"
            print("✅ 근거리 모드로 설정됨")
        elif choice == "2":
            radar['dac_min'] = 900
            radar['dac_max'] = 1400
            radar['detection_range'] = "5m"
            print("✅ 중거리 모드로 설정됨")
        elif choice == "3":
            radar['dac_min'] = 900
            radar['dac_max'] = 1800
            radar['detection_range'] = "10m"
            print("✅ 장거리 모드로 설정됨")
        elif choice == "4":
            try:
                new_min = int(input(f"DAC 최소값 (현재: {radar['dac_min']}): ") or radar['dac_min'])
                new_max = int(input(f"DAC 최대값 (현재: {radar['dac_max']}): ") or radar['dac_max'])
                new_fps = int(input(f"FPS (현재: {radar['fps']}): ") or radar['fps'])
                
                radar['dac_min'] = new_min
                radar['dac_max'] = new_max
                radar['fps'] = new_fps
                radar['detection_range'] = "사용자정의"
                print("✅ 사용자 정의 설정 적용됨")
            except ValueError:
                print("⚠️ 잘못된 입력입니다. 기존 설정을 유지합니다.")
    
    def preset_management(self):
        """수집 프리셋 관리"""
        print("\n📊 수집 프리셋 관리:")
        presets = self.config['collection_presets']
        
        print("현재 프리셋:")
        for name, preset in presets.items():
            plot_str = "플롯 ON" if preset['plot'] else "플롯 OFF"
            print(f"  {name}: {preset['duration']}초, {preset['frames']}프레임, {plot_str}")
        
        print("\n1. 프리셋 수정")
        print("2. 새 프리셋 추가")
        print("3. 프리셋 삭제")
        
        choice = input("선택하세요 (1-3, Enter=뒤로): ").strip()
        
        if choice == "1":
            preset_name = input("수정할 프리셋 이름: ").strip()
            if preset_name in presets:
                self.edit_preset(preset_name)
            else:
                print("❌ 존재하지 않는 프리셋입니다.")
        elif choice == "2":
            self.add_preset()
        elif choice == "3":
            preset_name = input("삭제할 프리셋 이름: ").strip()
            if preset_name in presets and preset_name not in ["quick_test", "standard"]:
                del presets[preset_name]
                print(f"✅ '{preset_name}' 프리셋이 삭제되었습니다.")
            else:
                print("❌ 기본 프리셋은 삭제할 수 없습니다.")
    
    def edit_preset(self, preset_name):
        """프리셋 편집"""
        preset = self.config['collection_presets'][preset_name]
        print(f"\n'{preset_name}' 프리셋 편집:")
        
        try:
            new_duration = int(input(f"수집 시간 (현재: {preset['duration']}초): ") or preset['duration'])
            new_frames = int(input(f"최대 프레임 (현재: {preset['frames']}개): ") or preset['frames'])
            new_plot = input(f"실시간 플롯 (현재: {preset['plot']}, Y/n): ").strip().lower()
            
            preset['duration'] = new_duration
            preset['frames'] = new_frames
            if new_plot in ['y', 'yes', '']:
                preset['plot'] = True
            elif new_plot in ['n', 'no']:
                preset['plot'] = False
            
            print(f"✅ '{preset_name}' 프리셋이 수정되었습니다.")
        except ValueError:
            print("⚠️ 잘못된 입력입니다.")
    
    def add_preset(self):
        """새 프리셋 추가"""
        print("\n새 프리셋 추가:")
        name = input("프리셋 이름: ").strip()
        
        if not name:
            print("❌ 이름을 입력해주세요.")
            return
        
        if name in self.config['collection_presets']:
            print("❌ 이미 존재하는 프리셋 이름입니다.")
            return
        
        try:
            duration = int(input("수집 시간 (초): ") or "60")
            frames = int(input("최대 프레임 수: ") or "300")
            plot = input("실시간 플롯 (Y/n): ").strip().lower() != 'n'
            
            self.config['collection_presets'][name] = {
                "duration": duration,
                "frames": frames,
                "plot": plot
            }
            print(f"✅ '{name}' 프리셋이 추가되었습니다.")
        except ValueError:
            print("❌ 잘못된 입력입니다.")
    
    def data_settings(self):
        """데이터 저장 설정"""
        print("\n💾 데이터 저장 설정:")
        data = self.config['data_settings']
        
        print(f"현재 설정:")
        print(f"  저장 디렉토리: {data['save_directory']}")
        print(f"  자동 백업: {data['auto_backup']}")
        print(f"  압축 저장: {data['compression']}")
        
        new_dir = input(f"저장 디렉토리 (현재: {data['save_directory']}): ").strip()
        if new_dir:
            data['save_directory'] = new_dir
            print(f"✅ 저장 디렉토리가 '{new_dir}'로 변경되었습니다.")
        
        backup = input("자동 백업 (Y/n): ").strip().lower()
        if backup == 'n':
            data['auto_backup'] = False
        elif backup in ['y', 'yes']:
            data['auto_backup'] = True
    
    def show_current_config(self):
        """현재 설정 보기"""
        print("\n📋 현재 설정:")
        print("─" * 50)
        print(json.dumps(self.config, indent=2, ensure_ascii=False))
    
    def reset_to_default(self):
        """기본값으로 복원"""
        confirm = input("⚠️ 모든 설정을 기본값으로 복원하시겠습니까? (y/N): ").strip().lower()
        if confirm == 'y':
            self.config = self.default_config.copy()
            print("✅ 설정이 기본값으로 복원되었습니다.")
        else:
            print("❌ 복원이 취소되었습니다.")
    
    def export_import_config(self):
        """설정 내보내기/가져오기"""
        print("\n📤📥 설정 내보내기/가져오기:")
        print("  1. 설정 내보내기")
        print("  2. 설정 가져오기")
        
        choice = input("선택하세요 (1-2): ").strip()
        
        if choice == "1":
            filename = input("내보낼 파일명 (기본: config_backup.json): ").strip() or "config_backup.json"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                print(f"✅ 설정이 '{filename}'으로 내보내졌습니다.")
            except Exception as e:
                print(f"❌ 내보내기 실패: {e}")
        
        elif choice == "2":
            filename = input("가져올 파일명: ").strip()
            if os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        imported_config = json.load(f)
                    self.config = imported_config
                    print(f"✅ '{filename}'에서 설정을 가져왔습니다.")
                except Exception as e:
                    print(f"❌ 가져오기 실패: {e}")
            else:
                print("❌ 파일이 존재하지 않습니다.")
    
    def run(self):
        """메인 실행 루프"""
        self.print_header()
        
        while True:
            self.show_main_menu()
            choice = input("선택하세요 (0-7): ").strip()
            
            if choice == "0":
                # 종료 전 저장 확인
                save = input("변경사항을 저장하시겠습니까? (Y/n): ").strip().lower()
                if save != 'n':
                    self.save_config()
                print("👋 설정 관리자를 종료합니다.")
                break
            elif choice == "1":
                self.hardware_settings()
            elif choice == "2":
                self.radar_settings()
            elif choice == "3":
                self.preset_management()
            elif choice == "4":
                self.data_settings()
            elif choice == "5":
                self.show_current_config()
            elif choice == "6":
                self.reset_to_default()
            elif choice == "7":
                self.export_import_config()
            else:
                print("❌ 잘못된 선택입니다.")

def main():
    """메인 함수"""
    try:
        manager = ConfigManager()
        manager.run()
    except KeyboardInterrupt:
        print("\n\n🛑 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        input("Enter 키를 눌러 종료...")

if __name__ == "__main__":
    main()