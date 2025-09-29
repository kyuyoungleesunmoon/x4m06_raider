#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X4M06 저장된 데이터 로더 및 분석기
수집된 하드웨어 데이터를 로드하고 기본 분석을 제공합니다.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

class X4M06DataLoader:
    """X4M06 저장 데이터 로더"""
    
    def __init__(self, data_dir="collected_data"):
        self.data_dir = data_dir
        self.session_data = None
        self.frames_data = None
        
    def list_sessions(self):
        """저장된 세션 목록 반환"""
        json_files = glob.glob(os.path.join(self.data_dir, "x4m06_data_*.json"))
        sessions = []
        
        for json_file in json_files:
            basename = os.path.basename(json_file)
            session_id = basename.replace("x4m06_data_", "").replace(".json", "")
            sessions.append({
                'session_id': session_id,
                'json_file': json_file,
                'npz_file': json_file.replace('.json', '.npz'),
                'summary_file': os.path.join(self.data_dir, f"summary_{session_id}.txt")
            })
        
        return sessions
    
    def load_session(self, session_id=None):
        """세션 데이터 로드"""
        sessions = self.list_sessions()
        
        if not sessions:
            print("❌ 저장된 세션이 없습니다")
            return False
        
        # 세션 ID가 지정되지 않으면 가장 최근 세션 사용
        if session_id is None:
            session = sessions[-1]  # 가장 최근 파일
            session_id = session['session_id']
        else:
            session = next((s for s in sessions if s['session_id'] == session_id), None)
            if not session:
                print(f"❌ 세션 {session_id}을 찾을 수 없습니다")
                return False
        
        try:
            # JSON 데이터 로드
            print(f"📂 세션 로드 중: {session_id}")
            with open(session['json_file'], 'r', encoding='utf-8') as f:
                self.session_data = json.load(f)
            
            # NumPy 데이터 로드 (더 빠른 분석용)
            if os.path.exists(session['npz_file']):
                npz_data = np.load(session['npz_file'])
                self.frames_data = npz_data['frames']
                self.timestamps = npz_data['timestamps']
                self.frame_numbers = npz_data['frame_numbers']
            
            print(f"✓ 로드 완료: {len(self.session_data['frames'])}개 프레임")
            return True
            
        except Exception as e:
            print(f"❌ 로드 실패: {e}")
            return False
    
    def print_session_info(self):
        """세션 정보 출력"""
        if not self.session_data:
            print("❌ 로드된 세션이 없습니다")
            return
        
        info = self.session_data['session_info']
        print(f"\n📊 === 세션 정보 ===")
        print(f"세션 ID: {info['session_id']}")
        print(f"장치: {info['device']}")
        print(f"수집 시간: {info['collection_time']}")
        print(f"총 프레임: {info['total_frames']}개")
        print(f"\n🎯 레이더 설정:")
        for key, value in info['settings'].items():
            print(f"  {key}: {value}")
        
        if self.frames_data is not None:
            print(f"\n📈 데이터 통계:")
            print(f"  프레임 크기: {self.frames_data.shape}")
            print(f"  전체 최대값: {np.max(self.frames_data):.6f}")
            print(f"  전체 최소값: {np.min(self.frames_data):.6f}")
            print(f"  전체 평균값: {np.mean(self.frames_data):.6f}")
            print(f"  표준편차: {np.std(self.frames_data):.6f}")
    
    def plot_frame_analysis(self, frame_indices=None, save_plot=False):
        """프레임 분석 플롯"""
        if self.frames_data is None:
            print("❌ 로드된 데이터가 없습니다")
            return
        
        if frame_indices is None:
            # 기본적으로 첫 번째, 중간, 마지막 프레임
            total_frames = len(self.frames_data)
            frame_indices = [0, total_frames//2, total_frames-1]
        
        plt.figure(figsize=(15, 10))
        
        # 개별 프레임 플롯
        plt.subplot(2, 2, 1)
        for i, frame_idx in enumerate(frame_indices):
            plt.plot(self.frames_data[frame_idx], label=f'프레임 {frame_idx}', alpha=0.7)
        plt.title('개별 프레임 비교')
        plt.xlabel('샘플 인덱스')
        plt.ylabel('신호 강도')
        plt.legend()
        plt.grid(True)
        
        # 전체 프레임 히트맵
        plt.subplot(2, 2, 2)
        # 처음 100프레임만 히트맵으로 표시 (시각화를 위해)
        display_frames = min(100, len(self.frames_data))
        plt.imshow(self.frames_data[:display_frames], aspect='auto', cmap='viridis')
        plt.title(f'프레임 히트맵 (처음 {display_frames}프레임)')
        plt.xlabel('샘플 인덱스')
        plt.ylabel('프레임 번호')
        plt.colorbar(label='신호 강도')
        
        # 시간별 최대값 변화
        plt.subplot(2, 2, 3)
        max_values = np.max(self.frames_data, axis=1)
        plt.plot(max_values)
        plt.title('시간별 최대 신호 강도 변화')
        plt.xlabel('프레임 번호')
        plt.ylabel('최대 신호 강도')
        plt.grid(True)
        
        # FFT 분석 (첫 번째 프레임)
        plt.subplot(2, 2, 4)
        fft_data = np.fft.fft(self.frames_data[0])
        freqs = np.fft.fftfreq(len(fft_data))
        plt.plot(freqs[:len(freqs)//2], np.abs(fft_data[:len(fft_data)//2]))
        plt.title('첫 번째 프레임 FFT')
        plt.xlabel('정규화 주파수')
        plt.ylabel('크기')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plot_filename = f"analysis_{self.session_data['session_info']['session_id']}.png"
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            print(f"✓ 플롯 저장: {plot_filename}")
        
        plt.show()
    
    def detect_motion(self, threshold_factor=3.0):
        """모션 감지 분석"""
        if self.frames_data is None:
            print("❌ 로드된 데이터가 없습니다")
            return []
        
        print(f"\n🎯 모션 감지 분석 (임계값 계수: {threshold_factor})")
        
        # 각 프레임의 표준편차 계산 (움직임 지표)
        frame_stds = np.std(self.frames_data, axis=1)
        
        # 임계값 설정
        mean_std = np.mean(frame_stds)
        std_std = np.std(frame_stds)
        threshold = mean_std + threshold_factor * std_std
        
        # 모션 감지
        motion_frames = np.where(frame_stds > threshold)[0]
        
        print(f"  기준 표준편차: {mean_std:.6f}")
        print(f"  표준편차의 표준편차: {std_std:.6f}")
        print(f"  임계값: {threshold:.6f}")
        print(f"  감지된 모션 프레임: {len(motion_frames)}개")
        
        if len(motion_frames) > 0:
            print(f"  모션 프레임 번호: {motion_frames[:10]}..." if len(motion_frames) > 10 else f"  모션 프레임 번호: {motion_frames}")
        
        return motion_frames
    
    def export_to_csv(self, output_file=None):
        """CSV로 내보내기"""
        if self.frames_data is None:
            print("❌ 로드된 데이터가 없습니다")
            return False
        
        if output_file is None:
            output_file = f"exported_{self.session_data['session_info']['session_id']}.csv"
        
        try:
            # 프레임별 통계를 CSV로 저장
            frame_stats = []
            for i, frame in enumerate(self.frames_data):
                frame_stats.append([
                    i,  # 프레임 번호
                    self.timestamps[i] if hasattr(self, 'timestamps') else i,  # 타임스탬프
                    np.max(frame),  # 최대값
                    np.min(frame),  # 최소값
                    np.mean(frame), # 평균값
                    np.std(frame)   # 표준편차
                ])
            
            np.savetxt(output_file, frame_stats, delimiter=',', 
                      header='frame_number,timestamp,max_value,min_value,mean_value,std_value',
                      comments='')
            
            print(f"✓ CSV 내보내기 완료: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ CSV 내보내기 실패: {e}")
            return False

def main():
    """메인 실행 함수"""
    print("📊 X4M06 저장 데이터 분석기")
    
    loader = X4M06DataLoader()
    
    # 세션 목록 표시
    sessions = loader.list_sessions()
    if not sessions:
        print("❌ 저장된 세션이 없습니다")
        return
    
    print(f"\n📁 저장된 세션 ({len(sessions)}개):")
    for i, session in enumerate(sessions):
        print(f"  {i+1}. {session['session_id']}")
    
    # 가장 최근 세션 로드
    if loader.load_session():
        loader.print_session_info()
        
        # 사용자 선택
        print(f"\n🎮 분석 옵션:")
        print("  1. 프레임 분석 플롯")
        print("  2. 모션 감지")
        print("  3. CSV 내보내기")
        print("  Enter: 모든 분석 실행")
        
        choice = input("\n선택하세요 (1-3 또는 Enter): ").strip()
        
        if choice == "1" or choice == "":
            loader.plot_frame_analysis(save_plot=True)
        
        if choice == "2" or choice == "":
            loader.detect_motion()
        
        if choice == "3" or choice == "":
            loader.export_to_csv()
        
        print(f"\n✨ 분석 완료!")

if __name__ == "__main__":
    main()