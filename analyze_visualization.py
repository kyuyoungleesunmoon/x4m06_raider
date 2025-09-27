import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def analyze_visualization_image():
    """sample_visualization.png 이미지 분석"""
    try:
        # 이미지 로드
        img_path = r'c:\X4M06_레이더\experiment_results\session_20250927_190149\simulation\sample_visualization.png'
        img = mpimg.imread(img_path)
        
        print("📊 sample_visualization.png 이미지 분석 결과")
        print("=" * 60)
        print(f"이미지 크기: {img.shape}")
        print(f"데이터 타입: {img.dtype}")
        print(f"값 범위: {img.min():.3f} ~ {img.max():.3f}")
        
        # 이미지 표시
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Sample Visualization - 레이더 시뮬레이션 결과', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"이미지 분석 중 오류: {e}")
        return False

if __name__ == "__main__":
    analyze_visualization_image()