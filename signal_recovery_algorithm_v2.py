#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X4M06 레이더 재밍 신호 복구 알고리즘
Deep Learning 기반 신호 복구 시스템

작성자: AI Assistant
날짜: 2024-10-04
버전: 2.0 (차원 호환성 개선)
"""

import os
import sys
import numpy as np
import h5py
import json
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 비활성화
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from scipy import signal
from scipy.ndimage import zoom
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정"""
    try:
        # Windows: 맑은 고딕
        font_path = "C:/Windows/Fonts/malgun.ttf"
        if os.path.exists(font_path):
            font_prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
        else:
            # 기본 폰트 사용
            plt.rcParams['font.family'] = 'DejaVu Sans'
        
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except Exception as e:
        print(f"⚠️ 폰트 설정 실패: {e}")
        return False

class SignalRecoveryDataLoader:
    """신호 복구를 위한 데이터 로더"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.signals = {'clean': [], 'jammed': []}
        self.scenarios = []
        self.spectrograms = {'clean': [], 'jammed': []}
        
    def load_stage1_data(self):
        """1단계 실험 데이터 로딩"""
        print("📁 1단계 실험 데이터 로딩 중...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            # 데이터셋 정보 확인
            scenarios = list(f.keys())
            print(f"   발견된 시나리오 수: {len(scenarios)}")
            
            # 각 시나리오 데이터 로딩
            for i, scenario in enumerate(scenarios):
                # 진행률 표시
                if (i + 1) % 50 == 0 or i == len(scenarios) - 1:
                    progress = (i + 1) / len(scenarios) * 100
                    print(f"데이터 로딩: {progress:.1f}% ({i+1}/{len(scenarios)})", end='\r')
                
                scenario_data = f[scenario]
                
                # 깨끗한 신호와 재밍된 신호 추출
                if 'clean_signal' in scenario_data and 'jammed_signal' in scenario_data:
                    clean_signal = scenario_data['clean_signal'][:]
                    jammed_signal = scenario_data['jammed_signal'][:]
                    
                    self.signals['clean'].append(clean_signal)
                    self.signals['jammed'].append(jammed_signal)
                    self.scenarios.append(scenario)
        
        print(f"\n✅ 데이터 로딩 완료: {len(self.signals['clean'])}개 샘플")
        
    def generate_spectrograms(self, nperseg=128, noverlap=64):
        """고정 크기 스펙트로그램 생성"""
        print("🔄 스펙트로그램 생성 중...")
        
        # 목표 크기 설정 (U-Net에 적합한 2의 거듭제곱)
        target_height = 64   # 주파수 빈
        target_width = 64    # 시간 빈
        
        clean_specs = []
        jammed_specs = []
        
        for i, (clean, jammed) in enumerate(zip(self.signals['clean'], self.signals['jammed'])):
            # STFT 계산
            _, _, Zxx_clean = signal.stft(clean, nperseg=nperseg, noverlap=noverlap)
            _, _, Zxx_jammed = signal.stft(jammed, nperseg=nperseg, noverlap=noverlap)
            
            # 로그 진폭 스펙트로그램
            clean_mag = np.log(np.abs(Zxx_clean) + 1e-8)
            jammed_mag = np.log(np.abs(Zxx_jammed) + 1e-8)
            
            # 고정 크기로 리사이징
            clean_resized = self._resize_to_target(clean_mag, target_height, target_width)
            jammed_resized = self._resize_to_target(jammed_mag, target_height, target_width)
            
            clean_specs.append(clean_resized)
            jammed_specs.append(jammed_resized)
        
        self.spectrograms['clean'] = np.array(clean_specs)
        self.spectrograms['jammed'] = np.array(jammed_specs)
        
        print(f"✅ 스펙트로그램 생성 완료")
        print(f"   크기: {self.spectrograms['clean'].shape}")
        
    def _resize_to_target(self, spectrogram, target_height, target_width):
        """스펙트로그램을 목표 크기로 리사이징"""
        current_height, current_width = spectrogram.shape
        
        # 리사이징 비율 계산
        zoom_factors = (
            target_height / current_height,
            target_width / current_width
        )
        
        # 리사이징 수행
        resized = zoom(spectrogram, zoom_factors, order=1)
        
        # 정확한 크기 보장 (부동소수점 오차 방지)
        if resized.shape != (target_height, target_width):
            # 크롭 또는 패딩으로 정확한 크기 맞추기
            h, w = resized.shape
            if h > target_height:
                resized = resized[:target_height, :]
            elif h < target_height:
                pad_h = target_height - h
                resized = np.pad(resized, ((0, pad_h), (0, 0)), mode='edge')
                
            h, w = resized.shape
            if w > target_width:
                resized = resized[:, :target_width]
            elif w < target_width:
                pad_w = target_width - w
                resized = np.pad(resized, ((0, 0), (0, pad_w)), mode='edge')
        
        return resized
        
    def normalize_data(self):
        """데이터 정규화"""
        print("📊 데이터 정규화 중...")
        
        # 전체 데이터의 최솟값과 최댓값 찾기
        all_data = np.concatenate([
            self.spectrograms['clean'].flatten(),
            self.spectrograms['jammed'].flatten()
        ])
        
        self.data_min = np.min(all_data)
        self.data_max = np.max(all_data)
        
        print(f"   데이터 범위: {self.data_min:.3f} ~ {self.data_max:.3f}")
        
        # Min-Max 정규화 [0, 1]
        self.spectrograms['clean'] = (self.spectrograms['clean'] - self.data_min) / (self.data_max - self.data_min)
        self.spectrograms['jammed'] = (self.spectrograms['jammed'] - self.data_min) / (self.data_max - self.data_min)
        
        print("✅ 정규화 완료")
        
    def split_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """데이터 분할"""
        print("🔀 데이터 분할 중...")
        
        X = self.spectrograms['jammed']  # 입력: 재밍된 신호
        y = self.spectrograms['clean']   # 타겟: 깨끗한 신호
        
        # 채널 차원 추가 (batch, height, width, channels)
        X = np.expand_dims(X, axis=-1)
        y = np.expand_dims(y, axis=-1)
        
        # 훈련/테스트 분할
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 훈련/검증 분할
        val_ratio = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state
        )
        
        print("✅ 데이터 분할 완료:")
        print(f"   훈련: {len(self.X_train)}개")
        print(f"   검증: {len(self.X_val)}개")
        print(f"   테스트: {len(self.X_test)}개")
        
        return (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test)

class UNetSignalRecovery:
    """U-Net 기반 신호 복구 모델"""
    
    def __init__(self, input_shape=(64, 64, 1)):
        self.input_shape = input_shape
        self.model = None
        
    def conv_block(self, inputs, filters, dropout_rate=0.1):
        """합성곱 블록"""
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        return x
    
    def encoder_block(self, inputs, filters, dropout_rate=0.1):
        """인코더 블록"""
        x = self.conv_block(inputs, filters, dropout_rate)
        p = layers.MaxPooling2D((2, 2))(x)
        return x, p
    
    def decoder_block(self, inputs, skip_features, filters, dropout_rate=0.1):
        """디코더 블록"""
        x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(inputs)
        x = layers.Concatenate()([x, skip_features])
        x = self.conv_block(x, filters, dropout_rate)
        return x
    
    def build_model(self):
        """U-Net 모델 구축"""
        print("🏗️ U-Net 모델 구축 중...")
        
        inputs = layers.Input(shape=self.input_shape)
        
        # 인코더 (수축 경로)
        s1, p1 = self.encoder_block(inputs, 64)        # 64x64 -> 32x32
        s2, p2 = self.encoder_block(p1, 128)           # 32x32 -> 16x16  
        s3, p3 = self.encoder_block(p2, 256)           # 16x16 -> 8x8
        s4, p4 = self.encoder_block(p3, 512)           # 8x8 -> 4x4
        
        # 바텀넥 (최하위 레벨)
        b1 = self.conv_block(p4, 1024)                 # 4x4
        
        # 디코더 (확장 경로)  
        d1 = self.decoder_block(b1, s4, 512)           # 4x4 -> 8x8
        d2 = self.decoder_block(d1, s3, 256)           # 8x8 -> 16x16
        d3 = self.decoder_block(d2, s2, 128)           # 16x16 -> 32x32
        d4 = self.decoder_block(d3, s1, 64)            # 32x32 -> 64x64
        
        # 출력 레이어
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d4)
        
        self.model = Model(inputs, outputs, name="UNet_SignalRecovery")
        
        print("✅ U-Net 모델 구축 완료")
        print(f"   파라미터 수: {self.model.count_params():,}")
        
        return self.model
    
    def compile_model(self, learning_rate=1e-4):
        """모델 컴파일"""
        print("⚙️ 모델 컴파일 중...")
        
        # 커스텀 손실 함수 (L1 + SSIM)
        def combined_loss(y_true, y_pred):
            l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
            ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
            return l1_loss + 0.5 * ssim_loss
        
        # 옵티마이저와 메트릭 설정
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=['mae', 'mse']
        )
        
        print("✅ 모델 컴파일 완료")
        
    def train(self, train_data, val_data, epochs=50, batch_size=16):
        """모델 훈련"""
        print("🚀 모델 훈련 시작...")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # 콜백 설정
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'best_signal_recovery_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # 훈련 실행
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        print("✅ 모델 훈련 완료")
        return history
    
    def evaluate(self, test_data):
        """모델 평가"""
        print("📊 모델 평가 중...")
        
        X_test, y_test = test_data
        
        # 예측 수행
        y_pred = self.model.predict(X_test, verbose=0)
        
        # 메트릭 계산
        mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
        mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
        
        # SSIM 계산 (각 이미지별로)
        ssim_scores = []
        for i in range(len(y_test)):
            score = ssim(
                y_test[i].squeeze(),
                y_pred[i].squeeze(),
                data_range=1.0
            )
            ssim_scores.append(score)
        
        avg_ssim = np.mean(ssim_scores)
        
        print(f"✅ 평가 완료:")
        print(f"   MSE: {mse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   SSIM: {avg_ssim:.4f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'ssim': avg_ssim,
            'predictions': y_pred
        }

class SignalRecoveryVisualizer:
    """신호 복구 결과 시각화"""
    
    def __init__(self, output_dir="recovery_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        setup_korean_font()
        
    def plot_training_history(self, history, save_path=None):
        """훈련 히스토리 시각화"""
        if save_path is None:
            save_path = os.path.join(self.output_dir, "Graph_4_training_history.png")
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 손실 함수
        axes[0,0].plot(history.history['loss'], label='훈련 손실', linewidth=2)
        axes[0,0].plot(history.history['val_loss'], label='검증 손실', linewidth=2)
        axes[0,0].set_title('Graph 4-1: 모델 손실 함수', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('에포크')
        axes[0,0].set_ylabel('손실')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # MAE
        axes[0,1].plot(history.history['mae'], label='훈련 MAE', linewidth=2)
        axes[0,1].plot(history.history['val_mae'], label='검증 MAE', linewidth=2)
        axes[0,1].set_title('Graph 4-2: 평균 절대 오차 (MAE)', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('에포크')
        axes[0,1].set_ylabel('MAE')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # MSE
        axes[1,0].plot(history.history['mse'], label='훈련 MSE', linewidth=2)
        axes[1,0].plot(history.history['val_mse'], label='검증 MSE', linewidth=2)
        axes[1,0].set_title('Graph 4-3: 평균 제곱 오차 (MSE)', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('에포크')
        axes[1,0].set_ylabel('MSE')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 학습률 (있는 경우)
        if 'lr' in history.history:
            axes[1,1].plot(history.history['lr'], linewidth=2, color='orange')
            axes[1,1].set_title('Graph 4-4: 학습률 변화', fontsize=14, fontweight='bold')
            axes[1,1].set_xlabel('에포크')
            axes[1,1].set_ylabel('학습률')
            axes[1,1].set_yscale('log')
        else:
            axes[1,1].text(0.5, 0.5, '학습률 정보 없음', ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Graph 4-4: 학습률 정보', fontsize=14, fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 훈련 히스토리 저장: {save_path}")
        
    def plot_recovery_examples(self, X_test, y_test, y_pred, n_examples=6, save_path=None):
        """복구 결과 예시 시각화"""
        if save_path is None:
            save_path = os.path.join(self.output_dir, "Graph_5_recovery_examples.png")
            
        fig, axes = plt.subplots(3, n_examples, figsize=(20, 12))
        
        # 랜덤하게 예시 선택
        indices = np.random.choice(len(X_test), n_examples, replace=False)
        
        for i, idx in enumerate(indices):
            # 원본 재밍된 신호
            im1 = axes[0, i].imshow(X_test[idx].squeeze(), aspect='auto', cmap='viridis')
            axes[0, i].set_title(f'재밍된 신호 #{idx+1}', fontsize=10)
            axes[0, i].set_ylabel('주파수 빈')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            # 복구된 신호
            im2 = axes[1, i].imshow(y_pred[idx].squeeze(), aspect='auto', cmap='viridis')
            axes[1, i].set_title(f'복구된 신호 #{idx+1}', fontsize=10)
            axes[1, i].set_ylabel('주파수 빈')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            # 실제 깨끗한 신호
            im3 = axes[2, i].imshow(y_test[idx].squeeze(), aspect='auto', cmap='viridis')
            axes[2, i].set_title(f'실제 깨끗한 신호 #{idx+1}', fontsize=10)
            axes[2, i].set_ylabel('주파수 빈')
            axes[2, i].set_xlabel('시간 빈')
            plt.colorbar(im3, ax=axes[2, i], fraction=0.046, pad=0.04)
        
        plt.suptitle('Graph 5: 신호 복구 결과 예시', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 복구 예시 저장: {save_path}")
        
    def plot_performance_metrics(self, metrics, save_path=None):
        """성능 메트릭 시각화"""
        if save_path is None:
            save_path = os.path.join(self.output_dir, "Graph_6_performance_metrics.png")
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # MSE 히스토그램
        axes[0].bar(['MSE'], [metrics['mse']], color='skyblue', alpha=0.8)
        axes[0].set_title('Graph 6-1: 평균 제곱 오차 (MSE)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('MSE 값')
        axes[0].grid(True, alpha=0.3)
        
        # MAE 히스토그램  
        axes[1].bar(['MAE'], [metrics['mae']], color='lightcoral', alpha=0.8)
        axes[1].set_title('Graph 6-2: 평균 절대 오차 (MAE)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('MAE 값')
        axes[1].grid(True, alpha=0.3)
        
        # SSIM 히스토그램
        axes[2].bar(['SSIM'], [metrics['ssim']], color='lightgreen', alpha=0.8)
        axes[2].set_title('Graph 6-3: 구조적 유사도 (SSIM)', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('SSIM 값')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)
        
        # 성능 수치 텍스트 추가
        for i, (metric, value) in enumerate([('MSE', metrics['mse']), ('MAE', metrics['mae']), ('SSIM', metrics['ssim'])]):
            axes[i].text(0, value * 0.5, f'{value:.4f}', ha='center', va='bottom', 
                        fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 성능 메트릭 저장: {save_path}")

def main():
    """메인 실행 함수"""
    try:
        print("🚀 재밍 신호 복구 알고리즘 시작")
        print("=" * 60)
        
        # 1단계 데이터 경로 자동 감지
        result_dirs = [d for d in os.listdir('.') if d.startswith('stage1_results_')]
        if not result_dirs:
            raise FileNotFoundError("1단계 실험 결과를 찾을 수 없습니다.")
            
        latest_dir = sorted(result_dirs)[-1]
        data_path = os.path.join(latest_dir, 'stage1_signals.h5')
        
        print(f"📁 1단계 데이터 경로: {data_path}")
        
        # 데이터 로더 초기화 및 로딩
        data_loader = SignalRecoveryDataLoader(data_path)
        data_loader.load_stage1_data()
        data_loader.generate_spectrograms()
        data_loader.normalize_data()
        
        # 데이터 분할
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.split_data()
        
        # U-Net 모델 생성 및 훈련
        unet = UNetSignalRecovery(input_shape=X_train.shape[1:])
        model = unet.build_model()
        unet.compile_model()
        
        # 모델 훈련
        history = unet.train(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            epochs=30,  # 시간 단축을 위해 30 에포크
            batch_size=16
        )
        
        # 모델 평가
        metrics = unet.evaluate((X_test, y_test))
        
        # 결과 시각화
        visualizer = SignalRecoveryVisualizer()
        visualizer.plot_training_history(history)
        visualizer.plot_recovery_examples(X_test, y_test, metrics['predictions'])
        visualizer.plot_performance_metrics(metrics)
        
        # 결과 요약 저장
        summary = {
            'model_performance': {
                'mse': float(metrics['mse']),
                'mae': float(metrics['mae']),
                'ssim': float(metrics['ssim'])
            },
            'data_info': {
                'total_samples': len(data_loader.signals['clean']),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'spectrogram_shape': list(X_train.shape[1:])
            },
            'model_info': {
                'total_parameters': int(model.count_params()),
                'input_shape': list(unet.input_shape),
                'architecture': 'U-Net'
            }
        }
        
        summary_path = os.path.join(visualizer.output_dir, 'recovery_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print("🎉 신호 복구 알고리즘 완료!")
        print(f"📊 최종 성능:")
        print(f"   MSE: {metrics['mse']:.6f}")
        print(f"   MAE: {metrics['mae']:.6f}")
        print(f"   SSIM: {metrics['ssim']:.4f}")
        print(f"💾 결과 저장 경로: {visualizer.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()