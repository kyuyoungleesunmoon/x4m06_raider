#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
재밍 신호 복구 알고리즘 (Deep Learning Based Signal Recovery)
1단계 시뮬레이션 데이터를 활용한 U-Net 기반 신호 복원

주요 기능:
1. HDF5 데이터 로딩 및 전처리
2. U-Net 아키텍처 구현
3. 스펙트로그램 기반 신호 복원
4. 성능 평가 및 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from tqdm import tqdm
import time
from datetime import datetime
import os

# 한글 폰트 설정
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

class SignalRecoveryDataLoader:
    """재밍 신호 복구를 위한 데이터 로더"""
    
    def __init__(self, data_path):
        """
        Args:
            data_path: 1단계 실험 결과 HDF5 파일 경로
        """
        self.data_path = data_path
        self.signals = {'clean': [], 'jammed': [], 'jamming': []}
        self.parameters = []
        self.metrics = []
        
    def load_data(self):
        """HDF5 파일에서 신호 데이터 로딩"""
        print("📁 1단계 실험 데이터 로딩 중...")
        
        with h5py.File(self.data_path, 'r') as f:
            scenario_keys = [key for key in f.keys() if key.startswith('scenario_')]
            
            print(f"   발견된 시나리오 수: {len(scenario_keys)}")
            
            for key in tqdm(scenario_keys, desc="데이터 로딩"):
                grp = f[key]
                
                # 신호 데이터
                clean_signal = grp['clean_signal'][:]
                jammed_signal = grp['jammed_signal'][:]
                jamming_signal = grp['jamming_signal'][:]
                
                self.signals['clean'].append(clean_signal)
                self.signals['jammed'].append(jammed_signal)
                self.signals['jamming'].append(jamming_signal)
                
                # 파라미터
                params = {
                    'power_ratio': grp.attrs['power_ratio'],
                    'freq_offset': grp.attrs['freq_offset'], 
                    'time_delay': grp.attrs['time_delay'],
                    'chirp_slope_ratio': grp.attrs['chirp_slope_ratio']
                }
                self.parameters.append(params)
                
                # 성능 지표
                metrics = {
                    'snr_db': grp.attrs['snr_db'],
                    'correlation': grp.attrs['correlation'],
                    'peak_shift': grp.attrs['peak_shift']
                }
                self.metrics.append(metrics)
        
        # numpy 배열로 변환
        for key in self.signals.keys():
            self.signals[key] = np.array(self.signals[key])
            
        print(f"✅ 데이터 로딩 완료: {len(self.signals['clean'])}개 샘플")
        
    def generate_spectrograms(self, nperseg=64, noverlap=32):
        """신호를 스펙트로그램으로 변환"""
        print("🔄 스펙트로그램 생성 중...")
        
        self.spectrograms = {'clean': [], 'jammed': []}
        
        for i, (clean, jammed) in enumerate(zip(self.signals['clean'], self.signals['jammed'])):
            # 깨끗한 신호 스펙트로그램
            f_clean, t_clean, Zxx_clean = signal.stft(
                clean, nperseg=nperseg, noverlap=noverlap
            )
            
            # 재밍된 신호 스펙트로그램  
            f_jammed, t_jammed, Zxx_jammed = signal.stft(
                jammed, nperseg=nperseg, noverlap=noverlap
            )
            
            # 진폭 스펙트로그램 (로그 스케일)
            clean_mag = np.log(np.abs(Zxx_clean) + 1e-8)
            jammed_mag = np.log(np.abs(Zxx_jammed) + 1e-8)
            
            self.spectrograms['clean'].append(clean_mag)
            self.spectrograms['jammed'].append(jammed_mag)
        
        self.spectrograms['clean'] = np.array(self.spectrograms['clean'])
        self.spectrograms['jammed'] = np.array(self.spectrograms['jammed'])
        
        # 주파수/시간 축 저장
        self.freq_axis = f_clean
        self.time_axis = t_clean
        
        print(f"✅ 스펙트로그램 생성 완료")
        print(f"   크기: {self.spectrograms['clean'].shape}")
        
        return self.spectrograms
    
    def normalize_data(self):
        """데이터 정규화 (0-1 범위)"""
        print("📊 데이터 정규화 중...")
        
        # 전체 데이터에서 min/max 계산
        all_data = np.concatenate([
            self.spectrograms['clean'].flatten(),
            self.spectrograms['jammed'].flatten()
        ])
        
        self.data_min = np.min(all_data)
        self.data_max = np.max(all_data)
        
        print(f"   데이터 범위: {self.data_min:.3f} ~ {self.data_max:.3f}")
        
        # 정규화 적용
        self.spectrograms['clean'] = (self.spectrograms['clean'] - self.data_min) / (self.data_max - self.data_min)
        self.spectrograms['jammed'] = (self.spectrograms['jammed'] - self.data_min) / (self.data_max - self.data_min)
        
        print("✅ 정규화 완료")
    
    def prepare_training_data(self, test_size=0.2, val_size=0.2):
        """훈련/검증/테스트 데이터 분할"""
        print("🔀 데이터 분할 중...")
        
        X = self.spectrograms['jammed']  # 입력: 재밍된 스펙트로그램
        y = self.spectrograms['clean']   # 출력: 깨끗한 스펙트로그램
        
        # 채널 차원 추가 (H, W, C)
        X = np.expand_dims(X, axis=-1)
        y = np.expand_dims(y, axis=-1)
        
        # 먼저 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # 훈련 데이터에서 검증 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )
        
        self.train_data = (X_train, y_train)
        self.val_data = (X_val, y_val)
        self.test_data = (X_test, y_test)
        
        print(f"✅ 데이터 분할 완료:")
        print(f"   훈련: {X_train.shape[0]}개")
        print(f"   검증: {X_val.shape[0]}개") 
        print(f"   테스트: {X_test.shape[0]}개")
        
        return self.train_data, self.val_data, self.test_data

class UNetSignalRecovery:
    """U-Net 기반 신호 복구 모델"""
    
    def __init__(self, input_shape):
        """
        Args:
            input_shape: 입력 스펙트로그램 크기 (H, W, C)
        """
        self.input_shape = input_shape
        self.model = None
        
    def conv_block(self, inputs, filters, dropout_rate=0.1):
        """합성곱 블록"""
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
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
        s1, p1 = self.encoder_block(inputs, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)
        s4, p4 = self.encoder_block(p3, 512)
        
        # 바텀넥 (최하위 레벨)
        b1 = self.conv_block(p4, 1024)
        
        # 디코더 (확장 경로)
        d1 = self.decoder_block(b1, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)
        d4 = self.decoder_block(d3, s1, 64)
        
        # 출력 레이어
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d4)
        
        self.model = Model(inputs, outputs, name="UNet_SignalRecovery")
        
        print("✅ U-Net 모델 구축 완료")
        print(f"   파라미터 수: {self.model.count_params():,}")
        
        return self.model
    
    def compile_model(self, learning_rate=1e-4):
        """모델 컴파일"""
        print("⚙️ 모델 컴파일 중...")
        
        # 복합 손실 함수 (L1 + SSIM)
        def combined_loss(y_true, y_pred):
            l1_loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)
            ssim_loss = 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)
            return l1_loss + 0.1 * ssim_loss
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=combined_loss,
            metrics=['mae', 'mse']
        )
        
        print("✅ 모델 컴파일 완료")
    
    def train(self, train_data, val_data, epochs=100, batch_size=16):
        """모델 훈련"""
        print(f"🚀 모델 훈련 시작 (에포크: {epochs}, 배치: {batch_size})")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # 훈련 시작
        start_time = time.time()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print(f"✅ 모델 훈련 완료 (소요시간: {training_time:.1f}초)")
        
        return history
    
    def evaluate(self, test_data):
        """테스트 데이터로 모델 평가"""
        print("📊 모델 평가 중...")
        
        X_test, y_test = test_data
        
        # 예측 수행
        y_pred = self.model.predict(X_test)
        
        # 성능 지표 계산
        mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
        mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
        
        # SSIM 계산
        ssim_scores = []
        for i in range(len(y_test)):
            ssim = tf.image.ssim(
                tf.expand_dims(y_test[i], 0),
                tf.expand_dims(y_pred[i], 0),
                max_val=1.0
            ).numpy()[0]
            ssim_scores.append(ssim)
        
        avg_ssim = np.mean(ssim_scores)
        
        print(f"📈 평가 결과:")
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
    
    def __init__(self, data_loader, model, results_dir):
        self.data_loader = data_loader
        self.model = model
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def plot_training_history(self, history):
        """훈련 이력 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('그래프 4: U-Net 훈련 과정 분석', fontsize=16, fontweight='bold')
        
        # 손실 함수
        axes[0,0].plot(history.history['loss'], label='훈련 손실', color='blue')
        axes[0,0].plot(history.history['val_loss'], label='검증 손실', color='red')
        axes[0,0].set_title('4-A: 손실 함수 변화')
        axes[0,0].set_xlabel('에포크')
        axes[0,0].set_ylabel('손실')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # MAE
        axes[0,1].plot(history.history['mae'], label='훈련 MAE', color='blue')
        axes[0,1].plot(history.history['val_mae'], label='검증 MAE', color='red')
        axes[0,1].set_title('4-B: 평균 절대 오차')
        axes[0,1].set_xlabel('에포크')
        axes[0,1].set_ylabel('MAE')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # MSE
        axes[1,0].plot(history.history['mse'], label='훈련 MSE', color='blue')
        axes[1,0].plot(history.history['val_mse'], label='검증 MSE', color='red')
        axes[1,0].set_title('4-C: 평균 제곱 오차')
        axes[1,0].set_xlabel('에포크')
        axes[1,0].set_ylabel('MSE')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 학습률 (옵션)
        if 'lr' in history.history:
            axes[1,1].plot(history.history['lr'], color='green')
            axes[1,1].set_title('4-D: 학습률 변화')
            axes[1,1].set_xlabel('에포크')
            axes[1,1].set_ylabel('학습률')
            axes[1,1].set_yscale('log')
        else:
            axes[1,1].text(0.5, 0.5, '학습률 정보 없음', 
                          transform=axes[1,1].transAxes, ha='center', va='center')
            axes[1,1].set_title('4-D: 학습률 정보')
        
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/graph4_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_recovery_examples(self, test_data, predictions, num_examples=6):
        """신호 복구 예시 시각화"""
        X_test, y_test = test_data
        
        fig, axes = plt.subplots(num_examples, 3, figsize=(15, 4*num_examples))
        fig.suptitle('그래프 5: 신호 복구 결과 예시', fontsize=16, fontweight='bold')
        
        for i in range(num_examples):
            # 재밍된 신호 (입력)
            im1 = axes[i,0].imshow(X_test[i,:,:,0], aspect='auto', cmap='viridis')
            axes[i,0].set_title(f'5-{chr(65+i*3)}: 재밍된 신호 {i+1}')
            axes[i,0].set_ylabel('주파수 빈')
            plt.colorbar(im1, ax=axes[i,0])
            
            # 목표 신호 (정답)
            im2 = axes[i,1].imshow(y_test[i,:,:,0], aspect='auto', cmap='viridis')
            axes[i,1].set_title(f'5-{chr(66+i*3)}: 원본 깨끗한 신호 {i+1}')
            axes[i,1].set_ylabel('주파수 빈')
            plt.colorbar(im2, ax=axes[i,1])
            
            # 복구된 신호 (예측)
            im3 = axes[i,2].imshow(predictions[i,:,:,0], aspect='auto', cmap='viridis')
            axes[i,2].set_title(f'5-{chr(67+i*3)}: 복구된 신호 {i+1}')
            axes[i,2].set_ylabel('주파수 빈')
            plt.colorbar(im3, ax=axes[i,2])
            
            # 하단 행에만 x축 라벨
            if i == num_examples - 1:
                for j in range(3):
                    axes[i,j].set_xlabel('시간 빈')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/graph5_recovery_examples.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_analysis(self, evaluation_results):
        """성능 분석 시각화"""
        predictions = evaluation_results['predictions']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('그래프 6: 신호 복구 성능 분석', fontsize=16, fontweight='bold')
        
        # SSIM 분포
        X_test, y_test = self.data_loader.test_data
        ssim_scores = []
        for i in range(len(y_test)):
            ssim = tf.image.ssim(
                tf.expand_dims(y_test[i], 0),
                tf.expand_dims(predictions[i], 0),
                max_val=1.0
            ).numpy()[0]
            ssim_scores.append(ssim)
        
        axes[0,0].hist(ssim_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].axvline(np.mean(ssim_scores), color='red', linestyle='--', 
                         label=f'평균: {np.mean(ssim_scores):.3f}')
        axes[0,0].set_title('6-A: SSIM 분포')
        axes[0,0].set_xlabel('SSIM 점수')
        axes[0,0].set_ylabel('빈도')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 픽셀별 오차 분포
        pixel_errors = np.abs(y_test - predictions).flatten()
        axes[0,1].hist(pixel_errors, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,1].axvline(np.mean(pixel_errors), color='red', linestyle='--',
                         label=f'평균: {np.mean(pixel_errors):.4f}')
        axes[0,1].set_title('6-B: 픽셀별 절대 오차 분포')
        axes[0,1].set_xlabel('절대 오차')
        axes[0,1].set_ylabel('빈도')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 원본 vs 복구 신호 상관관계
        original_flat = y_test.flatten()
        recovered_flat = predictions.flatten()
        correlation = np.corrcoef(original_flat, recovered_flat)[0,1]
        
        # 샘플링하여 산점도 그리기 (메모리 절약)
        sample_size = min(10000, len(original_flat))
        idx = np.random.choice(len(original_flat), sample_size, replace=False)
        
        axes[1,0].scatter(original_flat[idx], recovered_flat[idx], alpha=0.5, s=1)
        axes[1,0].plot([0, 1], [0, 1], 'r--', alpha=0.8)
        axes[1,0].set_title(f'6-C: 원본 vs 복구 신호 상관관계\n(R = {correlation:.4f})')
        axes[1,0].set_xlabel('원본 신호')
        axes[1,0].set_ylabel('복구된 신호')
        axes[1,0].grid(True, alpha=0.3)
        
        # 재밍 강도별 성능
        # 1단계 데이터의 재밍 파라미터와 매칭
        power_ratios = [params['power_ratio'] for params in self.data_loader.parameters]
        test_indices = np.arange(len(power_ratios))[-len(ssim_scores):]  # 테스트 데이터 인덱스
        test_power_ratios = [power_ratios[i] for i in test_indices]
        
        # 전력비별 SSIM 평균
        unique_powers = sorted(set(test_power_ratios))
        power_ssims = []
        for power in unique_powers:
            power_indices = [i for i, p in enumerate(test_power_ratios) if p == power]
            power_ssim = np.mean([ssim_scores[i] for i in power_indices])
            power_ssims.append(power_ssim)
        
        axes[1,1].plot(unique_powers, power_ssims, 'o-', linewidth=2, markersize=8)
        axes[1,1].set_title('6-D: 재밍 강도별 복구 성능')
        axes[1,1].set_xlabel('전력비')
        axes[1,1].set_ylabel('SSIM 점수')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/graph6_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """메인 실행 함수"""
    print("🚀 재밍 신호 복구 알고리즘 시작")
    print("=" * 60)
    
    # 결과 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"signal_recovery_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # 1. 데이터 로딩
        # 가장 최근 1단계 실험 결과 찾기
        stage1_dirs = [d for d in os.listdir('.') if d.startswith('stage1_results_')]
        if not stage1_dirs:
            raise FileNotFoundError("1단계 실험 결과를 찾을 수 없습니다. 먼저 stage1_experiment.py를 실행하세요.")
        
        latest_stage1 = max(stage1_dirs)
        data_path = os.path.join(latest_stage1, 'stage1_signals.h5')
        
        print(f"📁 1단계 데이터 경로: {data_path}")
        
        # 데이터 로더 초기화
        data_loader = SignalRecoveryDataLoader(data_path)
        data_loader.load_data()
        
        # 2. 스펙트로그램 생성 및 전처리
        data_loader.generate_spectrograms()
        data_loader.normalize_data()
        
        # 3. 훈련/검증/테스트 데이터 준비
        train_data, val_data, test_data = data_loader.prepare_training_data()
        
        # 4. U-Net 모델 구축
        input_shape = train_data[0].shape[1:]  # (H, W, C)
        unet = UNetSignalRecovery(input_shape)
        model = unet.build_model()
        unet.compile_model()
        
        # 모델 구조 요약
        print("\n📋 모델 구조:")
        model.summary()
        
        # 5. 모델 훈련
        print(f"\n🎯 훈련 시작...")
        history = unet.train(train_data, val_data, epochs=50, batch_size=8)
        
        # 6. 모델 평가
        evaluation_results = unet.evaluate(test_data)
        
        # 7. 결과 시각화
        visualizer = SignalRecoveryVisualizer(data_loader, model, results_dir)
        
        print(f"\n🎨 결과 시각화 중...")
        visualizer.plot_training_history(history)
        visualizer.plot_recovery_examples(test_data, evaluation_results['predictions'])
        visualizer.plot_performance_analysis(evaluation_results)
        
        # 8. 모델 저장
        model_path = os.path.join(results_dir, 'signal_recovery_model.h5')
        model.save(model_path)
        print(f"💾 모델 저장: {model_path}")
        
        # 9. 결과 요약 저장
        summary = {
            'experiment_info': {
                'timestamp': timestamp,
                'data_source': data_path,
                'model_type': 'U-Net',
                'training_samples': len(train_data[0]),
                'validation_samples': len(val_data[0]),
                'test_samples': len(test_data[0])
            },
            'model_performance': evaluation_results,
            'training_epochs': len(history.history['loss']),
            'input_shape': input_shape,
            'model_parameters': model.count_params()
        }
        
        summary_path = os.path.join(results_dir, 'recovery_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n🎉 재밍 신호 복구 알고리즘 완료!")
        print(f"📊 최종 성능:")
        print(f"   MSE: {evaluation_results['mse']:.6f}")
        print(f"   MAE: {evaluation_results['mae']:.6f}")
        print(f"   SSIM: {evaluation_results['ssim']:.4f}")
        print(f"📁 결과 저장 위치: {results_dir}/")
        
        return data_loader, unet, evaluation_results
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    data_loader, unet, results = main()