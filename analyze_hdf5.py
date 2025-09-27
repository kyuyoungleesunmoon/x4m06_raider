#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HDF5 파일 상세 분석 스크립트
"""

import h5py
import numpy as np

def analyze_hdf5_file():
    """HDF5 파일 상세 분석"""
    file_path = 'experiment_results/session_20250927_190149/simulation/radar_jamming_dataset_1000.h5'
    
    with h5py.File(file_path, 'r') as f:
        print("=== HDF5 파일 상세 정보 ===")
        
        # 파일 크기 계산
        total_size = sum(d.size * d.dtype.itemsize for d in f.values()) / (1024**2)
        print(f"총 파일 크기: {total_size:.1f} MB")
        
        print("\\n📊 데이터셋 정보:")
        for key, dataset in f.items():
            size_mb = dataset.size * dataset.dtype.itemsize / (1024**2)
            print(f"  {key}:")
            print(f"    형태: {dataset.shape}")
            print(f"    타입: {dataset.dtype}")
            print(f"    크기: {size_mb:.1f} MB")
            print()
        
        # 첫 번째 샘플 통계
        print("🔍 첫 번째 샘플 데이터 분석:")
        
        # Clean signal 분석
        clean_signal = f['clean_signals'][0]
        print(f"  Clean signal:")
        print(f"    실수부 범위: {np.min(clean_signal.real):.3f} - {np.max(clean_signal.real):.3f}")
        print(f"    허수부 범위: {np.min(clean_signal.imag):.3f} - {np.max(clean_signal.imag):.3f}")
        print(f"    크기 범위: {np.min(np.abs(clean_signal)):.3f} - {np.max(np.abs(clean_signal)):.3f}")
        
        # Jammed signal 분석  
        jammed_signal = f['jammed_signals'][0]
        print(f"  Jammed signal:")
        print(f"    실수부 범위: {np.min(jammed_signal.real):.3f} - {np.max(jammed_signal.real):.3f}")
        print(f"    허수부 범위: {np.min(jammed_signal.imag):.3f} - {np.max(jammed_signal.imag):.3f}")
        print(f"    크기 범위: {np.min(np.abs(jammed_signal)):.3f} - {np.max(np.abs(jammed_signal)):.3f}")
        
        # Spectrogram 분석
        clean_spec = f['clean_spectrograms'][0]
        jammed_spec = f['jammed_spectrograms'][0]
        print(f"  Clean spectrogram:")
        print(f"    값 범위: {np.min(clean_spec):.3f} - {np.max(clean_spec):.3f}")
        print(f"    평균: {np.mean(clean_spec):.3f}")
        
        print(f"  Jammed spectrogram:")
        print(f"    값 범위: {np.min(jammed_spec):.3f} - {np.max(jammed_spec):.3f}")
        print(f"    평균: {np.mean(jammed_spec):.3f}")
        
        # 전체 데이터 통계
        print("\\n📈 전체 데이터셋 통계:")
        all_clean_specs = f['clean_spectrograms'][:]
        all_jammed_specs = f['jammed_spectrograms'][:]
        
        print(f"  Clean spectrograms 전체:")
        print(f"    최솟값: {np.min(all_clean_specs):.3f}")
        print(f"    최댓값: {np.max(all_clean_specs):.3f}")
        print(f"    평균: {np.mean(all_clean_specs):.3f}")
        print(f"    표준편차: {np.std(all_clean_specs):.3f}")
        
        print(f"  Jammed spectrograms 전체:")
        print(f"    최솟값: {np.min(all_jammed_specs):.3f}")
        print(f"    최댓값: {np.max(all_jammed_specs):.3f}")
        print(f"    평균: {np.mean(all_jammed_specs):.3f}")
        print(f"    표준편차: {np.std(all_jammed_specs):.3f}")

if __name__ == "__main__":
    analyze_hdf5_file()