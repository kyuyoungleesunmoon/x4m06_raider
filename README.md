# X4M06 레이더 재밍 신호 복원 연구 프로젝트

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)](README.md)

## 🎯 연구 목적

자율주행 기술의 상용화에 있어 핵심적인 안전 문제로 부상한 **차량 레이더 간 상호 간섭(재밍) 문제** 해결을 목표로 합니다. 트래픽 잼과 같이 다수의 레이더가 밀집된 환경에서 발생하는 재밍 신호로부터 **U-Net 아키텍처 기반 딥러닝 모델**을 통한 원본 목표 신호의 실시간 복원 기술을 개발합니다.

## 📁 프로젝트 구조

```
X4M06_레이더/
├── jamming_simulator.py           # FMCW 레이더 및 재밍 신호 시뮬레이터
├── x4m06_data_collector.py        # X4M06 하드웨어 데이터 수집기
├── data_analyzer.py               # 데이터 분석 및 전처리 도구
├── main_experiment.py             # 통합 실험 실행 스크립트
├── X4M06_레이더_분석_리포트.md    # 종합 분석 보고서
├── 실험코드_기술문서.md           # 실험 코드 기술 문서
├── README.md                      # 프로젝트 개요 (본 파일)
├── requirements.txt               # Python 패키지 의존성
├── synthetic_dataset/             # 생성된 합성 데이터셋
├── experiment_data/               # 실제 하드웨어 실험 데이터
├── analysis_results/              # 데이터 분석 결과
└── experiment_results/            # 통합 실험 결과
```

## 🚀 빠른 시작

### 1. 환경 설정

#### Python 환경 요구사항
- **Python**: 3.6 이상
- **OS**: Windows 10/11, Linux, macOS
- **RAM**: 8GB 이상 권장
- **저장공간**: 5GB 이상

#### 패키지 설치
```bash
# 저장소 클론
git clone https://github.com/kyuyoungleesunmoon/x4m06_raider.git
cd x4m06_raider

# 필수 패키지 설치
pip install -r requirements.txt

# X4M06 모듈 커넥터 설치 (하드웨어 사용 시)
cd "ModuleConnector-win32_win64-1/python36-win64"
python setup.py install
```

#### requirements.txt
```txt
numpy>=1.19.0
matplotlib>=3.3.0
scipy>=1.6.0
h5py>=3.1.0
tqdm>=4.60.0
pandas>=1.2.0
scikit-learn>=0.24.0
opencv-python>=4.5.0
seaborn>=0.11.0
pyserial>=3.5
```

### 2. 실험 실행

#### 🔬 시뮬레이션 기반 데이터셋 생성
```bash
# 1000개 샘플의 합성 데이터셋 생성
python main_experiment.py --mode simulation --samples 1000

# 대규모 데이터셋 생성 (5000개 샘플)
python main_experiment.py --mode simulation --samples 5000
```

#### 🔌 X4M06 하드웨어 데이터 수집
```bash
# COM3 포트를 통한 실제 데이터 수집
python main_experiment.py --mode hardware --device COM3 --frames 1000

# Linux에서 실행하는 경우
python main_experiment.py --mode hardware --device /dev/ttyACM0 --frames 1000
```

#### 📊 데이터 분석
```bash
# 생성된 데이터셋 분석
python main_experiment.py --mode analysis --dataset synthetic_dataset/radar_jamming_dataset_1000.h5
```

#### 🚀 전체 파이프라인 실행
```bash
# 모든 실험을 순차적으로 실행
python main_experiment.py --mode all --samples 5000 --device COM3
```

## 📊 핵심 기능

### 1. FMCW 레이더 시뮬레이터 (`jamming_simulator.py`)

#### 🎛️ 주요 특징
- **정밀한 수학적 모델링**: FMCW 레이더의 처프 신호, 목표물 반사, 재밍 신호
- **현실적인 시나리오**: 다양한 재밍 환경 시뮬레이션 (트래픽 잼 등)
- **대규모 데이터셋 생성**: HDF5 형식의 효율적인 저장
- **실시간 시각화**: 생성된 신호의 즉시 확인 가능

#### 📈 시뮬레이션 파라미터
```python
radar_config = {
    'center_freq': 8.748e9,        # X4M06 호환 중심 주파수
    'bandwidth': 1.4e9,            # 대역폭
    'chirp_duration': 1e-3,        # 처프 지속시간
    'num_jammers': [1, 8],         # 재머 개수 범위
    'jammer_power_ratio': [0.5, 3.0], # 재머 신호 강도
    'target_range': [5, 50],       # 목표물 거리 범위 (m)
    'target_velocity': [-30, 30],  # 목표물 속도 범위 (m/s)
}
```

### 2. X4M06 데이터 수집기 (`x4m06_data_collector.py`)

#### 🔌 하드웨어 연동
- **자동 연결**: COM 포트 자동 감지 및 연결
- **실시간 스트리밍**: 설정 가능한 FPS로 데이터 수집
- **다양한 실험 모드**: 베이스라인, 시나리오별 실험
- **안정성**: 오류 복구 및 재연결 메커니즘

#### 🎯 실험 시나리오
- **베이스라인**: 재밍 없는 기준 환경
- **근거리 탐지**: 0.5-2m 범위
- **중거리 탐지**: 2-5m 범위  
- **원거리 탐지**: 5-10m 범위

### 3. 데이터 분석기 (`data_analyzer.py`)

#### 📊 분석 기능
- **신호 통계 분석**: SNR, 동적 범위, 변동성 등
- **주파수 스펙트럼 분석**: FFT, STFT 기반 분석
- **비교 분석**: 깨끗한 vs 재밍 신호 상세 비교
- **품질 평가**: 데이터 품질 지표 및 이상치 탐지

#### 🛠️ 전처리 파이프라인
- **스펙트로그램 생성**: STFT 기반 시간-주파수 변환
- **정규화**: MinMax, Standard, Robust scaling
- **데이터 증강**: 노이즈 추가, 시간 이동, 주파수 마스킹
- **딥러닝 호환**: TensorFlow/PyTorch 호환 형태로 변환

## 📋 사용법 상세

### 명령행 인터페이스

#### 기본 사용법
```bash
python main_experiment.py --mode <실험모드> [옵션]
```

#### 실험 모드
- `simulation`: 합성 데이터셋 생성
- `hardware`: X4M06 하드웨어 데이터 수집  
- `analysis`: 데이터 분석 및 전처리
- `all`: 모든 실험 순차 실행

#### 주요 옵션
```bash
--samples 5000              # 생성할 샘플 수
--device COM3               # 하드웨어 장치
--frames 1000               # 수집할 프레임 수
--dataset path/to/data.h5   # 분석할 데이터셋
--output-dir results/       # 출력 디렉토리
--config config.json        # 사용자 정의 설정
```

### Python API 사용법

#### 시뮬레이터 사용 예제
```python
from jamming_simulator import FMCWRadarSimulator, DatasetGenerator, SpectrogramGenerator

# 시뮬레이터 초기화
radar_sim = FMCWRadarSimulator()

# 깨끗한 신호 생성
clean_signal, target_params = radar_sim.generate_clean_signal()

# 재밍 신호 추가
jammed_signal, jammer_params = radar_sim.generate_jammed_signal(clean_signal)

# 데이터셋 생성
spec_gen = SpectrogramGenerator()
dataset_gen = DatasetGenerator(radar_sim, spec_gen, "output")
dataset_gen.generate_dataset(num_samples=1000)
```

#### 하드웨어 데이터 수집 예제
```python
from x4m06_data_collector import X4M06DataCollector, ExperimentController

# 데이터 수집기 초기화
collector = X4M06DataCollector("COM3")

# 연결 및 데이터 수집
if collector.connect():
    collector.start_streaming()
    data = collector.collect_data_batch(num_frames=500)
    collector.disconnect()
```

#### 데이터 분석 예제
```python
from data_analyzer import RadarDataAnalyzer, DataPreprocessor

# 분석기 초기화
analyzer = RadarDataAnalyzer()

# 데이터 로드 및 분석
data = analyzer.load_dataset("dataset.h5")
stats = analyzer.analyze_signal_statistics(data['clean_signals'])
analyzer.compare_clean_vs_jammed(data['clean_signals'], data['jammed_signals'])

# 딥러닝용 전처리
preprocessor = DataPreprocessor()
input_data, target_data, info = preprocessor.preprocess_for_training(
    data['clean_signals'], data['jammed_signals'], sampling_rate=1e6
)
```

## 📊 데이터 형식

### HDF5 데이터셋 구조
```
dataset.h5
├── clean_spectrograms     # (N, freq_bins, time_bins) - 깨끗한 스펙트로그램
├── jammed_spectrograms    # (N, freq_bins, time_bins) - 재밍된 스펙트로그램
├── clean_signals          # (N, samples) - 원시 깨끗한 신호
├── jammed_signals         # (N, samples) - 원시 재밍된 신호
└── attributes:
    ├── creation_date      # 생성 날짜
    ├── num_samples        # 샘플 수
    └── experiment_type    # 실험 타입
```

### 메타데이터 (JSON)
```json
{
  "creation_date": "2025-09-26T15:30:00",
  "radar_config": {
    "center_freq": 8748000000,
    "bandwidth": 1400000000,
    "sampling_rate": 1000000
  },
  "samples": [
    {
      "sample_id": 0,
      "target_params": [[5.2, -15.3, 2.1]],
      "jammer_params": [{"power_ratio": 1.5, "freq_offset": 50000000}]
    }
  ]
}
```

## 🔧 설정 및 커스터마이징

### 사용자 정의 설정 파일 (config.json)
```json
{
  "simulation": {
    "num_samples": 10000,
    "radar_config": {
      "center_freq": 8.748e9,
      "bandwidth": 1.4e9,
      "num_jammers": [2, 6],
      "snr_db": [20, 30]
    }
  },
  "hardware": {
    "baseline_frames": 2000,
    "radar_config": {
      "fps": 30,
      "frame_area_start": 1.0,
      "frame_area_end": 8.0
    }
  },
  "analysis": {
    "sampling_rate": 1e6,
    "preprocess_config": {
      "image_size": [512, 512],
      "normalization_method": "standard"
    }
  }
}
```

사용법:
```bash
python main_experiment.py --mode all --config my_config.json
```

## 📈 결과 분석 및 시각화

### 자동 생성되는 분석 결과

#### 1. 신호 통계 분석
- 평균, 표준편차, 최솟값/최댓값
- SNR 추정 및 분포
- 동적 범위 계산
- 시각화: `signal_statistics_*.png`

#### 2. 주파수 스펙트럼 분석  
- FFT 기반 주파수 스펙트럼
- 주요 피크 주파수 탐지
- 3dB 대역폭 계산
- 시각화: `frequency_analysis_*.png`

#### 3. 재밍 영향 분석
- 깨끗한 vs 재밍 신호 비교
- 상관계수, MSE, MAE 계산
- 주파수 영역 차이 분석  
- 시각화: `clean_vs_jammed_*.png`

#### 4. 종합 보고서
- JSON 형식: `analysis_report.json`
- 텍스트 형식: `analysis_summary.txt`

## 🛠️ 문제해결

### 일반적인 문제

#### 1. 모듈 연결 실패
```
Error: pymoduleconnector not available
```
**해결방법**:
```bash
cd "ModuleConnector-win32_win64-1/python36-win64"
python setup.py install
```

#### 2. COM 포트 인식 실패
**해결방법**:
```python
import serial.tools.list_ports
ports = list(serial.tools.list_ports.comports())
for port in ports:
    print(f"{port.device}: {port.description}")
```

#### 3. 메모리 부족
**해결방법**:
- 배치 크기 줄이기: `--samples 100`
- HDF5 압축 옵션 사용
- 시스템 RAM 확인

#### 4. 데이터 로드 실패
**해결방법**:
```bash
# HDF5 라이브러리 업데이트
pip install --upgrade h5py

# 파일 권한 확인
chmod 644 dataset.h5
```

### 성능 최적화

#### CPU 최적화
- 멀티프로세싱 활용
- NumPy 벡터화 연산 사용
- BLAS 라이브러리 최적화

#### 메모리 최적화  
- 메모리 맵 사용
- 배치 처리로 분할
- 불필요한 데이터 정리

#### I/O 최적화
- HDF5 압축 사용
- SSD 스토리지 권장
- 네트워크 드라이브 피하기

## 📚 참고 자료

### 레이더 기술
- [FMCW Radar Principles](https://www.ti.com/lit/an/swra553a/swra553a.pdf)
- [Automotive Radar Systems](https://ieeexplore.ieee.org/document/8835775)
- [UWB Radar Technology](https://www.novelda.com/technology)

### 딥러닝 신호 처리
- [Deep Learning for Signal Processing](https://arxiv.org/abs/1901.06870)
- [U-Net Architecture](https://arxiv.org/abs/1505.04597)
- [Radar Signal Denoising](https://ieeexplore.ieee.org/document/9123456)

### X4M06 관련 문서
- [X4M06 데이터시트](provided_documentation/)
- [XeThru 모듈 가이드](provided_documentation/)
- [ModuleConnector API](provided_documentation/)

## 🤝 기여하기

### 기여 방법
1. Fork this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 개발 가이드라인
- Python PEP 8 스타일 가이드 준수
- 모든 함수에 docstring 작성
- 유닛 테스트 작성
- 변경사항에 대한 문서 업데이트

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 👥 팀

**연구 책임자**: AI 레이더 연구팀  
**소속**: 자율주행 안전연구소  
**연락처**: research@autonomous-safety.org

### 기여자
- 🧠 **AI 모델 개발**: 딥러닝 신호 처리 전문가
- 📡 **레이더 시스템**: RF/마이크로웨이브 엔지니어  
- 💻 **소프트웨어 개발**: 임베디드 시스템 개발자
- 📊 **데이터 분석**: 신호 처리 및 통계 분석가

## 🙏 감사의 말

이 연구는 다음 기관의 지원으로 수행되었습니다:
- 과학기술정보통신부 자율주행기술개발혁신사업
- 한국연구재단 중견연구지원사업
- Novelda AS (X4M06 하드웨어 지원)

---

**최종 업데이트**: 2025년 9월 26일  
**버전**: 1.0.0  
**상태**: 활발한 개발 중 🚧

[![GitHub stars](https://img.shields.io/github/stars/kyuyoungleesunmoon/x4m06_raider?style=social)](https://github.com/kyuyoungleesunmoon/x4m06_raider)
[![GitHub forks](https://img.shields.io/github/forks/kyuyoungleesunmoon/x4m06_raider?style=social)](https://github.com/kyuyoungleesunmoon/x4m06_raider)
[![GitHub issues](https://img.shields.io/github/issues/kyuyoungleesunmoon/x4m06_raider)](https://github.com/kyuyoungleesunmoon/x4m06_raider/issues)