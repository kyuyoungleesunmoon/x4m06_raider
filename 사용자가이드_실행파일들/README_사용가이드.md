# X4M06 레이더 데이터 수집 시스템 사용 가이드

## 📋 개요
이 시스템은 X4M06 UWB 레이더에서 실시간으로 데이터를 수집하고 저장하는 도구입니다.

## 🔧 시스템 요구사항

### 하드웨어
- X4M06 UWB 레이더 모듈
- USB 연결 케이블
- Windows 10/11 PC

### 소프트웨어
- Python 3.6 환경 (conda 가상환경 권장)
- X4M06 pymoduleconnector 라이브러리
- 필수 패키지: numpy, matplotlib, json

## 🏁 처음부터 완전 설치 가이드

### 📦 1단계: 필수 소프트웨어 설치

#### A. Anaconda/Miniconda 설치
1. **Miniconda 다운로드**: https://docs.conda.io/en/latest/miniconda.html
2. **Windows 64-bit 설치파일** 다운로드 및 실행
3. 설치 시 "Add Anaconda to PATH" 옵션 체크
4. 설치 완료 후 **Anaconda Prompt** 실행

#### B. Git 설치 (선택사항)
1. **Git for Windows** 다운로드: https://git-scm.com/download/win
2. 기본 설정으로 설치

### 🐍 2단계: Python 가상환경 생성

**Anaconda Prompt**에서 다음 명령어 실행:

```bash
# 1. Python 3.6 가상환경 생성
conda create -n x4m06_env python=3.6

# 2. 가상환경 활성화
conda activate x4m06_env

# 3. 기본 패키지 설치
conda install numpy matplotlib pandas

# 4. pip 업그레이드
python -m pip install --upgrade pip
```

### 📁 3단계: 프로젝트 파일 준비

```bash
# 1. 작업 디렉토리 생성 및 이동
mkdir C:\X4M06_레이더
cd C:\X4M06_레이더

# 2. pymoduleconnector 설치 (중요!)
# 제공된 python36-win64 폴더에서:
cd "C:\Users\User\Downloads\X4M06_Package\ModuleConnector\ModuleConnector-win32_win64-1\python36-win64"
python setup.py install

# 3. 다시 작업 디렉토리로 이동
cd C:\X4M06_레이더
```

### 🔌 4단계: 하드웨어 연결 및 확인

```bash
# 1. X4M06를 USB 포트에 연결
# 2. COM 포트 확인
# Windows + R → devmgmt.msc → 포트(COM & LPT) 확인

# 3. 연결 테스트
python quick_test.py
```

### ⚙️ 5단계: 환경 설정 파일 실행

```bash
# 1. 설정 관리자 실행
python config_manager.py

# 메뉴에서 다음 설정:
# - COM 포트 설정 (보통 COM3)
# - 탐지 범위 설정 (기본: 0-5m)
# - 샘플링 속도 설정 (기본: 20 FPS)
```

### 🚀 6단계: 첫 데이터 수집 테스트

```bash
# 1. 간편 수집기로 테스트
python easy_data_collector.py

# 2. 메뉴에서 "빠른 테스트" 선택
# 3. 30초간 데이터 수집 확인
# 4. collected_data 폴더에 파일 생성 확인
```

### 📋 7단계: 배치 파일 생성 (선택사항)

자주 사용할 경우 배치 파일 생성:

```batch
@echo off
echo X4M06 레이더 시스템 시작...
call conda activate x4m06_env
cd /d C:\X4M06_레이더
python launcher.py
pause
```

파일명: `start_x4m06.bat`로 저장

## 🔍 설치 확인 체크리스트

설치가 완료되었다면 다음을 확인하세요:

- [ ] Anaconda Prompt에서 `conda activate x4m06_env` 정상 동작
- [ ] `python --version`에서 Python 3.6.x 표시
- [ ] `python -c "import pymoduleconnector"` 에러 없음
- [ ] `python quick_test.py`에서 레이더 연결 성공
- [ ] `collected_data` 폴더 존재 및 데이터 파일 생성됨

## 🚀 빠른 시작 가이드 (이미 설치된 경우)

**⚠️ 처음 설치하는 경우, 먼저 위의 "처음부터 완전 설치 가이드"를 따라하세요!**

### 1단계: 환경 활성화 및 연결 확인

```bash
# 1. Anaconda Prompt 실행
# 2. 가상환경 활성화
conda activate x4m06_env

# 3. 작업 디렉토리로 이동
cd C:\X4M06_레이더

# 4. 레이더 연결 확인 (X4M06가 COM3에 연결된 상태)
python quick_test.py
```

### 2단계: 데이터 수집 실행

#### A. 🌟 통합 런처 (가장 권장)

```bash
# 배치 파일로 실행 (가장 간단)
run_x4m06.bat

# 또는 Python으로 직접 실행
python launcher.py
```

- 모든 기능이 하나의 메뉴에 통합
- 환경 체크 자동 수행
- 초보자부터 전문가까지 대응

#### B. 간편 데이터 수집기 (초보자용)

```bash
python easy_data_collector.py
```

- 간단한 메뉴 선택으로 데이터 수집
- 5가지 사전 설정 모드
- 자동 설정 적용
- 안전하고 안정적

#### C. 고급 데이터 수집기 (전문가용)

```bash
python x4m06_data_saver.py
```

- 상세한 설정 조정 가능
- 실시간 플롯 옵션
- 커스텀 수집 시간/프레임 수
- 명령행 파라미터 지원

#### D. 설정 관리자

```bash
python config_manager.py
```

- 레이더 파라미터 조정
- COM 포트 설정
- 탐지 범위 변경
- 설정 저장/불러오기

## 📊 수집된 데이터 위치
```
C:\X4M06_레이더\collected_data\
├── x4m06_data_[세션ID].json     # 상세 데이터
├── x4m06_data_[세션ID].npz      # NumPy 압축 데이터  
└── summary_[세션ID].txt         # 수집 요약
```

## 🔍 데이터 분석
```bash
python hardware_data_analyzer.py
```
- 저장된 데이터 시각화
- 모션 감지 분석
- CSV 형식 내보내기

## ⚙️ 주요 설정 옵션

### 탐지 범위 설정
- **근거리 (0-2m)**: DAC 900-1000
- **중거리 (0-5m)**: DAC 900-1400 (기본값)
- **장거리 (0-10m)**: DAC 900-1800

### 데이터 수집 설정
- **샘플링 속도**: 10-30 FPS
- **수집 시간**: 10초-10분
- **최대 프레임**: 100-5000개

## 🆘 문제 해결

### 🔌 연결 문제

#### "COM 포트를 찾을 수 없음" 에러

```bash
# 1. 장치 관리자에서 COM 포트 확인
# Windows + R → devmgmt.msc → 포트(COM & LPT)

# 2. 다른 COM 포트로 테스트
python quick_test.py

# 3. USB 케이블 및 포트 교체
```

#### "Access Denied" 또는 "포트가 사용 중" 에러

```bash
# 1. 다른 프로그램 종료 (아두이노 IDE, 시리얼 모니터 등)
# 2. 작업 관리자에서 python.exe 프로세스 모두 종료
# 3. 레이더 USB 연결 해제 후 5초 대기 후 재연결
# 4. 컴퓨터 재부팅
```

### 🐍 Python 환경 문제

#### "conda 명령을 찾을 수 없음" 에러

```bash
# 1. Anaconda Prompt 대신 일반 CMD 사용한 경우
# → 반드시 "Anaconda Prompt" 사용

# 2. PATH 설정 문제
# → Anaconda 재설치 시 "Add to PATH" 옵션 체크

# 3. 가상환경이 생성되지 않은 경우
conda info --envs  # 환경 목록 확인
```

#### "ModuleNotFoundError: pymoduleconnector" 에러

```bash
# 1. 가상환경 활성화 확인
conda activate x4m06_env
python -c "import pymoduleconnector"  # 테스트

# 2. pymoduleconnector 재설치
cd "C:\Users\User\Downloads\X4M06_Package\ModuleConnector\ModuleConnector-win32_win64-1\python36-win64"
python setup.py install --force-reinstall

# 3. DLL 파일 확인
# python36-win64/pymoduleconnector/moduleconnectorwrapper/ 폴더에
# ModuleConnector-x64.dll 파일 존재하는지 확인
```

#### "ImportError: DLL load failed" 에러

```bash
# 1. Visual C++ 재배포 패키지 설치 필요
# Microsoft Visual C++ 2015-2019 Redistributable 다운로드
# https://aka.ms/vs/16/release/vc_redist.x64.exe

# 2. 32비트/64비트 버전 확인
python -c "import platform; print(platform.architecture())"
# ('64bit', 'WindowsPE') 출력 확인

# 3. 가상환경 재생성
conda remove -n x4m06_env --all
conda create -n x4m06_env python=3.6
```

### 📊 데이터 수집 문제

#### "레이더 응답 없음" 에러

```bash
# 1. 레이더 물리적 리셋
# - USB 연결 해제 → 10초 대기 → 재연결

# 2. 소프트웨어 리셋
python -c "
import pymoduleconnector as mc
device = mc.ModuleConnector('COM3')
device.get_xep().module_reset()
device.close()
"

# 3. 다른 COM 포트 시도
python config_manager.py  # COM 포트 변경
```

#### "실시간 플롯이 표시되지 않음" 에러

```bash
# 1. matplotlib 백엔드 확인
python -c "import matplotlib; print(matplotlib.get_backend())"
# 'TkAgg' 출력되어야 함

# 2. tkinter 설치 확인
python -c "import tkinter"  # 에러 없어야 함

# 3. 백엔드 강제 설정
export MPLBACKEND=TkAgg  # Linux/Mac
set MPLBACKEND=TkAgg     # Windows
```

### 💾 데이터 저장 문제

#### "Permission denied" 저장 에러

```bash
# 1. collected_data 폴더 권한 확인
# 폴더 우클릭 → 속성 → 보안 → 편집 → 모든 권한 허용

# 2. 다른 위치에 저장
python x4m06_data_saver.py --output_dir "D:\radar_data"

# 3. 관리자 권한으로 실행
# Anaconda Prompt를 "관리자 권한으로 실행"
```

#### "디스크 공간 부족" 에러

```bash
# 1. 용량 확인
dir collected_data  # 파일 크기 확인

# 2. 오래된 데이터 삭제
# collected_data 폴더의 오래된 파일들 삭제

# 3. 수집 설정 조정
# - 프레임 수 줄이기 (1000 → 500)
# - 수집 시간 단축 (300초 → 120초)
```

### 🔧 성능 문제

#### "수집 속도가 너무 느림" 문제

```bash
# 1. 실시간 플롯 비활성화
python easy_data_collector.py
# → "빠른 테스트" 또는 "장시간 수집" 모드 선택

# 2. USB 포트 변경
# USB 3.0 포트 사용 권장

# 3. 백그라운드 프로그램 종료
# 안티바이러스, 백업 프로그램 등 일시 중지
```

### 📱 일반적인 해결 순서

문제 발생 시 다음 순서로 시도:

1. **프로그램 재시작**
   ```bash
   # Ctrl+C로 현재 프로그램 종료 → 재실행
   ```

2. **레이더 재연결**
   ```bash
   # USB 분리 → 10초 대기 → 재연결
   ```

3. **가상환경 재활성화**
   ```bash
   conda deactivate
   conda activate x4m06_env
   ```

4. **컴퓨터 재부팅**
   ```bash
   # 모든 방법이 실패할 경우
   ```

### 🚨 응급 복구 방법

모든 것이 작동하지 않을 때:

```bash
# 1. 가상환경 완전 재생성
conda remove -n x4m06_env --all
conda create -n x4m06_env python=3.6
conda activate x4m06_env

# 2. 패키지 재설치
conda install numpy matplotlib pandas
cd "C:\Users\User\Downloads\X4M06_Package\ModuleConnector\ModuleConnector-win32_win64-1\python36-win64"
python setup.py install

# 3. 연결 테스트
cd C:\X4M06_레이더
python quick_test.py
```

## ❓ 자주 묻는 질문 (FAQ)

### Q1: Python 3.6을 꼭 사용해야 하나요?

**A:** 네, 반드시 Python 3.6을 사용하세요. X4M06 pymoduleconnector 라이브러리가 Python 3.6 전용으로 컴파일되어 있어 다른 버전에서는 DLL 에러가 발생합니다.

### Q2: 수집된 데이터 파일이 너무 큽니다. 어떻게 해야 하나요?

**A:** 다음 방법들을 시도해보세요:
- NPZ 파일만 사용 (JSON보다 용량 50% 적음)
- 수집 프레임 수 줄이기 (1000 → 500)
- 필요한 데이터만 선별적 수집
- 주기적으로 오래된 데이터 백업/삭제

### Q3: 여러 대의 X4M06를 동시에 사용할 수 있나요?

**A:** 각각 다른 COM 포트에 연결하면 가능합니다:
```bash
# 터미널 1
python x4m06_data_saver.py --device COM3

# 터미널 2
python x4m06_data_saver.py --device COM4
```

### Q4: Linux나 Mac에서도 사용할 수 있나요?

**A:** 현재 버전은 Windows 전용입니다. DLL 파일이 Windows 전용으로 컴파일되어 있어 다른 OS에서는 작동하지 않습니다.

### Q5: 데이터 수집 중 컴퓨터를 절전 모드로 둘 수 있나요?

**A:** 권장하지 않습니다. 절전 모드 시 USB 포트가 비활성화되어 연결이 끊어질 수 있습니다. 장시간 수집 시에는 절전 모드를 해제하세요.

### Q6: 실시간 플롯에 한글이 깨져 보입니다.

**A:** 시스템이 이미 안정적인 영어 라벨을 사용하도록 설정되어 있습니다. 만약 문제가 있다면:
```bash
# 폰트 확인
python -c "import matplotlib.font_manager as fm; print([f.name for f in fm.fontManager.ttflist if 'DejaVu' in f.name])"
```

### Q7: 수집 속도를 더 빠르게 할 수 있나요?

**A:** 현재 설정(20 FPS)이 X4M06의 권장 속도입니다. 더 높은 속도는 데이터 품질 저하를 야기할 수 있습니다.

## 📞 지원 및 문의

### 🔧 문제 신고 시 포함할 정보

1. **에러 메시지** (정확한 영문 메시지)
2. **실행한 명령어** (예: `python easy_data_collector.py`)
3. **Python 버전** (`python --version` 결과)
4. **운영체제** (Windows 10/11)
5. **COM 포트** (장치 관리자에서 확인한 포트 번호)
6. **로그 파일** (`logs/` 폴더 내용)

### 📧 연락처

- 문제 발생 시 로그 파일 확인: `logs/` 폴더
- 오류 메시지와 함께 문의
- 수집된 데이터 샘플 첨부 권장

## 📝 주의사항 및 권장사항

### ⚠️ 주의사항

- **데이터 수집 중 레이더 모듈 분리 금지**
- **절전 모드 비활성화** (장시간 수집 시)
- **안티바이러스 실시간 감시 예외 설정** (성능 향상)
- **5m 이상 탐지 시 전력 소모 증가**

### 💡 권장사항

- **정기적인 데이터 백업** (주 1회 이상)
- **배치 파일 사용** (`run_x4m06.bat` 권장)
- **테스트 수집 먼저 실행** (장시간 수집 전)
- **USB 3.0 포트 사용** (성능 향상)

### 📊 데이터 관리 팁

```text
collected_data/
├── 2025-09/        # 월별 폴더로 정리
├── 2025-10/
└── archive/        # 오래된 데이터 보관
```

---

## 📋 체크리스트

### 설치 완료 체크리스트

- [ ] Anaconda/Miniconda 설치됨
- [ ] Python 3.6 가상환경 생성됨 (`x4m06_env`)
- [ ] pymoduleconnector 설치됨
- [ ] X4M06 하드웨어 연결됨 (COM 포트 확인)
- [ ] `python quick_test.py` 성공
- [ ] `collected_data` 폴더 존재
- [ ] 첫 데이터 수집 테스트 완료

### 일일 사용 체크리스트

- [ ] X4M06 USB 연결 확인
- [ ] Anaconda Prompt 실행
- [ ] `conda activate x4m06_env` 실행
- [ ] 작업 디렉토리 이동: `cd C:\X4M06_레이더`
- [ ] 연결 테스트: `python quick_test.py`
- [ ] 데이터 수집 실행: `run_x4m06.bat`

---

**🎯 이제 X4M06 레이더 시스템을 완전히 활용할 준비가 되었습니다!**