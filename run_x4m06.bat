@echo off
echo ====================================
echo    X4M06 레이더 데이터 수집 시스템
echo ====================================
echo.

REM 가상환경 활성화 확인
echo [1/3] Python 환경 확인 중...
call conda activate x4m06_env 2>nul
if errorlevel 1 (
    echo 경고: x4m06_env 가상환경을 찾을 수 없습니다.
    echo 기본 Python을 사용합니다.
) else (
    echo 성공: x4m06_env 환경 활성화됨
)

REM 작업 디렉토리 확인
echo.
echo [2/3] 작업 디렉토리 확인 중...
if not exist "%~dp0launcher.py" (
    echo 오류: 필요한 파일을 찾을 수 없습니다.
    echo 올바른 디렉토리에서 실행해주세요.
    pause
    exit /b 1
)

REM 데이터 디렉토리 생성
if not exist "%~dp0collected_data" (
    mkdir "%~dp0collected_data"
    echo 정보: collected_data 폴더가 생성되었습니다.
)

REM 로그 디렉토리 생성
if not exist "%~dp0logs" (
    mkdir "%~dp0logs"
)

echo 성공: 모든 구성 요소가 준비되었습니다.
echo.

echo [3/3] X4M06 시스템 론처 실행 중...
echo.
python "%~dp0launcher.py"

echo.
echo ====================================
echo 시스템이 종료되었습니다.
echo 수집된 데이터: collected_data 폴더
echo ====================================
pause