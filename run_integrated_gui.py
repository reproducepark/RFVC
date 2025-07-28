#!/usr/bin/env python3
"""
Face-VC 통합 GUI 실행 스크립트
seed-vc conda 환경에서 실행해야 합니다.
"""

import os
import sys
import subprocess
import argparse

def check_conda_env():
    """conda 환경 확인"""
    current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if current_env != 'seed-vc':
        print(f"Warning: 현재 conda 환경이 'seed-vc'가 아닙니다. (현재: {current_env})")
        print("'seed-vc' 환경에서 실행하는 것을 권장합니다.")
        response = input("계속 진행하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

def install_requirements():
    """필요한 패키지 설치"""
    print("필요한 패키지를 설치합니다...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("패키지 설치 완료!")
    except subprocess.CalledProcessError as e:
        print(f"패키지 설치 중 오류 발생: {e}")
        sys.exit(1)

def run_gui():
    """GUI 실행"""
    print("Face-VC 통합 GUI를 시작합니다...")
    try:
        from integrated_gui import main
        main()
    except ImportError as e:
        print(f"모듈 import 오류: {e}")
        print("필요한 패키지가 설치되지 않았습니다.")
        print("다음 명령어로 패키지를 설치한 후 다시 시도하세요:")
        print("python run_integrated_gui.py --install")
        sys.exit(1)
    except Exception as e:
        print(f"GUI 실행 중 오류 발생: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Face-VC 통합 GUI 실행")
    parser.add_argument("--install", action="store_true", help="필요한 패키지 설치")
    parser.add_argument("--skip-env-check", action="store_true", help="conda 환경 확인 건너뛰기")
    
    args = parser.parse_args()
    
    if not args.skip_env_check:
        check_conda_env()
    
    if args.install:
        install_requirements()
    
    run_gui()

if __name__ == "__main__":
    main() 