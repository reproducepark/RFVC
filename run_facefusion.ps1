# Facefusion 실행 스크립트
# PowerShell에서 실행

# 현재 디렉토리를 프로젝트 루트로 설정
Set-Location $PSScriptRoot

# facefusion 디렉토리로 이동
Set-Location "facefusion"

# Facefusion 실행
python facefusion.py run --ui-layouts simple_webcam --processors face_swapper --face-swapper-model hyperswap_1a_256 --execution-providers cuda --log-level info

# 실행 완료 후 원래 디렉토리로 복귀
Set-Location .. 