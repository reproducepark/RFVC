# Face-VC 통합 GUI

FaceFusion과 Seed-VC를 통합한 실시간 처리 GUI입니다.

## 실행 방법

### 1. 개별 실행
- **facefusion gui 실행법**
  ```bash
  cd facefusion
  python facefusion.py run --ui-layouts simple_webcam
  ```

- **seed-vc 실행법**
  ```bash
  cd seed-vc
  python real-time-gui.py --checkpoint-path runs\yoon2\ft_model.pth  --config-path runs\yoon2\config_dit_mel_seed_uvit_xlsr_tiny.yml
  ```

### 2. 통합 GUI 실행 (권장)
```bash
# seed-vc conda 환경에서 실행
conda activate seed-vc

# 필요한 패키지 설치 (처음 실행 시)
python run_integrated_gui.py --install

# GUI 실행
python run_integrated_gui.py
```

또는 직접 실행:
```bash
python integrated_gui.py
```

## 통합 GUI 기능

### FaceFusion 섹션
- 소스 이미지 선택 (얼굴 교체용)
- UDP 스트림으로 실시간 전송 (udp://localhost:27000)
- FaceFusion 시작/정지 버튼
- VLC나 다른 미디어 플레이어로 스트림 시청 가능

### Seed-VC 섹션
- 참조 오디오 선택 (음성 변환용)
- Voice Conversion 시작/정지 버튼
- 실시간 상태 표시

## 시스템 요구사항
- Python 3.8+
- CUDA 지원 GPU (권장)
- seed-vc conda 환경
- 웹캠 및 마이크

## 테스트

GUI 실행 전에 시스템이 올바르게 설정되었는지 확인하려면:

```bash
python test_integration.py
```

이 스크립트는 다음을 확인합니다:
- 필요한 패키지들이 설치되어 있는지
- FaceFusion과 Seed-VC 모듈들이 올바르게 import되는지
- GPU/CUDA 설정이 올바른지
- 모델 파일들이 존재하는지
- 웹캠과 오디오 디바이스에 접근할 수 있는지

## UDP 스트림 사용법

FaceFusion이 시작되면 UDP 스트림이 `udp://localhost:27000`으로 전송됩니다.

### VLC로 스트림 시청:
1. VLC 미디어 플레이어 실행
2. `Ctrl+N` 또는 `미디어 > 네트워크 스트림 열기`
3. 네트워크 URL에 `udp://localhost:27000` 입력
4. 재생 버튼 클릭

### 다른 미디어 플레이어:
- **PotPlayer**: `Ctrl+U` > `udp://localhost:27000`
- **MPV**: `mpv udp://localhost:27000`
- **FFplay**: `ffplay udp://localhost:27000`

## 주의사항
- seed-vc conda 환경에서 실행해야 합니다.
- 모델 파일들이 올바른 경로에 있어야 합니다.
- 웹캠과 마이크 권한이 필요합니다.
- GPU 메모리가 충분해야 합니다 (최소 8GB 권장).
- UDP 스트림을 시청하려면 VLC나 다른 미디어 플레이어가 필요합니다.