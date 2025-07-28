#!/usr/bin/env python3
"""
통합 GUI 테스트 스크립트
기본적인 import와 설정이 올바른지 확인합니다.
"""

import os
import sys
import torch

def test_imports():
    """필요한 모듈들이 올바르게 import되는지 테스트"""
    print("=== Import 테스트 ===")
    
    try:
        import gradio as gr
        print("✓ Gradio import 성공")
    except ImportError as e:
        print(f"✗ Gradio import 실패: {e}")
        return False
    
    try:
        import cv2
        print("✓ OpenCV import 성공")
    except ImportError as e:
        print(f"✗ OpenCV import 실패: {e}")
        return False
    
    try:
        import librosa
        print("✓ Librosa import 성공")
    except ImportError as e:
        print(f"✗ Librosa import 실패: {e}")
        return False
    
    try:
        import sounddevice as sd
        print("✓ Sounddevice import 성공")
    except ImportError as e:
        print(f"✗ Sounddevice import 실패: {e}")
        return False
    
    try:
        import torchaudio
        print("✓ Torchaudio import 성공")
    except ImportError as e:
        print(f"✗ Torchaudio import 실패: {e}")
        return False
    
    return True

def test_facefusion_imports():
    """FaceFusion 관련 import 테스트"""
    print("\n=== FaceFusion Import 테스트 ===")
    
    try:
        sys.path.append('facefusion')
        from facefusion import state_manager
        print("✓ FaceFusion state_manager import 성공")
    except ImportError as e:
        print(f"✗ FaceFusion state_manager import 실패: {e}")
        return False
    
    try:
        from facefusion.face_analyser import get_average_face, get_many_faces
        print("✓ FaceFusion face_analyser import 성공")
    except ImportError as e:
        print(f"✗ FaceFusion face_analyser import 실패: {e}")
        return False
    
    try:
        from facefusion.filesystem import filter_image_paths
        print("✓ FaceFusion filesystem import 성공")
    except ImportError as e:
        print(f"✗ FaceFusion filesystem import 실패: {e}")
        return False
    
    return True

def test_seed_vc_imports():
    """Seed-VC 관련 import 테스트"""
    print("\n=== Seed-VC Import 테스트 ===")
    
    try:
        sys.path.append('seed-vc')
        from modules.commons import recursive_munch, build_model, load_checkpoint
        print("✓ Seed-VC commons import 성공")
    except ImportError as e:
        print(f"✗ Seed-VC commons import 실패: {e}")
        return False
    
    try:
        from modules.audio import mel_spectrogram
        print("✓ Seed-VC audio import 성공")
    except ImportError as e:
        print(f"✗ Seed-VC audio import 실패: {e}")
        return False
    
    try:
        from modules.campplus.DTDNN import CAMPPlus
        print("✓ Seed-VC CAMPPlus import 성공")
    except ImportError as e:
        print(f"✗ Seed-VC CAMPPlus import 실패: {e}")
        return False
    
    return True

def test_device():
    """디바이스 설정 테스트"""
    print("\n=== 디바이스 테스트 ===")
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"✓ CUDA 사용 가능: {device}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✓ MPS 사용 가능: {device}")
    else:
        device = torch.device("cpu")
        print(f"✓ CPU 사용: {device}")
    
    return True

def test_model_files():
    """모델 파일 존재 여부 테스트"""
    print("\n=== 모델 파일 테스트 ===")
    
    checkpoint_path = "seed-vc/runs/yoon2/ft_model.pth"
    config_path = "seed-vc/runs/yoon2/config_dit_mel_seed_uvit_xlsr_tiny.yml"
    
    if os.path.exists(checkpoint_path):
        print(f"✓ 체크포인트 파일 존재: {checkpoint_path}")
    else:
        print(f"✗ 체크포인트 파일 없음: {checkpoint_path}")
        return False
    
    if os.path.exists(config_path):
        print(f"✓ 설정 파일 존재: {config_path}")
    else:
        print(f"✗ 설정 파일 없음: {config_path}")
        return False
    
    return True

def test_webcam():
    """웹캠 접근 테스트"""
    print("\n=== 웹캠 테스트 ===")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ 웹캠 접근 성공")
            cap.release()
            return True
        else:
            print("✗ 웹캠 접근 실패")
            return False
    except Exception as e:
        print(f"✗ 웹캠 테스트 오류: {e}")
        return False

def test_audio_devices():
    """오디오 디바이스 테스트"""
    print("\n=== 오디오 디바이스 테스트 ===")
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        
        print(f"✓ 입력 디바이스 {len(input_devices)}개 발견")
        print(f"✓ 출력 디바이스 {len(output_devices)}개 발견")
        
        if input_devices and output_devices:
            return True
        else:
            print("✗ 오디오 디바이스가 충분하지 않음")
            return False
    except Exception as e:
        print(f"✗ 오디오 디바이스 테스트 오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("Face-VC 통합 GUI 테스트 시작\n")
    
    tests = [
        ("기본 Import", test_imports),
        ("FaceFusion Import", test_facefusion_imports),
        ("Seed-VC Import", test_seed_vc_imports),
        ("디바이스 설정", test_device),
        ("모델 파일", test_model_files),
        ("웹캠", test_webcam),
        ("오디오 디바이스", test_audio_devices),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name} 테스트 실패")
        except Exception as e:
            print(f"✗ {test_name} 테스트 오류: {e}")
    
    print(f"\n=== 테스트 결과 ===")
    print(f"통과: {passed}/{total}")
    
    if passed == total:
        print("✓ 모든 테스트 통과! GUI를 실행할 수 있습니다.")
        return True
    else:
        print("✗ 일부 테스트 실패. 문제를 해결한 후 다시 시도하세요.")
        return False

if __name__ == "__main__":
    main() 