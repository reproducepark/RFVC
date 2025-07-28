import os
import sys
import time
import torch
import torchaudio
import gradio as gr
import cv2
import numpy as np
import librosa
import sounddevice as sd
from typing import Generator
import yaml
from dotenv import load_dotenv

# 환경 설정
load_dotenv()
os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# facefusion 관련 import
sys.path.append('facefusion')
from facefusion import state_manager
from facefusion.face_analyser import get_average_face, get_many_faces
from facefusion.filesystem import filter_image_paths
from facefusion.vision import normalize_frame_color, read_static_images
from facefusion.processors.core import get_processors_modules
from facefusion.audio import create_empty_audio_frame

# seed-vc 관련 import
sys.path.append('seed-vc')
from modules.commons import recursive_munch, build_model, load_checkpoint
from modules.audio import mel_spectrogram
from modules.campplus.DTDNN import CAMPPlus
from hf_utils import load_custom_model_from_hf

@torch.no_grad()
def custom_infer(model_set,
                 reference_wav,
                 new_reference_wav_name,
                 input_wav_res,
                 block_frame_16k,
                 skip_head,
                 skip_tail,
                 return_length,
                 diffusion_steps,
                 inference_cfg_rate,
                 max_prompt_length,
                 cd_difference=2.0,
                 ):
    global prompt_condition, mel2, style2
    global reference_wav_name
    global prompt_len
    global ce_dit_difference
    (
        model,
        semantic_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    ) = model_set
    sr = mel_fn_args["sampling_rate"]
    hop_length = mel_fn_args["hop_size"]
    if ce_dit_difference != cd_difference:
        ce_dit_difference = cd_difference
        print(f"Setting ce_dit_difference to {cd_difference} seconds.")
    if prompt_condition is None or reference_wav_name != new_reference_wav_name or prompt_len != max_prompt_length:
        prompt_len = max_prompt_length
        print(f"Setting max prompt length to {max_prompt_length} seconds.")
        reference_wav = reference_wav[:int(sr * prompt_len)]
        reference_wav_tensor = torch.from_numpy(reference_wav).to(device)
        
        ori_waves_16k = torchaudio.functional.resample(reference_wav_tensor, sr, 16000)
        S_ori = semantic_fn(ori_waves_16k.unsqueeze(0))
        feat2 = torchaudio.compliance.kaldi.fbank(
            ori_waves_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = campplus_model(feat2.unsqueeze(0))
        
        mel2 = to_mel(reference_wav_tensor.unsqueeze(0))
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
        prompt_condition = model.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
        )[0]
        
        reference_wav_name = new_reference_wav_name

    converted_waves_16k = input_wav_res
    if device.type == "mps":
        start_event = torch.mps.event.Event(enable_timing=True)
        end_event = torch.mps.event.Event(enable_timing=True)
        torch.mps.synchronize()
    else:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

    start_event.record()
    S_alt = semantic_fn(converted_waves_16k.unsqueeze(0))
    end_event.record()
    if device.type == "mps":
        torch.mps.synchronize()  # MPS - Wait for the events to be recorded!
    else:
        torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Time taken for semantic_fn: {elapsed_time_ms}ms")

    ce_dit_frame_difference = int(ce_dit_difference * 50)
    S_alt = S_alt[:, ce_dit_frame_difference:]
    target_lengths = torch.LongTensor([(skip_head + return_length + skip_tail - ce_dit_frame_difference) / 50 * sr // hop_length]).to(S_alt.device)
    print(f"target_lengths: {target_lengths}")
    cond = model.length_regulator(
        S_alt, ylens=target_lengths , n_quantizers=3, f0=None
    )[0]
    cat_condition = torch.cat([prompt_condition, cond], dim=1)
    with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
        vc_target = model.cfm.inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
            mel2,
            style2,
            None,
            n_timesteps=diffusion_steps,
            inference_cfg_rate=inference_cfg_rate,
        )
        vc_target = vc_target[:, :, mel2.size(-1) :]
        print(f"vc_target.shape: {vc_target.shape}")
        vc_wave = vocoder_fn(vc_target).squeeze()
    output_len = return_length * sr // 50
    tail_len = skip_tail * sr // 50
    output = vc_wave[-output_len - tail_len: -tail_len]

    return output

# 전역 변수
device = None
flag_vc = False
flag_facefusion = False
prompt_len = 3
ce_dit_difference = 2.0
fp16 = False



# seed-vc 관련 전역 변수
prompt_condition, mel2, style2 = None, None, None
reference_wav_name = ""
model_set = None
stream = None

# seed-vc 설정
SEED_VC_CONFIG = {
    'diffusion_steps': 10,
    'inference_cfg_rate': 0.0,
    'max_prompt_length': 2.0,
    'block_time': 0.40,
    'crossfade_length': 0.04,
    'extra_time_ce': 7.0,
    'extra_time': 0.5,
    'extra_time_right': 0.02
}

class IntegratedGUI:
    def __init__(self):
        self.setup_device()
        self.setup_facefusion()
        self.setup_seed_vc()
        
        # seed-vc 관련 변수들
        self.reference_wav = None
        self.gui_config = SEED_VC_CONFIG.copy()
        self.zc = None
        self.block_frame = None
        self.block_frame_16k = None
        self.crossfade_frame = None
        self.sola_buffer_frame = None
        self.sola_search_frame = None
        self.extra_frame = None
        self.extra_frame_right = None
        self.input_wav = None
        self.input_wav_res = None
        self.sola_buffer = None
        self.skip_head = None
        self.skip_tail = None
        self.return_length = None
        self.fade_in_window = None
        self.fade_out_window = None
        self.resampler = None
        self.resampler2 = None
        self.vad_speech_detected = False
        self.set_speech_detected_false_at_end_flag = False
        
    def setup_device(self):
        global device
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
    
    def setup_facefusion(self):
        """facefusion 초기 설정"""
        # facefusion은 별도 프로세스로 실행하므로 여기서는 설정만
        print("FaceFusion 설정 완료 - 별도 프로세스로 실행됩니다.")
        
    def setup_seed_vc(self):
        """seed-vc 모델 로드"""
        global model_set
        try:
            # 고정된 설정으로 모델 로드
            checkpoint_path = "seed-vc/runs/yoon2/ft_model.pth"
            config_path = "seed-vc/runs/yoon2/config_dit_mel_seed_uvit_xlsr_tiny.yml"
            
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint file not found: {checkpoint_path}")
                print("Seed-VC 기능이 비활성화됩니다.")
                return
                
            if not os.path.exists(config_path):
                print(f"Warning: Config file not found: {config_path}")
                print("Seed-VC 기능이 비활성화됩니다.")
                return
                
            print("Loading Seed-VC models...")
            model_set = self.load_seed_vc_models(checkpoint_path, config_path)
            print("✓ Seed-VC models loaded successfully")
        except Exception as e:
            print(f"Error loading Seed-VC models: {e}")
            print("Seed-VC 기능이 비활성화됩니다.")
            model_set = None
    
    def load_seed_vc_models(self, checkpoint_path, config_path):
        """seed-vc 모델 로드 함수"""
        config = yaml.safe_load(open(config_path, "r"))
        model_params = recursive_munch(config["model_params"])
        model_params.dit_type = 'DiT'
        model = build_model(model_params, stage="DiT")
        sr = config["preprocess_params"]["sr"]

        # Load checkpoints
        model, _, _, _ = load_checkpoint(
            model,
            None,
            checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        for key in model:
            model[key].eval()
            model[key].to(device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        # Load CAMPPlus
        campplus_ckpt_path = load_custom_model_from_hf(
            "funasr/campplus", "campplus_cn_common.bin", config_filename=None
        )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        campplus_model.eval()
        campplus_model.to(device)

        # Load vocoder (HiFiGAN)
        from modules.hifigan.generator import HiFTGenerator
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor
        hift_config = yaml.safe_load(open('seed-vc/configs/hifigan.yml', 'r'))
        hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
        hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
        hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder_fn = hift_gen

        # Load speech tokenizer (XLSR)
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        model_name = config['model_params']['speech_tokenizer']['name']
        output_layer = config['model_params']['speech_tokenizer']['output_layer']
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
        wav2vec_model = wav2vec_model.to(device)
        wav2vec_model = wav2vec_model.eval()
        wav2vec_model = wav2vec_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = wav2vec_feature_extractor(ori_waves_16k_input_list,
                                                   return_tensors="pt",
                                                   return_attention_mask=True,
                                                   padding=True,
                                                   sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = wav2vec_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori

        # Mel spectrogram function
        mel_fn_args = {
            "n_fft": config['preprocess_params']['spect_params']['n_fft'],
            "win_size": config['preprocess_params']['spect_params']['win_length'],
            "hop_size": config['preprocess_params']['spect_params']['hop_length'],
            "num_mels": config['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": sr,
            "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

        return (
            model,
            semantic_fn,
            vocoder_fn,
            campplus_model,
            to_mel,
            mel_fn_args,
        )

    def update_source_face(self, files):
        """facefusion 소스 이미지 업데이트"""
        if files:
            # Gradio File 컴포넌트에서 파일 경로 추출
            if hasattr(files, 'name'):
                file_path = files.name
            elif isinstance(files, list):
                file_path = files[0].name
            else:
                file_path = str(files)
                
            # state_manager에 소스 이미지 경로 저장
            state_manager.set_item('source_paths', [file_path])
            print(f"소스 이미지 선택됨: {file_path}")
            return gr.Image(value=file_path, visible=True)
        else:
            # 소스 이미지 제거
            state_manager.clear_item('source_paths')
            print("소스 이미지가 선택되지 않았습니다.")
            return gr.Image(value=None, visible=False)

    def start_facefusion(self):
        """facefusion 웹캠 시작 (UDP 전송)"""
        global flag_facefusion
        if not flag_facefusion:
            flag_facefusion = True
            
            # facefusion을 별도 프로세스로 실행
            import subprocess
            import sys
            import os
            
            # 현재 작업 디렉토리를 facefusion 폴더로 변경
            original_cwd = os.getcwd()
            os.chdir('facefusion')
            
            try:
                # 소스 이미지 경로 가져오기
                source_paths = state_manager.get_item('source_paths')
                source_args = []
                if source_paths:
                    # 절대 경로로 변환
                    abs_source_paths = []
                    for path in source_paths:
                        if os.path.isabs(path):
                            abs_source_paths.append(path)
                        else:
                            abs_source_paths.append(os.path.abspath(path))
                    source_args = ['--source-paths'] + abs_source_paths
                
                # facefusion simple_webcam 레이아웃 실행 (필수 인수만)
                cmd = [
                    sys.executable, 'facefusion.py', 'run', 
                    '--ui-layouts', 'simple_webcam',
                    '--processors', 'face_swapper',
                    '--face-swapper-model', 'hyperswap_1a_256',
                    '--execution-providers', 'cuda',
                    '--log-level', 'info'
                ] + source_args
                
                print(f"실행 명령어: {' '.join(cmd)}")
                
                self.facefusion_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # 프로세스가 시작되었는지 확인
                import time
                time.sleep(2)
                
                if self.facefusion_process.poll() is None:
                    print("FaceFusion 프로세스 시작됨")
                    if source_paths:
                        print(f"소스 이미지: {source_paths}")
                    
                    # 프로세스 출력 모니터링 스레드 시작
                    import threading
                    self.monitor_thread = threading.Thread(target=self.monitor_facefusion_process)
                    self.monitor_thread.daemon = True
                    self.monitor_thread.start()
                    
                    return "FaceFusion UDP 스트림 시작됨 (udp://localhost:27000)"
                else:
                    # 프로세스가 즉시 종료된 경우
                    stdout, stderr = self.facefusion_process.communicate()
                    error_msg = f"FaceFusion 프로세스 시작 실패:\nstdout: {stdout}\nstderr: {stderr}"
                    print(error_msg)
                    flag_facefusion = False
                    return error_msg
                
            except Exception as e:
                print(f"FaceFusion 프로세스 시작 실패: {e}")
                flag_facefusion = False
                return f"FaceFusion 시작 실패: {e}"
            finally:
                os.chdir(original_cwd)
        return "FaceFusion 이미 실행 중"

    def monitor_facefusion_process(self):
        """facefusion 프로세스 출력 모니터링"""
        global flag_facefusion
        
        try:
            while flag_facefusion and self.facefusion_process and self.facefusion_process.poll() is None:
                # stdout 읽기
                stdout_line = self.facefusion_process.stdout.readline()
                if stdout_line:
                    print(f"[FaceFusion] {stdout_line.strip()}")
                
                # stderr 읽기
                stderr_line = self.facefusion_process.stderr.readline()
                if stderr_line:
                    print(f"[FaceFusion ERROR] {stderr_line.strip()}")
                
                import time
                time.sleep(0.1)
                
        except Exception as e:
            print(f"프로세스 모니터링 오류: {e}")
        finally:
            if self.facefusion_process and self.facefusion_process.poll() is not None:
                print("FaceFusion 프로세스가 종료되었습니다.")
                flag_facefusion = False







    def stop_facefusion(self):
        """facefusion 웹캠 정지"""
        global flag_facefusion
        flag_facefusion = False
        
        # facefusion 프로세스 종료
        if hasattr(self, 'facefusion_process') and self.facefusion_process:
            try:
                print("FaceFusion 프로세스 종료 중...")
                self.facefusion_process.terminate()
                
                # 잠시 대기 후 강제 종료
                import time
                time.sleep(3)
                
                if self.facefusion_process.poll() is None:
                    self.facefusion_process.kill()
                    print("FaceFusion 프로세스 강제 종료됨")
                else:
                    print("FaceFusion 프로세스 종료됨")
                    
            except Exception as e:
                print(f"FaceFusion 프로세스 종료 오류: {e}")
            finally:
                self.facefusion_process = None
            
        return "FaceFusion UDP 스트림 정지됨"

    def update_reference_audio(self, file):
        """seed-vc 참조 오디오 업데이트"""
        if file:
            # Gradio File 컴포넌트에서 파일 경로 추출
            if hasattr(file, 'name'):
                file_path = file.name
            else:
                file_path = str(file)
            return gr.Audio(value=file_path, visible=True)
        return gr.Audio(value=None, visible=False)

    def start_seed_vc(self, reference_file):
        """seed-vc 시작"""
        global flag_vc, stream, model_set
        if not flag_vc and model_set and reference_file:
            try:
                # 참조 오디오 파일 경로 추출
                if hasattr(reference_file, 'name'):
                    reference_audio_path = reference_file.name
                else:
                    reference_audio_path = str(reference_file)
                
                # GPU 캐시 정리
                if device.type == "mps":
                    torch.mps.empty_cache()
                else:
                    torch.cuda.empty_cache()
                
                # 참조 오디오 로드
                self.reference_wav, _ = librosa.load(
                    reference_audio_path, sr=model_set[-1]["sampling_rate"]
                )
                
                # 설정 초기화
                self.gui_config['samplerate'] = model_set[-1]["sampling_rate"]
                self.gui_config['channels'] = 1
                self.zc = self.gui_config['samplerate'] // 50
                
                # 프레임 계산
                self.block_frame = (
                    int(
                        np.round(
                            self.gui_config['block_time']
                            * self.gui_config['samplerate']
                            / self.zc
                        )
                    )
                    * self.zc
                )
                self.block_frame_16k = 320 * self.block_frame // self.zc
                self.crossfade_frame = (
                    int(
                        np.round(
                            self.gui_config['crossfade_length']
                            * self.gui_config['samplerate']
                            / self.zc
                        )
                    )
                    * self.zc
                )
                self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
                self.sola_search_frame = self.zc
                self.extra_frame = (
                    int(
                        np.round(
                            self.gui_config['extra_time_ce']
                            * self.gui_config['samplerate']
                            / self.zc
                        )
                    )
                    * self.zc
                )
                self.extra_frame_right = (
                    int(
                        np.round(
                            self.gui_config['extra_time_right']
                            * self.gui_config['samplerate']
                            / self.zc
                        )
                    )
                    * self.zc
                )
                
                # 버퍼 초기화
                self.input_wav = torch.zeros(
                    self.extra_frame
                    + self.crossfade_frame
                    + self.sola_search_frame
                    + self.block_frame
                    + self.extra_frame_right,
                    device=device,
                    dtype=torch.float32,
                )
                self.input_wav_res = torch.zeros(
                    320 * self.input_wav.shape[0] // self.zc,
                    device=device,
                    dtype=torch.float32,
                )
                self.sola_buffer = torch.zeros(
                    self.sola_buffer_frame, device=device, dtype=torch.float32
                )
                
                # 기타 설정
                self.skip_head = self.extra_frame // self.zc
                self.skip_tail = self.extra_frame_right // self.zc
                self.return_length = (
                    self.block_frame + self.sola_buffer_frame + self.sola_search_frame
                ) // self.zc
                
                # 페이드 윈도우
                self.fade_in_window = (
                    torch.sin(
                        0.5
                        * np.pi
                        * torch.linspace(
                            0.0,
                            1.0,
                            steps=self.sola_buffer_frame,
                            device=device,
                            dtype=torch.float32,
                        )
                    )
                    ** 2
                )
                self.fade_out_window = 1 - self.fade_in_window
                
                # 리샘플러
                self.resampler = torchaudio.transforms.Resample(
                    orig_freq=self.gui_config['samplerate'],
                    new_freq=16000,
                    dtype=torch.float32,
                ).to(device)
                
                if model_set[-1]["sampling_rate"] != self.gui_config['samplerate']:
                    self.resampler2 = torchaudio.transforms.Resample(
                        orig_freq=model_set[-1]["sampling_rate"],
                        new_freq=self.gui_config['samplerate'],
                        dtype=torch.float32,
                    ).to(device)
                else:
                    self.resampler2 = None
                
                # 오디오 스트림 설정
                stream = sd.Stream(
                    callback=self.audio_callback,
                    blocksize=self.block_frame,
                    samplerate=self.gui_config['samplerate'],
                    channels=self.gui_config['channels'],
                    dtype="float32"
                )
                stream.start()
                flag_vc = True
                return "Voice conversion started"
            except Exception as e:
                return f"Error starting voice conversion: {e}"
        return "Voice conversion already running or model not loaded"

    def stop_seed_vc(self):
        """seed-vc 정지"""
        global flag_vc, stream
        flag_vc = False
        if stream:
            stream.abort()
            stream.close()
            stream = None
        return "Voice conversion stopped"

    def audio_callback(self, indata, outdata, frames, times, status):
        """seed-vc 오디오 콜백"""
        global flag_vc, model_set
        if not flag_vc or model_set is None:
            outdata[:] = 0
            return
            
        try:
            start_time = time.perf_counter()
            indata = librosa.to_mono(indata.T)
            
            # 입력 버퍼 업데이트
            self.input_wav[: -self.block_frame] = self.input_wav[
                self.block_frame :
            ].clone()
            self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(device)
            
            # 16kHz 리샘플링
            self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
                self.block_frame_16k :
            ].clone()
            self.input_wav_res[-320 * (indata.shape[0] // self.zc + 1) :] = (
                torch.from_numpy(librosa.resample(
                    self.input_wav[-indata.shape[0] - 2 * self.zc :].cpu().numpy(), 
                    orig_sr=self.gui_config['samplerate'], 
                    target_sr=16000
                )[320:])
            )
            
            print(f"preprocess time: {time.perf_counter() - start_time:.2f}")
            
            # Voice conversion 추론
            if self.gui_config['extra_time_ce'] - self.gui_config['extra_time'] < 0:
                raise ValueError("Content encoder extra context must be greater than DiT extra context!")
            
            infer_start_time = time.perf_counter()
            infer_wav = custom_infer(
                model_set,
                self.reference_wav,
                "reference_audio",  # 고정된 이름 사용
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.skip_tail,
                self.return_length,
                int(self.gui_config['diffusion_steps']),
                self.gui_config['inference_cfg_rate'],
                self.gui_config['max_prompt_length'],
                self.gui_config['extra_time_ce'] - self.gui_config['extra_time'],
            )
            
            if self.resampler2 is not None:
                infer_wav = self.resampler2(infer_wav)
            
            print(f"inference time: {time.perf_counter() - infer_start_time:.2f}")
            
            # SOLA 알고리즘 적용
            conv_input = infer_wav[
                None, None, : self.sola_buffer_frame + self.sola_search_frame
            ]
            
            cor_nom = torch.nn.functional.conv1d(conv_input, self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(
                torch.nn.functional.conv1d(
                    conv_input**2,
                    torch.ones(1, 1, self.sola_buffer_frame, device=device),
                )
                + 1e-8
            )
            
            tensor = cor_nom[0, 0] / cor_den[0, 0]
            if tensor.numel() > 1:
                sola_offset = torch.argmax(tensor, dim=0).item()
            else:
                sola_offset = tensor.item()
            
            print(f"sola_offset = {int(sola_offset)}")
            
            # 출력 처리
            infer_wav = infer_wav[sola_offset:]
            infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
            infer_wav[: self.sola_buffer_frame] += (
                self.sola_buffer * self.fade_out_window
            )
            self.sola_buffer[:] = infer_wav[
                self.block_frame : self.block_frame + self.sola_buffer_frame
            ]
            
            # 출력 크기 맞추기
            output_audio = infer_wav[: self.block_frame].cpu().numpy()
            expected_size = outdata.shape[0]
            if len(output_audio) != expected_size:
                if len(output_audio) > expected_size:
                    output_audio = output_audio[:expected_size]
                else:
                    padding = np.zeros(expected_size - len(output_audio))
                    output_audio = np.concatenate([output_audio, padding])
            
            outdata[:] = output_audio.reshape(-1, 1)
            
            total_time = time.perf_counter() - start_time
            print(f"Total time: {total_time:.2f}")
            
        except Exception as e:
            print(f"Audio callback error: {e}")
            outdata[:] = 0



    def create_interface(self):
        """Gradio 인터페이스 생성"""
        with gr.Blocks(title="Face-VC Integrated GUI", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# Face-VC 통합 GUI")
            gr.Markdown("FaceFusion과 Seed-VC를 통합한 실시간 처리 GUI")
            
            with gr.Row():
                # FaceFusion 섹션
                with gr.Column(scale=1):
                    gr.Markdown("## FaceFusion")
                    
                    # 소스 이미지 선택
                    source_file = gr.File(
                        label="소스 이미지 선택",
                        file_count="single",
                        file_types=["image"]
                    )
                    source_image = gr.Image(label="선택된 소스 이미지", visible=False)
                    
                    # FaceFusion 상태 표시
                    facefusion_status = gr.Textbox(label="FaceFusion 상태", value="대기 중")
                    
                    # UDP 스트림 정보
                    gr.Markdown("**UDP 스트림 주소:** `udp://localhost:27000`")
                    gr.Markdown("VLC나 다른 미디어 플레이어에서 위 주소로 접속하여 스트림을 볼 수 있습니다.")
                    gr.Markdown("**사용법:** 1) 소스 이미지를 선택하고 2) FaceFusion 시작 버튼을 누르면 자동으로 소스 이미지가 적용됩니다.")
                    
                    # FaceFusion 컨트롤
                    with gr.Row():
                        facefusion_start_btn = gr.Button("FaceFusion 시작", variant="primary")
                        facefusion_stop_btn = gr.Button("FaceFusion 정지", variant="secondary")
                
                # Seed-VC 섹션
                with gr.Column(scale=1):
                    gr.Markdown("## Seed-VC")
                    
                    # 참조 오디오 선택
                    reference_file = gr.File(
                        label="참조 오디오 선택",
                        file_count="single",
                        file_types=["audio"]
                    )
                    reference_audio = gr.Audio(label="선택된 참조 오디오", visible=False)
                    
                    # Seed-VC 컨트롤
                    with gr.Row():
                        seed_vc_start_btn = gr.Button("Seed-VC 시작", variant="primary", interactive=model_set is not None)
                        seed_vc_stop_btn = gr.Button("Seed-VC 정지", variant="secondary")
                    
                    # 상태 표시
                    initial_status = "모델 로드됨 - 대기 중" if model_set is not None else "모델 로드 실패 - Seed-VC 비활성화"
                    vc_status = gr.Textbox(label="Voice Conversion 상태", value=initial_status)
            
            # 이벤트 연결
            source_file.change(
                self.update_source_face,
                inputs=source_file,
                outputs=source_image
            )
            
            reference_file.change(
                self.update_reference_audio,
                inputs=reference_file,
                outputs=reference_audio
            )
            
            facefusion_start_btn.click(
                self.start_facefusion,
                outputs=facefusion_status
            )
            
            facefusion_stop_btn.click(
                self.stop_facefusion,
                outputs=facefusion_status
            )
            
            seed_vc_start_btn.click(
                self.start_seed_vc,
                inputs=reference_file,
                outputs=vc_status
            )
            
            seed_vc_stop_btn.click(
                self.stop_seed_vc,
                outputs=vc_status
            )
        
        return interface

def main():
    """메인 함수"""
    gui = IntegratedGUI()
    interface = gui.create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main() 