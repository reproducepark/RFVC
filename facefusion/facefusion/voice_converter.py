import os
import sys
import time
import torch
import torchaudio
import yaml
import numpy as np
from typing import Optional, Tuple, Any

# 오디오 처리 라이브러리들
try:
    import librosa
    import sounddevice as sd
    AUDIO_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Audio libraries not available: {e}")
    print("Voice conversion will be disabled")
    AUDIO_LIBS_AVAILABLE = False
    librosa = None
    sd = None

# seed-vc 관련 import
# 현재 facefusion 디렉토리에서 seed-vc 디렉토리로의 경로 설정
current_dir = os.path.dirname(__file__)
facefusion_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(facefusion_dir)
seed_vc_path = os.path.join(project_root, 'seed-vc')
sys.path.append(seed_vc_path)

try:
    from modules.commons import recursive_munch, build_model, load_checkpoint
    from modules.audio import mel_spectrogram
    from modules.campplus.DTDNN import CAMPPlus
    from hf_utils import load_custom_model_from_hf
except ImportError as e:
    print(f"Warning: Could not import seed-vc modules: {e}")
    print("Voice conversion will be disabled")
    recursive_munch = None
    build_model = None
    load_checkpoint = None
    mel_spectrogram = None
    CAMPPlus = None
    load_custom_model_from_hf = None

class VoiceConverter:
    def __init__(self):
        # 오디오 라이브러리 가용성 확인
        if not AUDIO_LIBS_AVAILABLE:
            print("Audio libraries not available - VoiceConverter disabled")
            self.available = False
            return
            
        self.device = self._setup_device()
        self.model_set = None
        self.stream = None
        self.is_running = False
        self.available = True
        
        # seed-vc 설정
        self.config = {
            'diffusion_steps': 10,
            'inference_cfg_rate': 0.0,
            'max_prompt_length': 2.0,
            'block_time': 0.40,
            'crossfade_length': 0.04,
            'extra_time_ce': 7.0,
            'extra_time': 0.5,
            'extra_time_right': 0.02
        }
        
        # 전역 변수들
        self.prompt_condition = None
        self.mel2 = None
        self.style2 = None
        self.reference_wav_name = ""
        self.prompt_len = 3
        self.ce_dit_difference = 2.0
        self.fp16 = False
        
        # 오디오 처리 관련 변수들
        self.reference_wav = None
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
        
    def _setup_device(self):
        """디바이스 설정"""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Voice Converter using device: {device}")
        return device
    
    def load_models(self, checkpoint_path: str, config_path: str) -> bool:
        """seed-vc 모델 로드"""
        # 모듈이 로드되지 않은 경우
        if any(module is None for module in [recursive_munch, build_model, load_checkpoint, mel_spectrogram, CAMPPlus, load_custom_model_from_hf]):
            print("Seed-VC modules not available")
            return False
            
        try:
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint file not found: {checkpoint_path}")
                return False
                
            if not os.path.exists(config_path):
                print(f"Warning: Config file not found: {config_path}")
                return False
                
            print("Loading Seed-VC models...")
            self.model_set = self._load_seed_vc_models(checkpoint_path, config_path)
            print("✓ Seed-VC models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading Seed-VC models: {e}")
            return False
    
    def _load_seed_vc_models(self, checkpoint_path: str, config_path: str):
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
            model[key].to(self.device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        # Load CAMPPlus
        campplus_ckpt_path = load_custom_model_from_hf(
            "funasr/campplus", "campplus_cn_common.bin", config_filename=None
        )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        campplus_model.eval()
        campplus_model.to(self.device)

        # Load vocoder (HiFiGAN)
        from modules.hifigan.generator import HiFTGenerator
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor
        
        # 현재 디렉토리에서 seed-vc 디렉토리로의 경로 설정
        current_dir = os.path.dirname(__file__)
        facefusion_dir = os.path.dirname(current_dir)
        project_root = os.path.dirname(facefusion_dir)
        seed_vc_path = os.path.join(project_root, 'seed-vc')
        
        hift_config_path = os.path.join(seed_vc_path, 'configs', 'hifigan.yml')
        hift_config = yaml.safe_load(open(hift_config_path, 'r'))
        hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
        hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
        hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
        hift_gen.eval()
        hift_gen.to(self.device)
        vocoder_fn = hift_gen

        # Load speech tokenizer (XLSR)
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        model_name = config['model_params']['speech_tokenizer']['name']
        output_layer = config['model_params']['speech_tokenizer']['output_layer']
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
        wav2vec_model = wav2vec_model.to(self.device)
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
                                                   sampling_rate=16000).to(self.device)
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
    
    @torch.no_grad()
    def custom_infer(self, model_set, reference_wav, new_reference_wav_name, input_wav_res,
                    block_frame_16k, skip_head, skip_tail, return_length, diffusion_steps,
                    inference_cfg_rate, max_prompt_length, cd_difference=2.0):
        """Voice conversion 추론"""
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
        
        if self.ce_dit_difference != cd_difference:
            self.ce_dit_difference = cd_difference
            print(f"Setting ce_dit_difference to {cd_difference} seconds.")
            
        if self.prompt_condition is None or self.reference_wav_name != new_reference_wav_name or self.prompt_len != max_prompt_length:
            self.prompt_len = max_prompt_length
            print(f"Setting max prompt length to {max_prompt_length} seconds.")
            reference_wav = reference_wav[:int(sr * self.prompt_len)]
            reference_wav_tensor = torch.from_numpy(reference_wav).to(self.device)
            
            ori_waves_16k = torchaudio.functional.resample(reference_wav_tensor, sr, 16000)
            S_ori = semantic_fn(ori_waves_16k.unsqueeze(0))
            feat2 = torchaudio.compliance.kaldi.fbank(
                ori_waves_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000
            )
            feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
            self.style2 = campplus_model(feat2.unsqueeze(0))
            
            self.mel2 = to_mel(reference_wav_tensor.unsqueeze(0))
            target2_lengths = torch.LongTensor([self.mel2.size(2)]).to(self.mel2.device)
            self.prompt_condition = model.length_regulator(
                S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
            )[0]
            
            self.reference_wav_name = new_reference_wav_name

        converted_waves_16k = input_wav_res
        if self.device.type == "mps":
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
        if self.device.type == "mps":
            torch.mps.synchronize()
        else:
            torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Time taken for semantic_fn: {elapsed_time_ms}ms")

        ce_dit_frame_difference = int(self.ce_dit_difference * 50)
        S_alt = S_alt[:, ce_dit_frame_difference:]
        target_lengths = torch.LongTensor([(skip_head + return_length + skip_tail - ce_dit_frame_difference) / 50 * sr // hop_length]).to(S_alt.device)
        print(f"target_lengths: {target_lengths}")
        cond = model.length_regulator(
            S_alt, ylens=target_lengths , n_quantizers=3, f0=None
        )[0]
        cat_condition = torch.cat([self.prompt_condition, cond], dim=1)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.fp16 else torch.float32):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(self.mel2.device),
                self.mel2,
                self.style2,
                None,
                n_timesteps=diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, self.mel2.size(-1) :]
            print(f"vc_target.shape: {vc_target.shape}")
            vc_wave = vocoder_fn(vc_target).squeeze()
        output_len = return_length * sr // 50
        tail_len = skip_tail * sr // 50
        output = vc_wave[-output_len - tail_len: -tail_len]

        return output
    
    def start_voice_conversion(self, reference_audio_path: str) -> bool:
        """Voice conversion 시작"""
        if not hasattr(self, 'available') or not self.available:
            print("VoiceConverter not available")
            return False
            
        if not self.model_set or self.is_running:
            return False
            
        try:
            # GPU 캐시 정리
            if self.device.type == "mps":
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()
            
            # 참조 오디오 로드
            self.reference_wav, _ = librosa.load(
                reference_audio_path, sr=self.model_set[-1]["sampling_rate"]
            )
            
            # 설정 초기화
            self.config['samplerate'] = self.model_set[-1]["sampling_rate"]
            self.config['channels'] = 1
            self.zc = self.config['samplerate'] // 50
            
            # 프레임 계산
            self.block_frame = (
                int(
                    np.round(
                        self.config['block_time']
                        * self.config['samplerate']
                        / self.zc
                    )
                )
                * self.zc
            )
            self.block_frame_16k = 320 * self.block_frame // self.zc
            self.crossfade_frame = (
                int(
                    np.round(
                        self.config['crossfade_length']
                        * self.config['samplerate']
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
                        self.config['extra_time_ce']
                        * self.config['samplerate']
                        / self.zc
                    )
                )
                * self.zc
            )
            self.extra_frame_right = (
                int(
                    np.round(
                        self.config['extra_time_right']
                        * self.config['samplerate']
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
                device=self.device,
                dtype=torch.float32,
            )
            self.input_wav_res = torch.zeros(
                320 * self.input_wav.shape[0] // self.zc,
                device=self.device,
                dtype=torch.float32,
            )
            self.sola_buffer = torch.zeros(
                self.sola_buffer_frame, device=self.device, dtype=torch.float32
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
                        device=self.device,
                        dtype=torch.float32,
                    )
                )
                ** 2
            )
            self.fade_out_window = 1 - self.fade_in_window
            
            # 리샘플러
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=self.config['samplerate'],
                new_freq=16000,
                dtype=torch.float32,
            ).to(self.device)
            
            if self.model_set[-1]["sampling_rate"] != self.config['samplerate']:
                self.resampler2 = torchaudio.transforms.Resample(
                    orig_freq=self.model_set[-1]["sampling_rate"],
                    new_freq=self.config['samplerate'],
                    dtype=torch.float32,
                ).to(self.device)
            else:
                self.resampler2 = None
            
            # 오디오 스트림 설정
            self.stream = sd.Stream(
                callback=self.audio_callback,
                blocksize=self.block_frame,
                samplerate=self.config['samplerate'],
                channels=self.config['channels'],
                dtype="float32"
            )
            self.stream.start()
            self.is_running = True
            return True
        except Exception as e:
            print(f"Error starting voice conversion: {e}")
            return False
    
    def stop_voice_conversion(self):
        """Voice conversion 정지"""
        if not hasattr(self, 'available') or not self.available:
            return
            
        self.is_running = False
        if self.stream:
            self.stream.abort()
            self.stream.close()
            self.stream = None
    
    def audio_callback(self, indata, outdata, frames, times, status):
        """오디오 콜백"""
        if not hasattr(self, 'available') or not self.available:
            outdata[:] = 0
            return
            
        if not self.is_running or self.model_set is None:
            outdata[:] = 0
            return
            
        try:
            start_time = time.perf_counter()
            indata = librosa.to_mono(indata.T)
            
            # 입력 버퍼 업데이트
            self.input_wav[: -self.block_frame] = self.input_wav[
                self.block_frame :
            ].clone()
            self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(self.device)
            
            # 16kHz 리샘플링
            self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
                self.block_frame_16k :
            ].clone()
            self.input_wav_res[-320 * (indata.shape[0] // self.zc + 1) :] = (
                torch.from_numpy(librosa.resample(
                    self.input_wav[-indata.shape[0] - 2 * self.zc :].cpu().numpy(), 
                    orig_sr=self.config['samplerate'], 
                    target_sr=16000
                )[320:])
            )
            
            print(f"preprocess time: {time.perf_counter() - start_time:.2f}")
            
            # Voice conversion 추론
            if self.config['extra_time_ce'] - self.config['extra_time'] < 0:
                raise ValueError("Content encoder extra context must be greater than DiT extra context!")
            
            infer_start_time = time.perf_counter()
            infer_wav = self.custom_infer(
                self.model_set,
                self.reference_wav,
                "reference_audio",  # 고정된 이름 사용
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.skip_tail,
                self.return_length,
                int(self.config['diffusion_steps']),
                self.config['inference_cfg_rate'],
                self.config['max_prompt_length'],
                self.config['extra_time_ce'] - self.config['extra_time'],
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
                    torch.ones(1, 1, self.sola_buffer_frame, device=self.device),
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