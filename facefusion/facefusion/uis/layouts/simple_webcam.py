import gradio as gr
import os

from facefusion import state_manager
from facefusion.uis.components import source, webcam
from facefusion.voice_converter import VoiceConverter


def pre_check() -> bool:
	return True


def webcam_buttons():
	# 웹캠 버튼만 표시하는 함수
	start_button = gr.Button(
		value="START",
		variant="primary",  # 기본 primary variant 사용
		size="lg"
	)
	stop_button = gr.Button(
		value="STOP", 
		variant="secondary",  # secondary variant 사용
		size="lg"
	)
	
	# 버튼 이벤트 연결
	start_button.click(webcam.start_simple, outputs=None)
	stop_button.click(webcam.stop, outputs=None)
	
	return start_button, stop_button


def voice_converter_buttons():
	# Voice Converter 버튼들
	vc_start_button = gr.Button(
		value="VOICE START",
		variant="primary",
		size="lg"
	)
	vc_stop_button = gr.Button(
		value="VOICE STOP", 
		variant="secondary",
		size="lg"
	)
	
	# 버튼 이벤트 연결
	vc_start_button.click(start_voice_conversion, outputs=None)
	vc_stop_button.click(stop_voice_conversion, outputs=None)
	
	return vc_start_button, vc_stop_button


def start_voice_conversion():
	"""Voice conversion 시작"""
	global voice_converter
	if voice_converter and hasattr(voice_converter, 'available') and voice_converter.available and voice_converter.model_set:
		# 참조 오디오 파일 경로 가져오기
		reference_audio_path = state_manager.get_item('reference_audio_path')
		if reference_audio_path and os.path.exists(reference_audio_path):
			success = voice_converter.start_voice_conversion(reference_audio_path)
			if success:
				print("Voice conversion started successfully")
			else:
				print("Failed to start voice conversion")
		else:
			print("Reference audio file not found or not selected")
	else:
		print("Voice converter not initialized, not available, or models not loaded")


def stop_voice_conversion():
	"""Voice conversion 정지"""
	global voice_converter
	if voice_converter and hasattr(voice_converter, 'available') and voice_converter.available:
		voice_converter.stop_voice_conversion()
		print("Voice conversion stopped")
	else:
		print("Voice converter not available")


def render() -> gr.Blocks:
	# 고정된 설정들을 미리 설정
	state_manager.set_item('processors', ['face_swapper'])
	state_manager.set_item('face_swapper_model', 'hyperswap_1a_256')
	state_manager.set_item('face_swapper_pixel_boost', '256x256')
	state_manager.set_item('execution_providers', ['cuda'])
	state_manager.set_item('execution_thread_count', 32)  # 8코어 16스레드 CPU에 최적화
	state_manager.set_item('webcam_device_id', 0)
	state_manager.set_item('webcam_mode', 'udp')
	state_manager.set_item('webcam_resolution', '1280x720')
	state_manager.set_item('webcam_fps', 30)

	with gr.Blocks(title="RFVC", theme=gr.themes.Soft()) as layout:
		gr.Markdown("# RFVC")
		gr.Markdown("Real-Time Face Voice Converter	")
		
		with gr.Row():
			# FaceFusion 섹션
			with gr.Column(scale=1):
				gr.Markdown("## FaceFusion")
				
				# SOURCE만 선택 가능
				with gr.Blocks():
					source.render()
				

				
				# 웹캠 버튼들
				with gr.Row():
					start_btn, stop_btn = webcam_buttons()
			
			# Voice Converter 섹션
			with gr.Column(scale=1):
				gr.Markdown("## Voice Converter")
				
				# Voice Converter 참조 오디오 선택
				with gr.Blocks():
					reference_audio = gr.File(
						label="SOURCE",
						file_count="single",
						file_types=["audio"]
					)
					
					# 참조 오디오 선택 이벤트
					def update_reference_audio(file):
						if file:
							if hasattr(file, 'name'):
								file_path = file.name
							else:
								file_path = str(file)
							state_manager.set_item('reference_audio_path', file_path)
							print(f"참조 오디오 선택됨: {file_path}")
							return gr.Audio(value=file_path, visible=True)
						else:
							state_manager.clear_item('reference_audio_path')
							print("참조 오디오가 선택되지 않았습니다.")
							return gr.Audio(value=None, visible=False)
					
					reference_audio_display = gr.Audio(label="선택된 참조 오디오", visible=False)
					reference_audio.change(update_reference_audio, inputs=reference_audio, outputs=reference_audio_display)
				
				# Voice Converter 버튼들
				with gr.Row():
					vc_start_btn, vc_stop_btn = voice_converter_buttons()
	
	return layout


def listen() -> None:
	source.listen()
	# 웹캠 컴포넌트의 listen은 더 이상 필요하지 않음


def run(ui: gr.Blocks) -> None:
	# Voice Converter 초기화
	global voice_converter
	try:
		voice_converter = VoiceConverter()
		
		# VoiceConverter가 사용 가능한지 확인
		if hasattr(voice_converter, 'available') and voice_converter.available:
			# 모델 로드 시도
			# 현재 디렉토리에서 seed-vc 디렉토리로의 경로 설정
			current_dir = os.path.dirname(__file__)
			layouts_dir = os.path.dirname(current_dir)
			components_dir = os.path.dirname(layouts_dir)
			uis_dir = os.path.dirname(components_dir)
			facefusion_dir = os.path.dirname(uis_dir)
			project_root = os.path.dirname(facefusion_dir)
			
			checkpoint_path = os.path.join(project_root, "seed-vc", "runs", "yoon2", "ft_model.pth")
			config_path = os.path.join(project_root, "seed-vc", "runs", "yoon2", "config_dit_mel_seed_uvit_xlsr_tiny.yml")
			
			if os.path.exists(checkpoint_path) and os.path.exists(config_path):
				success = voice_converter.load_models(checkpoint_path, config_path)
				if success:
					print("✓ Voice Converter initialized successfully")
				else:
					print("✗ Failed to initialize Voice Converter")
					voice_converter = None
			else:
				print("✗ Voice Converter model files not found - Voice conversion will be disabled")
				voice_converter = None
		else:
			print("✗ Voice Converter not available - Voice conversion will be disabled")
			voice_converter = None
	except Exception as e:
		print(f"✗ Error initializing Voice Converter: {e}")
		voice_converter = None
	
	ui.launch(favicon_path='facefusion.ico', inbrowser=state_manager.get_item('open_browser')) 