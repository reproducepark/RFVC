import gradio

from facefusion import state_manager
from facefusion.uis.components import source, webcam


def pre_check() -> bool:
	return True


def webcam_buttons():
	# 웹캠 버튼만 표시하는 함수
	start_button = gradio.Button(
		value="START",
		variant="primary",  # 기본 primary variant 사용
		size="lg"
	)
	stop_button = gradio.Button(
		value="STOP", 
		variant="secondary",  # secondary variant 사용
		size="lg"
	)
	
	# 버튼 이벤트 연결
	start_button.click(webcam.start_simple, outputs=None)
	stop_button.click(webcam.stop, outputs=None)
	
	return start_button, stop_button


def render() -> gradio.Blocks:
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

	with gradio.Blocks(title="FaceFusion Simple Webcam") as layout:
		with gradio.Column(scale=1):
			# SOURCE만 선택 가능
			with gradio.Blocks():
				source.render()
			
			# 웹캠 버튼만 표시 (화면 제거)
			with gradio.Blocks():
				start_btn, stop_btn = webcam_buttons()
	
	return layout


def listen() -> None:
	source.listen()
	# 웹캠 컴포넌트의 listen은 더 이상 필요하지 않음


def run(ui: gradio.Blocks) -> None:
	ui.launch(favicon_path='facefusion.ico', inbrowser=state_manager.get_item('open_browser')) 