import random
from typing import Optional

import gradio

from facefusion import metadata, wording

METADATA_BUTTON : Optional[gradio.Button] = None
ACTION_BUTTON : Optional[gradio.Button] = None


def render() -> None:
	global METADATA_BUTTON
	global ACTION_BUTTON

	action = random.choice(
	[
		{
			'wording': wording.get('about.join_our_community'),
			'url': 'https://join.facefusion.io'
		},
		{
			'wording': wording.get('about.read_the_documentation'),
			'url': 'https://docs.facefusion.io'
		}
	])

	ACTION_BUTTON = gradio.Button(
		value = action.get('wording'),
		link = action.get('url'),
		size = 'sm'
	)
