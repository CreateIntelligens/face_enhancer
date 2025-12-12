import base64
import os
import sys
from pathlib import Path
from typing import Optional
import warnings

# Suppress ONNX Runtime warnings
os.environ.setdefault('ORT_LOGGING_LEVEL', '3')

# Custom stderr filter to suppress specific ONNX Runtime warnings
class StderrFilter:
	def __init__(self, original_stderr):
		self.original_stderr = original_stderr
		self.suppress_patterns = [
			'GPU device discovery failed',
			'ReadFileContents Failed to open file'
		]
	
	def write(self, text):
		if not any(pattern in text for pattern in self.suppress_patterns):
			self.original_stderr.write(text)
	
	def flush(self):
		self.original_stderr.flush()
	
	def fileno(self):
		return self.original_stderr.fileno()

sys.stderr = StderrFilter(sys.stderr)

import cv2
import numpy
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from face_enhancer.core import face_classifier, face_detector, face_landmarker, face_recognizer, logger, state_manager
from face_enhancer.core.execution import get_available_execution_providers
from face_enhancer.core.processors.modules.face_enhancer import core as ff_face_enhancer
from face_enhancer.core.types import Mask, VisionFrame


class EnhanceRequest(BaseModel):
	target_image: str = Field(..., description = 'Base64-encoded target image (JPEG/PNG).')
	reference_image: Optional[str] = Field(None, description = 'Optional base64 reference image. Defaults to target.')
	face_enhancer_model: str = Field('gfpgan_1.4', description = 'Face enhancer model name.')
	face_enhancer_blend: int = Field(80, ge = 0, le = 100, description = 'Blend percentage (higher = more enhanced face).')
	face_enhancer_weight: float = Field(0.5, ge = 0.0, le = 1.0, description = 'Model weight (if supported by the model).')
	face_selector_mode: Optional[str] = Field(None, description = 'Face selector mode: many | one | reference.')
	face_selector_order: Optional[str] = Field(None, description = 'Ordering when mode=one or many.')
	reference_face_position: Optional[int] = Field(None, ge = 0, description = 'Index in reference gallery when mode=reference.')
	reference_face_distance: Optional[float] = Field(None, ge = 0.0, le = 1.0, description = 'Distance threshold for reference matching.')


class EnhanceResponse(BaseModel):
	image_base64: str


app = FastAPI(title = 'Face Enhancer Service', version = '0.1.0')
_state_initialized = False
STATIC_DIR = Path(__file__).resolve().parent / 'static'

if STATIC_DIR.exists():
	app.mount('/static', StaticFiles(directory = STATIC_DIR), name = 'static')


def _init_defaults() -> None:
	"""
	初始化 state manager 的預設值。
	"""
	global _state_initialized
	if _state_initialized:
		return

	# Execution defaults
	available_execution_providers = get_available_execution_providers()
	execution_providers = [ 'cuda', 'cpu' ] if 'cuda' in available_execution_providers else [ 'cpu' ]

	print(f"\n[CHECK] Configured Execution Providers: {execution_providers}")
	if 'cuda' in execution_providers:
		print("[CHECK] GPU (CUDA) is ENABLED and selected.\n")
	else:
		print("[CHECK] GPU is NOT enabled. Using CPU.\n")

	state_manager.init_item('execution_device_ids', [ 0 ])
	state_manager.init_item('execution_providers', execution_providers)
	state_manager.init_item('execution_thread_count', 4)
	state_manager.init_item('download_scope', 'full')
	state_manager.init_item('download_providers', [ 'github', 'huggingface' ])
	state_manager.init_item('log_level', 'info')

	# Detector / landmarker defaults
	state_manager.init_item('face_detector_model', 'yolo_face')
	state_manager.init_item('face_detector_size', '640x640')
	state_manager.init_item('face_detector_margin', [ 0, 0, 0, 0 ])
	state_manager.init_item('face_detector_angles', [ 0 ])
	state_manager.init_item('face_detector_score', 0.5)
	state_manager.init_item('face_landmarker_model', '2dfan4')
	state_manager.init_item('face_landmarker_score', 0.5)

	# Face selector defaults
	state_manager.init_item('face_selector_mode', 'reference')
	state_manager.init_item('face_selector_order', 'large-small')
	state_manager.init_item('reference_face_position', 0)
	state_manager.init_item('reference_face_distance', 0.3)

	# Face mask defaults
	state_manager.init_item('face_mask_types', [ 'box' ])
	state_manager.init_item('face_mask_blur', 0.3)
	state_manager.init_item('face_mask_padding', [ 0, 0, 0, 0 ])

	# Enhancer defaults
	state_manager.init_item('face_enhancer_model', 'gfpgan_1.4')
	state_manager.init_item('face_enhancer_blend', 80)
	state_manager.init_item('face_enhancer_weight', 0.5)

	_state_initialized = True


def _decode_image(data_uri: str) -> VisionFrame:
	if ',' in data_uri and data_uri.strip().startswith('data:'):
		_, base64_data = data_uri.split(',', 1)
	else:
		base64_data = data_uri

	try:
		raw_bytes = base64.b64decode(base64_data)
	except Exception as exc:
		raise HTTPException(status_code = 400, detail = 'Invalid base64 image') from exc

	image_array = numpy.frombuffer(raw_bytes, dtype = numpy.uint8)
	vision_frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

	if vision_frame is None:
		raise HTTPException(status_code = 400, detail = 'Unable to decode image')
	return vision_frame


def _encode_image(vision_frame : VisionFrame) -> str:
	success, buffer = cv2.imencode('.jpg', vision_frame)
	if not success:
		raise HTTPException(status_code = 500, detail = 'Failed to encode output image')
	return 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('ascii')


def _apply_request_to_state(payload : EnhanceRequest) -> None:
	state_manager.set_item('face_enhancer_model', payload.face_enhancer_model)
	state_manager.set_item('face_enhancer_blend', payload.face_enhancer_blend)
	state_manager.set_item('face_enhancer_weight', payload.face_enhancer_weight)

	if payload.face_selector_mode is not None:
		state_manager.set_item('face_selector_mode', payload.face_selector_mode)
	if payload.face_selector_order is not None:
		state_manager.set_item('face_selector_order', payload.face_selector_order)
	if payload.reference_face_position is not None:
		state_manager.set_item('reference_face_position', payload.reference_face_position)
	if payload.reference_face_distance is not None:
		state_manager.set_item('reference_face_distance', payload.reference_face_distance)


def _ensure_face_enhancer_ready() -> None:
	"""
	Make sure the selected face enhancer model is downloaded and loaded.
	"""
	ff_face_enhancer.pre_check()
	inference_pool = ff_face_enhancer.get_inference_pool()

	if not inference_pool.get('face_enhancer'):
		ff_face_enhancer.clear_inference_pool()
		inference_pool = ff_face_enhancer.get_inference_pool()

	if not inference_pool.get('face_enhancer'):
		logger.error('Face enhancer model "%s" not loaded; check model downloads and .assets permissions.', state_manager.get_item('face_enhancer_model'), __name__)
		raise HTTPException(status_code = 503, detail = 'Face enhancer model not loaded yet. Please retry after models finish downloading.')


def _ensure_models() -> None:
	face_detector.pre_check()
	face_landmarker.pre_check()
	face_recognizer.pre_check()
	face_classifier.pre_check()
	ff_face_enhancer.pre_check()


def enhance(payload : EnhanceRequest) -> EnhanceResponse:
	_init_defaults()
	_apply_request_to_state(payload)
	_ensure_face_enhancer_ready()

	target_frame = _decode_image(payload.target_image)
	reference_frame = _decode_image(payload.reference_image) if payload.reference_image else target_frame

	# Minimal mask placeholder; face_enhancer returns the same mask unchanged.
	temp_mask : Mask = numpy.ones(target_frame.shape[:2], dtype = numpy.uint8)
	temp_frame = target_frame.copy()

	enhanced_frame, _ = ff_face_enhancer.process_frame(
	{
		'reference_vision_frame': reference_frame,
		'target_vision_frame': target_frame,
		'temp_vision_frame': temp_frame,
		'temp_vision_mask': temp_mask
	})

	return EnhanceResponse(
		image_base64 = _encode_image(enhanced_frame)
	)


@app.on_event('startup')
async def _startup() -> None:  # pragma: no cover - FastAPI lifecycle
	_init_defaults()
	try:
		_ensure_models()
	except Exception as exc:
		# Keep server alive; downloads may fail if offline.
		logger.warn(f'Model preload failed: {exc}', __name__)


@app.get('/health')
async def health() -> dict[str, str]:
	return { 'status': 'ok' }


@app.get('/', response_class = HTMLResponse)
async def demo_page() -> HTMLResponse:
	demo_file = STATIC_DIR / 'demo.html'
	if not demo_file.exists():
		raise HTTPException(status_code = 404, detail = 'Demo page not found')
	return HTMLResponse(demo_file.read_text(encoding = 'utf-8'))


@app.post('/enhance', response_model = EnhanceResponse)
async def enhance_endpoint(payload : EnhanceRequest) -> EnhanceResponse:
	try:
		return enhance(payload)
	finally:
		# Free memory between requests; pools will re-create lazily.
		ff_face_enhancer.clear_inference_pool()
