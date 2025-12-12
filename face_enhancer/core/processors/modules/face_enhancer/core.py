from functools import lru_cache

import numpy

from face_enhancer.core import face_detector, face_landmarker, face_masker, inference_manager, state_manager
from face_enhancer.core.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from face_enhancer.core.face_analyser import scale_face
from face_enhancer.core.face_helper import paste_back, warp_face_by_face_landmark_5
from face_enhancer.core.face_masker import create_box_mask, create_occlusion_mask
from face_enhancer.core.face_selector import select_faces
from face_enhancer.core.filesystem import resolve_relative_path
from face_enhancer.core.processors.modules.face_enhancer.types import FaceEnhancerInputs, FaceEnhancerWeight
from face_enhancer.core.processors.types import ProcessorOutputs
from face_enhancer.core.thread_helper import thread_semaphore
from face_enhancer.core.types import Face, InferencePool, ModelOptions, ModelSet, VisionFrame
from face_enhancer.core.vision import blend_frame


@lru_cache()
def create_static_model_set(download_scope : str) -> ModelSet:
	return\
	{
		'gfpgan_1.4':
		{
			'__metadata__':
			{
				'vendor': 'TencentARC',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gfpgan_1.4.hash'),
					'path': resolve_relative_path('../.assets/models/gfpgan_1.4.hash')
				}
			},
			'sources':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gfpgan_1.4.onnx'),
					'path': resolve_relative_path('../.assets/models/gfpgan_1.4.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (512, 512)
		}
	}


def get_inference_pool() -> InferencePool:
	model_names = [ state_manager.get_item('face_enhancer_model') ]
	model_source_set = get_model_options().get('sources')

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	model_names = [ state_manager.get_item('face_enhancer_model') ]
	inference_manager.clear_inference_pool(__name__, model_names)


def get_model_options() -> ModelOptions:
	model_name = state_manager.get_item('face_enhancer_model')
	return create_static_model_set('full').get(model_name)


def pre_check() -> bool:
	model_hash_set = get_model_options().get('hashes')
	model_source_set = get_model_options().get('sources')

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def enhance_face(target_face : Face, temp_vision_frame : VisionFrame) -> VisionFrame:
	model_template = get_model_options().get('template')
	model_size = get_model_options().get('size')
	crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, target_face.landmark_set.get('5/68'), model_template, model_size)
	box_mask = create_box_mask(crop_vision_frame, state_manager.get_item('face_mask_blur'), (0, 0, 0, 0))
	crop_masks =\
	[
		box_mask
	]

	if 'occlusion' in state_manager.get_item('face_mask_types'):
		occlusion_mask = create_occlusion_mask(crop_vision_frame)
		crop_masks.append(occlusion_mask)

	crop_vision_frame = prepare_crop_frame(crop_vision_frame)
	face_enhancer_weight = numpy.array([ state_manager.get_item('face_enhancer_weight') ]).astype(numpy.double)
	crop_vision_frame = forward(crop_vision_frame, face_enhancer_weight)
	crop_vision_frame = normalize_crop_frame(crop_vision_frame)
	crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)
	paste_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
	temp_vision_frame = blend_paste_frame(temp_vision_frame, paste_vision_frame)
	return temp_vision_frame


def forward(crop_vision_frame : VisionFrame, face_enhancer_weight : FaceEnhancerWeight) -> VisionFrame:
	face_enhancer = get_inference_pool().get('face_enhancer')
	face_enhancer_inputs = {}

	for face_enhancer_input in face_enhancer.get_inputs():
		if face_enhancer_input.name == 'input':
			face_enhancer_inputs[face_enhancer_input.name] = crop_vision_frame
		if face_enhancer_input.name == 'weight':
			face_enhancer_inputs[face_enhancer_input.name] = face_enhancer_weight

	with thread_semaphore():
		crop_vision_frame = face_enhancer.run(None, face_enhancer_inputs)[0][0]

	return crop_vision_frame


def has_weight_input() -> bool:
	face_enhancer = get_inference_pool().get('face_enhancer')

	for deep_swapper_input in face_enhancer.get_inputs():
		if deep_swapper_input.name == 'weight':
			return True

	return False


def prepare_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
	crop_vision_frame = (crop_vision_frame - 0.5) / 0.5
	crop_vision_frame = numpy.expand_dims(crop_vision_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
	return crop_vision_frame


def normalize_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	crop_vision_frame = numpy.clip(crop_vision_frame, -1, 1)
	crop_vision_frame = (crop_vision_frame + 1) / 2
	crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
	crop_vision_frame = (crop_vision_frame * 255.0).round()
	crop_vision_frame = crop_vision_frame.astype(numpy.uint8)[:, :, ::-1]
	return crop_vision_frame


def blend_paste_frame(temp_vision_frame : VisionFrame, paste_vision_frame : VisionFrame) -> VisionFrame:
	face_enhancer_blend = 1 - (state_manager.get_item('face_enhancer_blend') / 100)
	temp_vision_frame = blend_frame(temp_vision_frame, paste_vision_frame, 1 - face_enhancer_blend)
	return temp_vision_frame


def process_frame(inputs : FaceEnhancerInputs) -> ProcessorOutputs:
	reference_vision_frame = inputs.get('reference_vision_frame')
	target_vision_frame = inputs.get('target_vision_frame')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')
	target_faces = select_faces(reference_vision_frame, target_vision_frame)

	if target_faces:
		for target_face in target_faces:
			target_face = scale_face(target_face, target_vision_frame, temp_vision_frame)
			temp_vision_frame = enhance_face(target_face, temp_vision_frame)

	return temp_vision_frame, temp_vision_mask
