import logging
from typing import List, Sequence

from face_enhancer.core.common_helper import create_float_range, create_int_range
from face_enhancer.core.types import Angle, DownloadProvider, DownloadProviderSet, DownloadScope, ExecutionProvider, ExecutionProviderSet, FaceDetectorModel, FaceDetectorSet, FaceLandmarkerModel, FaceMaskArea, FaceMaskAreaSet, FaceMaskRegion, FaceMaskRegionSet, FaceMaskType, FaceSelectorMode, FaceSelectorOrder, LogLevel, LogLevelSet, Score

face_detector_set : FaceDetectorSet =\
{
	'many': [ '640x640' ],
	'retinaface': [ '160x160', '320x320', '480x480', '512x512', '640x640' ],
	'scrfd': [ '160x160', '320x320', '480x480', '512x512', '640x640' ],
	'yolo_face': [ '640x640' ],
	'yunet': [ '640x640' ]
}
face_detector_models : List[FaceDetectorModel] = list(face_detector_set.keys())
face_landmarker_models : List[FaceLandmarkerModel] = [ 'many', '2dfan4', 'peppa_wutz' ]
face_selector_modes : List[FaceSelectorMode] = [ 'many', 'one', 'reference' ]
face_selector_orders : List[FaceSelectorOrder] = [ 'left-right', 'right-left', 'top-bottom', 'bottom-top', 'small-large', 'large-small', 'best-worst', 'worst-best' ]
face_mask_types : List[FaceMaskType] = [ 'box', 'occlusion', 'area', 'region' ]
face_mask_area_set : FaceMaskAreaSet =\
{
	'upper-face': [ 0, 1, 2, 31, 32, 33, 34, 35, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17 ],
	'lower-face': [ 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 35, 34, 33, 32, 31 ],
	'mouth': [ 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67 ]
}
face_mask_region_set : FaceMaskRegionSet =\
{
	'skin': 1,
	'left-eyebrow': 2,
	'right-eyebrow': 3,
	'left-eye': 4,
	'right-eye': 5,
	'glasses': 6,
	'nose': 10,
	'mouth': 11,
	'upper-lip': 12,
	'lower-lip': 13
}
face_mask_areas : List[FaceMaskArea] = list(face_mask_area_set.keys())
face_mask_regions : List[FaceMaskRegion] = list(face_mask_region_set.keys())

execution_provider_set : ExecutionProviderSet =\
{
	'cuda': 'CUDAExecutionProvider',
	'tensorrt': 'TensorrtExecutionProvider',
	'cpu': 'CPUExecutionProvider'
}
execution_providers : List[ExecutionProvider] = list(execution_provider_set.keys())

download_provider_set : DownloadProviderSet =\
{
	'github':
	{
		'urls':
		[
			'https://github.com'
		],
		'path': '/facefusion/facefusion-assets/releases/download/{base_name}/{file_name}'
	},
	'huggingface':
	{
		'urls':
		[
			'https://huggingface.co',
			'https://hf-mirror.com'
		],
		'path': '/facefusion/{base_name}/resolve/main/{file_name}'
	}
}
download_providers : List[DownloadProvider] = list(download_provider_set.keys())
download_scopes : List[DownloadScope] = [ 'lite', 'full' ]

log_level_set : LogLevelSet =\
{
	'error': logging.ERROR,
	'warn': logging.WARNING,
	'info': logging.INFO,
	'debug': logging.DEBUG
}
log_levels : List[LogLevel] = list(log_level_set.keys())

execution_thread_count_range : Sequence[int] = create_int_range(1, 32, 1)
face_detector_margin_range : Sequence[int] = create_int_range(0, 100, 1)
face_detector_angles : Sequence[Angle] = create_int_range(0, 270, 90)
face_detector_score_range : Sequence[Score] = create_float_range(0.0, 1.0, 0.05)
face_landmarker_score_range : Sequence[Score] = create_float_range(0.0, 1.0, 0.05)
face_mask_blur_range : Sequence[float] = create_float_range(0.0, 1.0, 0.05)
face_mask_padding_range : Sequence[int] = create_int_range(0, 100, 1)
reference_face_distance_range : Sequence[float] = create_float_range(0.0, 1.0, 0.05)
