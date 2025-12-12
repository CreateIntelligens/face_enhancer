from typing import Any, Literal, TypeAlias, TypedDict

from numpy.typing import NDArray

from face_enhancer.core.types import Mask, VisionFrame

FaceEnhancerInputs = TypedDict('FaceEnhancerInputs',
{
	'reference_vision_frame' : VisionFrame,
	'target_vision_frame' : VisionFrame,
	'temp_vision_frame' : VisionFrame,
	'temp_vision_mask' : Mask
})

FaceEnhancerModel = Literal['gfpgan_1.4']

FaceEnhancerWeight : TypeAlias = NDArray[Any]
