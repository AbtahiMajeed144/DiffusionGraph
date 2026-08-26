from diffusiongraph.paths.linear_condition import LinearConditionPath
from diffusiongraph.paths.slerp_noise import SlerpNoisePath
from diffusiongraph.paths.tangential_geodesic import TangentialGeodesicPath
from diffusiongraph.paths.string_method import StringMethodPath

# Which EDM checkpoint each path type runs on -- see slerp_noise.py /
# tangential_geodesic.py docstrings for why this split exists.
PATH_USES_CONDITIONAL_MODEL = {
    "linear_condition": True,
    "slerp_noise": False,
    "tangential_geodesic": False,
    "string_method": False,
}

PATH_REGISTRY = {
    "linear_condition": LinearConditionPath,
    "slerp_noise": SlerpNoisePath,
    "tangential_geodesic": TangentialGeodesicPath,
    "string_method": StringMethodPath,
}


def build_path(name: str, **kwargs):
    if name not in PATH_REGISTRY:
        raise ValueError(f"Unknown path type '{name}'. Choices: {list(PATH_REGISTRY)}")
    cls = PATH_REGISTRY[name]
    return cls(**kwargs) if kwargs else cls()
