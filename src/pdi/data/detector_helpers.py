from pdi.data.types import Detector, COLUMN_DETECTOR


def columns_to_detectors_masked(cols: list[str]) -> int:
    detectors = 0
    for col in cols:
        det = COLUMN_DETECTOR[col]
        detectors |= det.value
    return detectors


def columns_to_detectors(cols: list[str]) -> list[Detector]:
    detectors = columns_to_detectors_masked(cols)
    return detector_unmask(detectors)


def detector_mask(detectors: list[Detector]) -> int:
    mask = 0
    for d in detectors:
        mask |= d.value
    return mask


def detector_unmask(mask: int) -> list[Detector]:
    detectors = []
    i = 1
    while i <= mask:
        if (i & mask) > 0:
            detectors.append(Detector(i))
        i <<= 1
    return detectors
