import numpy as np
from PIL import Image, ImageDraw
from .pipeline_factory import create_pipeline


def _detect_skin_boxes(image, min_area=500):
    arr = np.array(image.convert("YCbCr"))
    cb = arr[:, :, 1]
    cr = arr[:, :, 2]
    mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not visited[y, x]:
                stack = [(y, x)]
                visited[y, x] = True
                min_x = max_x = x
                min_y = max_y = y
                area = 0
                while stack:
                    cy, cx = stack.pop()
                    area += 1
                    if cx < min_x:
                        min_x = cx
                    if cx > max_x:
                        max_x = cx
                    if cy < min_y:
                        min_y = cy
                    if cy > max_y:
                        max_y = cy
                    for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if area >= min_area:
                    boxes.append((min_x, min_y, max_x + 1, max_y + 1))
    return boxes


def apply_autodetailer(image, cfg):
    boxes = _detect_skin_boxes(image)
    if not boxes:
        print('[autodetailer] nothing detected')
        return image

    pipe = create_pipeline(cfg, img2img=True)
    for left, top, right, bottom in boxes:
        crop = image.crop((left, top, right, bottom))
        res = pipe(
            image=crop,
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt,
            num_inference_steps=cfg.num_steps,
            guidance_scale=cfg.cfg_scale,
            strength=0.6,
        ).images[0]
        image.paste(res, (left, top))
    return image
