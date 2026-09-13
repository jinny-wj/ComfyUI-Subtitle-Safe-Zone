"""CPU-only region planning. White protection pixels mean: do not cover.

No learned subject detector is used here. All scores are heuristic.
"""
import math
import numpy as np


def rectangle_means(plane, boxes):
    integral = np.pad(plane.astype(np.float64), ((1, 0), (1, 0)))
    integral = integral.cumsum(0).cumsum(1)
    x, y, w, h = np.asarray(boxes, dtype=int).T
    return (integral[y+h, x+w] - integral[y, x+w]
            - integral[y+h, x] + integral[y, x]) / (w*h)


def candidate_boxes(width, height, box_width, box_height, margin):
    if min(width, height, box_width, box_height) < 1 or margin < 0:
        raise ValueError("画面、字幕框尺寸必须为正，边距不能为负。")
    if box_width + 2*margin > width or box_height + 2*margin > height:
        raise ValueError("字幕框加边距超出画面，请减小字幕框比例或边距。")
    xs = sorted(set(np.linspace(margin, width-margin-box_width, 9).round().astype(int)))
    ys = sorted(set(np.linspace(margin, height-margin-box_height, 9).round().astype(int)))
    return [(int(x), int(y), box_width, box_height) for y in ys for x in xs]


def checked_array(value, shape, name):
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{name}尺寸必须为 {shape}，实际为 {arr.shape}。")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}包含 NaN 或无穷大。")
    if arr.min() < 0 or arr.max() > 1:
        raise ValueError(f"{name}数值必须在 0–1 之间。")
    return arr


def plan_region(frame_at, frame_count, width, height, box_width, box_height,
                margin=0, start=0, end=-1, sample_step=1, preferred="bottom",
                max_overlap=0.02, mask_at=None):
    """Find one fixed rectangle for [start, end); -1 means all remaining frames.

    frame_at / mask_at return one frame on demand, avoiding a full CPU copy.
    Mask scoring uses original resolution; image scoring uses a sparse grid.
    max_overlap is a soft-mask mean, NOT a probability of being safe.
    """
    end = frame_count if end == -1 else end
    if not 0 <= start < end <= frame_count:
        raise ValueError("帧范围无效：必须满足 0 ≤ 起始帧 < 结束帧 ≤ 总帧数。")
    if sample_step < 1:
        raise ValueError("采样间隔必须至少为 1。")
    if preferred not in ("bottom", "top", "center"):
        raise ValueError("未知的位置偏好。")
    if not math.isfinite(max_overlap) or not 0 <= max_overlap <= 1:
        raise ValueError("允许遮挡比例必须在 0–1 之间。")
    boxes = candidate_boxes(width, height, box_width, box_height, margin)
    indices = sorted(set(range(start, end, sample_step)) | {end-1})
    # Always evaluate original-resolution masks; thin protected features survive.
    worst = np.zeros(len(boxes))
    mask_total = np.zeros(len(boxes))
    visual_total = np.zeros(len(boxes))
    stride = max(1, math.ceil(max(width, height)/256))
    previous = None
    for index in indices:
        frame = checked_array(frame_at(index), (height, width, 3), "图像")
        rgb = frame[::stride, ::stride]
        gray = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        edge = np.zeros_like(gray)
        edge[:, 1:] += np.abs(gray[:, 1:] - gray[:, :-1])
        edge[1:, :] += np.abs(gray[1:, :] - gray[:-1, :])
        motion = np.zeros_like(gray) if previous is None else np.abs(gray-previous)
        previous = gray
        small_boxes = []
        for x, y, w, h in boxes:
            x0, y0 = x//stride, y//stride
            x1 = min(gray.shape[1], math.ceil((x+w)/stride))
            y1 = min(gray.shape[0], math.ceil((y+h)/stride))
            small_boxes.append((x0, y0, x1-x0, y1-y0))
        visual_total += rectangle_means(0.65*edge + 0.35*motion, small_boxes)
        if mask_at is not None:
            mask = checked_array(mask_at(index), (height, width), "保护遮罩")
            overlap = rectangle_means(mask, boxes).clip(0, 1)
            worst = np.maximum(worst, overlap)
            mask_total += overlap

    target_y = {"top": margin, "bottom": height-margin-box_height,
                "center": (height-box_height)/2}[preferred]
    # Position preference only breaks near ties; mask feasibility comes first.
    preference = np.array([
        abs(y-target_y)/height + 0.2*abs(x+w/2-width/2)/width
        for x, y, w, h in boxes
    ])
    visual = visual_total/len(indices) + 0.025*preference
    feasible = np.flatnonzero(worst <= max_overlap+1e-9)
    if len(feasible):
        selected = int(min(feasible, key=lambda i: (visual[i], worst[i], int(i))))
    else:
        selected = min(range(len(boxes)),
                       key=lambda i: (worst[i], mask_total[i], visual[i], i))

    warnings = []
    if mask_at is None:
        warnings.append("未提供主体遮罩：仅根据纹理和运动推荐，无法判断是否挡脸或挡产品。")
    if sample_step > 1:
        warnings.append("使用抽帧分析：未采样帧上的短暂遮挡可能被漏掉。")
    if mask_at is not None and not len(feasible):
        warnings.append("没有候选区域满足遮挡阈值；当前输出为遮挡最少的候选框，请缩小框或调整画面。")
    x, y, w, h = boxes[selected]
    return {
        "schema_version": "0.1.0", "box": {"x": x, "y": y, "width": w, "height": h},
        "frame_range": {"start_inclusive": start, "end_exclusive": end},
        "sampled_frames": indices, "candidate_count": len(boxes),
        "mask_provided": mask_at is not None,
        "sampled_mask_constraint_met": bool(mask_at is not None and len(feasible)),
        "all_frames_mask_constraint_met": bool(mask_at is not None and len(feasible)
                                                and len(indices) == end-start),
        "max_mask_overlap": float(worst[selected]) if mask_at is not None else None,
        "mean_mask_overlap": float(mask_total[selected]/len(indices)) if mask_at is not None else None,
        "overlap_threshold": max_overlap,
        "visual_cost": float(visual[selected]), "warnings": warnings,
    }
