import json
import math

from .version import __version__


class SubtitleSafeZone:
    CATEGORY = "Video/Subtitle Safe Zone"
    FUNCTION = "plan"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("preview", "region_mask", "x", "y", "width", "height",
                    "mask_constraint_met", "report_json")
    DESCRIPTION = "为指定视频帧范围推荐固定字幕区域。保护遮罩白色表示避让；不自动识别人脸、不绘制字幕。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "box_width_ratio": ("FLOAT", {"default": 0.6, "min": 0.05, "max": 1.0, "step": 0.01}),
                "box_height_ratio": ("FLOAT", {"default": 0.14, "min": 0.02, "max": 1.0, "step": 0.01}),
                "margin_ratio": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.4, "step": 0.01}),
                "preferred": (["bottom", "top", "center"],),
                "start_frame": ("INT", {"default": 0, "min": 0}),
                "end_frame": ("INT", {"default": -1, "min": -1,
                                      "tooltip": "不含结束帧；-1 表示直到视频末尾。"}),
                "sample_step": ("INT", {"default": 1, "min": 1, "max": 120,
                                        "tooltip": "1 为逐帧检查；增加会漏掉短暂遮挡。"}),
                "max_overlap": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.005}),
            },
            "optional": {"protect_mask": ("MASK", {"tooltip": "白色=避让；单张广播或与视频逐帧对应，尺寸须匹配。"})},
        }

    def plan(self, images, box_width_ratio=0.6, box_height_ratio=0.14,
             margin_ratio=0.04, preferred="bottom", start_frame=0, end_frame=-1,
             sample_step=1, max_overlap=0.02, protect_mask=None):
        try:
            import torch
            from .planner import plan_region
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "字幕避让节点无法加载宿主 NumPy/PyTorch。请先在测试环境核对兼容性；"
                "本节点不会安装、升级或修复依赖。原始原因：" + str(exc)
            ) from exc

        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("images 必须为 ComfyUI IMAGE：[帧数, 高, 宽, 3]。")
        count, height, width, _ = images.shape
        for name, value, low, high in (("宽度比例", box_width_ratio, 0.05, 1),
                                       ("高度比例", box_height_ratio, 0.02, 1),
                                       ("边距比例", margin_ratio, 0, 0.4)):
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name}必须在 {low}–{high} 之间。")
        box_w = max(1, round(width*box_width_ratio))
        box_h = max(1, round(height*box_height_ratio))
        margin = round(min(width, height)*margin_ratio)
        mask_at = None
        if protect_mask is not None:
            if protect_mask.ndim == 2:
                protect_mask = protect_mask.unsqueeze(0)
            if (protect_mask.ndim != 3 or protect_mask.shape[0] not in (1, count)
                    or tuple(protect_mask.shape[1:]) != (height, width)):
                raise ValueError("保护遮罩必须与画面同尺寸，帧数必须为 1 或与视频相同。")
            def mask_at(index):
                return protect_mask[0 if protect_mask.shape[0] == 1 else index].detach().float().cpu().numpy()

        report = plan_region(
            lambda i: images[i].detach().float().cpu().numpy(), count, width, height,
            box_w, box_h, margin, start_frame, end_frame, sample_step, preferred,
            max_overlap, mask_at,
        )
        x, y = report["box"]["x"], report["box"]["y"]
        interval = report["frame_range"]
        first, last = interval["start_inclusive"], interval["end_exclusive"]-1
        preview_indices = sorted({first, (first+last)//2, last})
        previews = []
        met = report["all_frames_mask_constraint_met"]
        color = torch.tensor([0.15, 0.9, 0.45] if met else [1.0, 0.65, 0.1])
        border = max(1, min(width, height)//200)
        for index in preview_indices:
            canvas = images[index].detach().float().cpu().clone()
            canvas[y:y+box_h, x:x+box_w] = canvas[y:y+box_h, x:x+box_w]*0.8 + color*0.2
            canvas[y:y+border, x:x+box_w] = color
            canvas[y+box_h-border:y+box_h, x:x+box_w] = color
            canvas[y:y+box_h, x:x+border] = color
            canvas[y:y+box_h, x+box_w-border:x+box_w] = color
            previews.append(canvas)
        region = torch.zeros((1, height, width), dtype=torch.float32)
        region[:, y:y+box_h, x:x+box_w] = 1.0
        report["preview_frame_indices"] = preview_indices
        report["plugin_version"] = __version__
        return (torch.stack(previews), region, x, y, box_w, box_h, met,
                json.dumps(report, ensure_ascii=False, indent=2))


NODE_CLASS_MAPPINGS = {"SubtitleSafeZonePlanner": SubtitleSafeZone}
NODE_DISPLAY_NAME_MAPPINGS = {"SubtitleSafeZonePlanner": "视频字幕稳定避让 / Subtitle Safe Zone"}
