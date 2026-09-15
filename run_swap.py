"""
Ground & Waist Aligned Body Swap (5-Layer PSD Architecture)
图层规范（从底至顶）：
1. 原图 (人像A完整原图) [隐藏图层 (visible=False)]
2. Clean Background (已擦除原躯干与腿的干净底图) [显示]
3. Photo B Body 不rmbg (带原背景的图B变形对齐层) [隐藏图层 (visible=False)]
4. Photo B Body (抠图+去头+对齐贴地贴腰身体) [显示]
5. Photo A Head (原人像A头部与发丝) [显示]
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pytoshop
from pytoshop.user import nested_layers
import rembg

def detect_pose(image_path, model_path, person_idx=None):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=5,
        output_segmentation_masks=False
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    # 1. 全图检测
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res_full = detector.detect(mp_img)
    
    raw_list = []
    if res_full.pose_landmarks:
        for lm in res_full.pose_landmarks:
            raw_list.append((lm, 0, w))
            
    # 2. 补充左半区(0~60%)与右半区(40%~100%)切片检测，彻底解决多人亲密贴合导致全图漏检/重叠误检的问题
    left_crop = img[:, :int(w * 0.6)]
    right_crop = img[:, int(w * 0.4):]
    off_r = int(w * 0.4)
    
    mp_left = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(left_crop, cv2.COLOR_BGR2RGB))
    mp_right = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(right_crop, cv2.COLOR_BGR2RGB))
    res_l = detector.detect(mp_left)
    res_r = detector.detect(mp_right)
    
    if res_l.pose_landmarks:
        raw_list.append((res_l.pose_landmarks[0], 0, left_crop.shape[1]))
    if res_r.pose_landmarks:
        raw_list.append((res_r.pose_landmarks[0], off_r, right_crop.shape[1]))

    if not raw_list:
        raise ValueError(f"未能检测到人体姿态: {image_path}")

    class NormLM:
        def __init__(self, x, y, z, vis, pres):
            self.x, self.y, self.z, self.visibility, self.presence = x, y, z, vis, pres

    # 3. 去重与坐标全局归一化
    candidates = []
    for lm, off_x, crop_w in raw_list:
        sh_x = (lm[11].x + lm[12].x) / 2.0 * crop_w + off_x
        is_dup = False
        for c in candidates:
            if abs(c['center_x'] - sh_x) < 0.15 * w:
                is_dup = True
                break
        if not is_dup:
            adj_lm = []
            for pt in lm:
                adj_lm.append(NormLM((pt.x * crop_w + off_x) / float(w), pt.y, pt.z, pt.visibility, pt.presence))
            candidates.append({'landmarks': adj_lm, 'center_x': sh_x})

    # 从左向右排序 (1:最左, 2:往右...)
    candidates.sort(key=lambda c: c['center_x'])
    total_persons = len(candidates)
    
    if total_persons > 1:
        print(f"   [多人检测] 在 {image_path} 中共识别到 {total_persons} 个人物 (从左到右: {[round(c['center_x'], 1) for c in candidates]})")
    
    # 用户指定人物序号 (1-indexed: 1, 2, ... 或负数 -1: 最后一个)
    if person_idx is not None:
        idx = person_idx - 1 if person_idx > 0 else person_idx
        if idx >= total_persons or idx < -total_persons:
            print(f"   [警告] 指定人物索引 {person_idx} 超出范围 (共 {total_persons} 人)，默认使用最靠近中心的人物")
            target_cand = min(candidates, key=lambda c: abs(c['center_x'] - w / 2.0))
        else:
            target_cand = candidates[idx]
            print(f"   [人物选定] 已选定第 {person_idx} 个人物 (画面X中心={target_cand['center_x']:.1f}px)")
    else:
        # 未指定时：如果有多人，默认优先选最居中的主要主体
        target_cand = min(candidates, key=lambda c: abs(c['center_x'] - w / 2.0))
        if total_persons > 1:
            print(f"   [默认主体] 自动选定画面中心主体 (画面X中心={target_cand['center_x']:.1f}px)")

    return target_cand['landmarks'], w, h

def get_person_geometry(landmarks, width, height):
    ls = np.array([landmarks[11].x * width, landmarks[11].y * height], dtype=np.float32)
    rs = np.array([landmarks[12].x * width, landmarks[12].y * height], dtype=np.float32)
    nose = np.array([landmarks[0].x * width, landmarks[0].y * height], dtype=np.float32)

    mid_shoulder = (ls + rs) / 2.0
    shoulder_width = float(np.linalg.norm(ls - rs))
    neck = mid_shoulder + 0.28 * (nose - mid_shoulder)

    lh = landmarks[23]
    rh = landmarks[24]
    has_visible_hip = (lh.visibility > 0.4 and rh.visibility > 0.4 and
                       0.0 <= lh.y <= 1.02 and 0.0 <= rh.y <= 1.02)
    
    if has_visible_hip:
        lh_pt = np.array([lh.x * width, lh.y * height], dtype=np.float32)
        rh_pt = np.array([rh.x * width, rh.y * height], dtype=np.float32)
        mid_waist = (lh_pt + rh_pt) / 2.0
        waist_width = float(np.linalg.norm(lh_pt - rh_pt))
        torso_height = float(mid_waist[1] - neck[1])
    else:
        lh_pt, rh_pt, mid_waist, waist_width = None, None, None, None
        torso_height = float(height - neck[1])

    # 腿部与脚踝姿态分析：判断是否为真正垂直站立 (Standing) 还是 坐姿/屈腿 (Sitting/Kneeling)
    foot_pts_y = []
    for idx in [27, 28, 29, 30, 31, 32]:
        lm = landmarks[idx]
        if lm.visibility > 0.35 and lm.presence > 0.35 and 0.0 <= lm.y <= 1.05:
            foot_pts_y.append(lm.y * height)
    
    # 严格的站立姿态判定：
    # 真正站立时，髋关节 -> 膝盖 -> 踝关节 必须沿重力方向呈现显著递增的垂直落差（大腿小腿垂直伸展）
    is_standing = False
    if len(foot_pts_y) >= 2 and has_visible_hip:
        hip_y = (landmarks[23].y + landmarks[24].y) / 2.0
        knee_y = (landmarks[25].y + landmarks[26].y) / 2.0
        ankle_y = (landmarks[27].y + landmarks[28].y) / 2.0
        thigh_drop = knee_y - hip_y
        calf_drop = ankle_y - knee_y
        # 站姿时大腿和小腿纵向落差均显著大于 0.08（图像归一化高度）
        if thigh_drop > 0.08 and calf_drop > 0.06:
            is_standing = True

    has_visible_legs = len(foot_pts_y) >= 2
    is_sitting = has_visible_legs and (not is_standing)
    is_full_body = is_standing
    ground_y = float(np.max(foot_pts_y)) if is_full_body else None

    # 头部解剖尺度测量 (Head Physical Anatomical Scale)
    # 鼻根到颈部基底中点的距离 (极度稳定的刚体尺度，不受身体 90 度旋转投影压缩影响)
    head_size = float(np.linalg.norm(nose - neck))
    eye_dist = float(np.linalg.norm(np.array([landmarks[2].x - landmarks[5].x, landmarks[2].y - landmarks[5].y])) * width)
    
    # 身体侧向偏角判定：头肩比 (Shoulder / Eye_dist)
    # 正常正面/半身时 shoulder_width / eye_dist 约 4.0 ~ 6.0
    # 当纯侧面 90 度时，2D 投影肩宽极度萎缩，比例会低于 3.3
    is_side_angle = False
    if eye_dist > 5.0 and (shoulder_width / eye_dist) < 3.3:
        is_side_angle = True

    return {
        'neck': neck,
        'left_shoulder': ls,
        'right_shoulder': rs,
        'mid_shoulder': mid_shoulder,
        'shoulder_width': shoulder_width,
        'head_size': head_size,
        'eye_dist': eye_dist,
        'is_side_angle': is_side_angle,
        'has_visible_hip': has_visible_hip,
        'mid_waist': mid_waist,
        'waist_width': waist_width,
        'torso_height': torso_height,
        'nose': nose,
        'is_full_body': is_full_body,
        'is_sitting': is_sitting,
        'has_visible_legs': has_visible_legs,
        'ground_y': ground_y,
        'img_height': height,
        'img_width': width,
        'landmarks': landmarks
    }

def inpaint_remove_original_body(img_a_bgr, rgba_a, geom_a, norm_up, aligned_body_b=None, geom_b=None):
    h, w = img_a_bgr.shape[:2]
    person_alpha = rgba_a[:, :, 3]
    neck_point = geom_a['neck']
    landmarks = geom_a['landmarks']

    # 1. 构建面部与下颌保护区（绝不可被抹除模糊）：
    # 提取头面部特征点 (0:鼻尖, 1-6:眼部, 7-8:耳部, 9-10:口唇)
    head_points = []
    for i in range(11):
        head_points.append([int(landmarks[i].x * w), int(landmarks[i].y * h)])
    head_points = np.array(head_points, dtype=np.int32)

    hull = cv2.convexHull(head_points)
    face_protected = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_protected, hull, 255)
    kernel_face = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    face_protected = cv2.dilate(face_protected, kernel_face, iterations=2)

    # 2. 智能下半身保留判定：
    # 当且仅当：
    # 1) 目标图A包含全身站姿 (geom_a['is_full_body'] 为 True)
    # 2) 供体B是纯半身照（没有明显下肢/不是坐姿抱膝等有腿姿态，即 not geom_b['has_visible_legs']）
    # 3) 供体B对齐后的下沿明显高于 A 的脚底（留出至少 15% 身高空间）
    # 此时属于典型的“换上衣/保留A原生裤子与腿”穿搭场景，自动锁定下摆，完整保留A原有的下半身！
    # 反之：如果B本身是坐姿（比如抱膝坐在沙发/椅子上），B带有自己的腿和裤子，必须抹除A原有的站姿腿，绝对不能出现四条腿拼接！
    retain_lower_body = False
    max_y_b = h
    if aligned_body_b is not None and geom_b is not None:
        if geom_a.get('is_full_body', False) and (not geom_b.get('is_full_body', False)) and (not geom_b.get('is_sitting', False)) and (not geom_b.get('has_visible_legs', False)):
            alpha_b = aligned_body_b[:, :, 3]
            y_indices_b = np.where(alpha_b > 20)[0]
            if len(y_indices_b) > 0:
                max_y_b = int(y_indices_b.max())
                if geom_a['ground_y'] is not None and (geom_a['ground_y'] - max_y_b) > 0.15 * h:
                    retain_lower_body = True
                    print(f"   [智能半身穿搭融合] 识别为标准半身上衣+全身下装组合，自动锁定下摆(Y={max_y_b})，完整保留A原有的下半身与腿部！")

    # 3. 根据脖子法向量切分出原图A的身体区域（排除头部）
    y_coords, x_coords = np.ogrid[:h, :w]
    dot = (x_coords - neck_point[0]) * norm_up[0] + (y_coords - neck_point[1]) * norm_up[1]
    
    if retain_lower_body:
        # 只抹除到 B 身体下沿以上 6 像素处，让 B 的下摆自然搭在 A 的腰部/裤头上
        cutoff_y = max(0, max_y_b - 6)
        is_body = (dot < 0.0) & (person_alpha > 20) & (y_coords <= cutoff_y)
    else:
        is_body = (dot < 0.0) & (person_alpha > 20)
        
    body_mask = is_body.astype(np.uint8) * 255

    # 4. 膨胀身体抹除遮罩以彻底消除衣物边缘，但严格扣除面部保护区！
    kernel_body = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    body_mask_dilated = cv2.dilate(body_mask, kernel_body, iterations=1)
    
    if retain_lower_body:
        # 绝不让膨胀遮罩蔓延到保留的裤子区域
        body_mask_dilated[cutoff_y:, :] = 0
        
    body_mask_final = cv2.subtract(body_mask_dilated, face_protected)

    if retain_lower_body:
        print("   [Inpaint] 仅抹除原图A的上半身衣物，100%保留原生下装与裤腿，保护面容与下颌...")
    else:
        print("   [Inpaint] 正在抹除原图A中原有的身体、双腿和鞋子，保护完整面容与下颌...")
        
    clean_bg = cv2.inpaint(img_a_bgr, body_mask_final, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return clean_bg

def extract_head_a(rgba_a, neck_point, norm_up):
    h, w = rgba_a.shape[:2]
    y_coords, x_coords = np.ogrid[:h, :w]
    dot = (x_coords - neck_point[0]) * norm_up[0] + (y_coords - neck_point[1]) * norm_up[1]

    feather = 10.0
    head_mask = np.clip((dot + feather) / (2 * feather), 0.0, 1.0)

    head_rgba = rgba_a.copy()
    head_rgba[:, :, 3] = (head_rgba[:, :, 3].astype(np.float32) * head_mask).astype(np.uint8)
    return head_rgba

def extract_body_b(rgba_b, neck_point, norm_up, geom_b=None):
    h, w = rgba_b.shape[:2]
    y_coords, x_coords = np.ogrid[:h, :w]
    dot = (x_coords - neck_point[0]) * norm_up[0] + (y_coords - neck_point[1]) * norm_up[1]

    feather = 10.0
    body_mask = np.clip((-dot + feather) / (2 * feather), 0.0, 1.0)

    # 当供体照片中有多人，或身体周围有他人干扰时，使用选定人物的关键点凸包限制横向范围
    if geom_b is not None and 'landmarks' in geom_b:
        lms = geom_b['landmarks']
        body_pts = []
        # 收集选定人物的身体主要关节点 (肩膀11,12、肘13,14、手腕15,16、髋23,24、膝25,26、踝27,28)
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if idx < len(lms):
                lm = lms[idx]
                if lm.visibility > 0.2 and 0 <= lm.x <= 1.0 and 0 <= lm.y <= 1.0:
                    body_pts.append([int(lm.x * w), int(lm.y * h)])
        if len(body_pts) >= 4:
            pts_arr = np.array(body_pts, dtype=np.int32)
            hull = cv2.convexHull(pts_arr)
            hull_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, hull, 255)
            # 适度膨胀以覆盖衣物轮廓与外沿
            sh_w = int(geom_b.get('shoulder_width', 50))
            k_size = max(15, int(sh_w * 0.35))
            if k_size % 2 == 0:
                k_size += 1
            kernel_hull = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            hull_mask = cv2.dilate(hull_mask, kernel_hull, iterations=2)
            body_mask = body_mask * (hull_mask.astype(np.float32) / 255.0)

    body_rgba = rgba_b.copy()
    body_rgba[:, :, 3] = (body_rgba[:, :, 3].astype(np.float32) * body_mask).astype(np.uint8)
    return body_rgba

def compute_alignment_matrix(geom_a, geom_b):
    neck_a = geom_a['neck']
    neck_b = geom_b['neck']

    # 1. 核心防御：检测到任一方为纯侧面 90 度 (is_side_angle) 时，2D 投影肩宽失效，
    # 自动切换为「头部解剖物理尺度 (Head-Anchored Scale)」！
    if geom_a['is_side_angle'] or geom_b['is_side_angle']:
        scale_head = geom_a['head_size'] / (geom_b['head_size'] + 1e-5)
        tx = neck_a[0] - neck_b[0] * scale_head
        ty = neck_a[1] - neck_b[1] * scale_head
        M = np.array([[scale_head, 0, tx], [0, scale_head, ty]], dtype=np.float32)
        print(f"   [对齐模式: 头部解剖尺度锚定 (侧身防崩保护)] 纯侧面姿态识别成功，基于头部真实物理比例等比缩放={scale_head:.2f}")
        return M

    # 2. 常规正面/微侧姿态：使用肩宽作为视觉骨架基准
    scale_shoulder = geom_a['shoulder_width'] / (geom_b['shoulder_width'] + 1e-5)

    if geom_a['is_full_body'] and geom_b['is_full_body']:
        height_a = geom_a['ground_y'] - neck_a[1]
        height_b = geom_b['ground_y'] - neck_b[1]
        
        scale_y = height_a / (height_b + 1e-5)
        # 全身对齐时：优先保证纵向接地与身高的自然比例，但横向不低于肩宽的 85%
        scale_x = scale_shoulder
        aspect_ratio = scale_x / scale_y
        if aspect_ratio < 0.85 or aspect_ratio > 1.25:
            scale_x = np.clip(scale_x, scale_y * 0.85, scale_y * 1.25)

        tx = neck_a[0] - neck_b[0] * scale_x
        ty = neck_a[1] - neck_b[1] * scale_y
        M = np.array([[scale_x, 0, tx], [0, scale_y, ty]], dtype=np.float32)
        print(f"   [对齐模式: 全身接地锁定] 缩放(X={scale_x:.2f}, Y={scale_y:.2f}), 脚底对准高度 Y={geom_a['ground_y']:.1f}")

    elif geom_a['has_visible_hip'] and geom_b['has_visible_hip']:
        # 半身/中景模式：必须以【肩宽】为主要物理尺度基准（等比缩放），
        # 绝不能用躯干垂直高度粗暴覆盖横向肩宽导致身体缩成细条！
        scale = scale_shoulder
        tx = neck_a[0] - neck_b[0] * scale
        ty = neck_a[1] - neck_b[1] * scale
        M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
        print(f"   [对齐模式: 半身肩宽锚定] 等比缩放={scale:.2f} (肩宽={geom_a['shoulder_width']:.1f}px), 颈部精准锁位")

    else:
        pts_src = np.array([geom_b['neck'], geom_b['left_shoulder'], geom_b['right_shoulder']], dtype=np.float32)
        pts_dst = np.array([geom_a['neck'], geom_a['left_shoulder'], geom_a['right_shoulder']], dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(pts_src, pts_dst)
        if M is None:
            scale = geom_a['shoulder_width'] / (geom_b['shoulder_width'] + 1e-5)
            M = np.array([[scale, 0, neck_a[0] - neck_b[0] * scale],
                          [0, scale, neck_a[1] - neck_b[1] * scale]], dtype=np.float32)
        print(f"   [对齐模式: 胸像近景对齐] 以颈部与肩宽为基准自适应对齐")

    return M

def fit_torso_width(aligned_img, geom_a, geom_b, M):
    """
    肩宽与胯宽双轴梯形渐变拟合（Trunk Dual-Axis Width Fitting）:
    - 以脊柱中轴为对称中心
    - 颈肩截面: 保持肩宽对齐 (scale = 1.0)
    - 腰胯截面: 平滑过渡至胯宽比例 (scale = hip_ratio)
    - 仅对躯干区域平滑变形，避免四肢拉扯失真
    """
    if not (geom_a['has_visible_hip'] and geom_b['has_visible_hip']):
        return aligned_img

    # 纯侧身姿态不适用正面梯形腰胯渐变
    if geom_a['is_side_angle'] or geom_b['is_side_angle']:
        return aligned_img

    h, w = aligned_img.shape[:2]
    neck_a = geom_a['neck']
    waist_a = geom_a['mid_waist']

    # 计算肩宽缩放与胯宽缩放的相对比例差
    # 目前全局已经按 shoulder_width 缩放过了
    # 如果图A胯部较宽/较窄，计算还需要在胯部位置补偿的横向系数
    hip_scale_needed = geom_a['waist_width'] / (geom_b['waist_width'] * M[0, 0] + 1e-5)
    # 限制在合理范围 0.85 ~ 1.25，防止极端姿态过度拉伸
    hip_scale_needed = float(np.clip(hip_scale_needed, 0.85, 1.25))

    if abs(hip_scale_needed - 1.0) < 0.03:
        return aligned_img # 差异极小无需变形

    print(f"   [肩胯双轴优化] 肩宽已锁定，腰胯额外补偿适配比: {hip_scale_needed:.2f}")

    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    y_neck = float(neck_a[1])
    y_waist = float(waist_a[1])
    cx = float(waist_a[0])

    # 纵向从颈部到腰胯的过渡权重 t (0.0 在脖子，1.0 在腰胯)
    t = np.clip((y_coords - y_neck) / (y_waist - y_neck + 1e-5), 0.0, 1.0)
    # 平滑 S 曲线 (Smoothstep)
    t = t * t * (3.0 - 2.0 * t)

    scale_x = (1.0 - t) * 1.0 + t * hip_scale_needed

    src_x = cx + (x_coords - cx) / scale_x
    src_y = y_coords

    warped = cv2.remap(
        aligned_img, src_x, src_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0) if aligned_img.shape[2] == 4 else (0, 0, 0)
    )
    return warped

def save_5layer_psd_and_png(img_a_rgb, clean_bg_bgr, raw_b_rgb, aligned_body_b, head_a, M, out_psd_path, out_png_path, geom_a, geom_b):
    h_a, w_a = img_a_rgb.shape[:2]
    clean_bg_rgb = cv2.cvtColor(clean_bg_bgr, cv2.COLOR_BGR2RGB)

    # 计算原图B（不rmbg）在此变换下的图像
    aligned_raw_b = cv2.warpAffine(
        raw_b_rgb, M, (w_a, h_a),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # 躯干肩宽与胯宽双轴微调拟合
    aligned_body_b = fit_torso_width(aligned_body_b, geom_a, geom_b, M)
    aligned_raw_b = fit_torso_width(aligned_raw_b, geom_a, geom_b, M)

    # 1. 导出合并 PNG 预览
    # 采用 Clean Background 中的 100% 原始未抠图头部与发丝，避免抠图毛刺
    comp = clean_bg_rgb.astype(np.float32)
    alpha_body = aligned_body_b[:, :, 3:4].astype(np.float32) / 255.0
    comp = aligned_body_b[:, :, :3] * alpha_body + comp * (1.0 - alpha_body)
    
    comp_arr = np.clip(comp, 0, 255).astype(np.uint8)
    Image.fromarray(comp_arr).save(out_png_path)
    print(f"[PNG 预览图]: {out_png_path}")

    # 2. 导出标准 5 图层 PSD
    # 注意：Photoshop 图层堆叠顺序是从下到上。
    # pytoshop 的 layer 列表：第一个为底层，最后一个为顶层。
    
    # 规范化图层命名与层级：
    # Layer 1 (基底备份): [01] Original (Photo A) - 隐藏
    l1_orig_a = nested_layers.Image(
        name='[01] Original (Photo A)',
        visible=False,
        top=0, left=0, bottom=h_a, right=w_a,
        channels={0: img_a_rgb[:, :, 0], 1: img_a_rgb[:, :, 1], 2: img_a_rgb[:, :, 2]}
    )

    # Layer 2 (干净底图): [02] Clean Background - 显示
    l2_clean_bg = nested_layers.Image(
        name='[02] Clean Background',
        visible=True,
        top=0, left=0, bottom=h_a, right=w_a,
        channels={0: clean_bg_rgb[:, :, 0], 1: clean_bg_rgb[:, :, 1], 2: clean_bg_rgb[:, :, 2]}
    )

    # Layer 3 (未抠图身体参考): [03] Photo B Body (Unmasked Ref) - 隐藏
    l3_raw_b = nested_layers.Image(
        name='[03] Photo B Body (Unmasked Ref)',
        visible=False,
        top=0, left=0, bottom=h_a, right=w_a,
        channels={0: aligned_raw_b[:, :, 0], 1: aligned_raw_b[:, :, 1], 2: aligned_raw_b[:, :, 2]}
    )

    # Layer 4 (对齐身体层): [04] Photo B Body (Aligned) - 显示
    l4_body_b = nested_layers.Image(
        name='[04] Photo B Body (Aligned)',
        visible=True,
        top=0, left=0, bottom=h_a, right=w_a,
        channels={-1: aligned_body_b[:, :, 3], 0: aligned_body_b[:, :, 0], 1: aligned_body_b[:, :, 1], 2: aligned_body_b[:, :, 2]}
    )

    # Layer 5 (顶层人像A头备用): [05] Photo A Head (Foreground) - 隐藏（按方案A默认关闭，保留备用）
    l5_head_a = nested_layers.Image(
        name='[05] Photo A Head (Foreground)',
        visible=False,
        top=0, left=0, bottom=h_a, right=w_a,
        channels={-1: head_a[:, :, 3], 0: head_a[:, :, 0], 1: head_a[:, :, 1], 2: head_a[:, :, 2]}
    )

    # 传入图层顺序：[05, 04, 03, 02, 01]
    # 在 Photoshop 图层面板中自顶向下排布：[05] 头在最顶层，[01] 原图在最底层
    layers_order = [l5_head_a, l4_body_b, l3_raw_b, l2_clean_bg, l1_orig_a]
    psd = nested_layers.nested_layers_to_psd(layers_order, color_mode=pytoshop.enums.ColorMode.rgb)
    with open(out_psd_path, 'wb') as f:
        psd.write(f)
    print(f"[5图层 PSD]: {out_psd_path}")

def run_swap(target_a, donor_b, output_prefix, target_person=None, donor_person=None):
    model_path = r"C:\Users\jooies\Downloads\person\models\pose_landmarker.task"
    out_psd = f"{output_prefix}.psd"
    out_png = f"{output_prefix}.png"

    print(">> 启动 5 图层 PSD 流水线...")
    rembg_session = rembg.new_session('u2netp')

    # 读取原图 A 与 B
    img_a_pil = Image.open(target_a).convert('RGB')
    img_a_rgb = np.array(img_a_pil)
    img_b_pil = Image.open(donor_b).convert('RGB')
    img_b_rgb = np.array(img_b_pil)

    # 1. 骨骼点检测（支持多人照片指定选人）
    print("[1/5] 检测双人骨骼点（颈肩、腰胯、脚底接地）...")
    lm_a, w_a, h_a = detect_pose(target_a, model_path, person_idx=target_person)
    lm_b, w_b, h_b = detect_pose(donor_b, model_path, person_idx=donor_person)

    geom_a = get_person_geometry(lm_a, w_a, h_a)
    geom_b = get_person_geometry(lm_b, w_b, h_b)

    # 2. 抠图与部位拆分
    print("[2/5] 智能抠图并分离头部与身体...")
    rgba_a = np.array(rembg.remove(Image.fromarray(img_a_rgb), session=rembg_session))
    rgba_b = np.array(rembg.remove(Image.fromarray(img_b_rgb), session=rembg_session))

    neck_up_a = geom_a['nose'] - geom_a['mid_shoulder']
    norm_up_a = neck_up_a / (np.linalg.norm(neck_up_a) + 1e-5)

    neck_up_b = geom_b['nose'] - geom_b['mid_shoulder']
    norm_up_b = neck_up_b / (np.linalg.norm(neck_up_b) + 1e-5)

    head_a = extract_head_a(rgba_a, geom_a['neck'], norm_up_a)
    body_b = extract_body_b(rgba_b, geom_b['neck'], norm_up_b, geom_b=geom_b)

    # 3. 计算姿态几何对齐变换（自适应腰部/地平线锁定）
    print("[3/5] 计算姿态几何对齐变换（自适应腰部/地平线锁定）...")
    M = compute_alignment_matrix(geom_a, geom_b)
    aligned_body_b = cv2.warpAffine(
        body_b, M, (w_a, h_a),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # 4. 擦除原图A身体（智能保留下半身）
    print("[4/5] 底图人像遮罩擦除 (生成 Clean Background)...")
    img_a_bgr = cv2.imread(target_a)
    clean_bg = inpaint_remove_original_body(img_a_bgr, rgba_a, geom_a, norm_up_a, aligned_body_b=aligned_body_b, geom_b=geom_b)

    # 5. 打包 5 图层 PSD 与 预览 PNG
    print("[5/5] 保存 5 图层 PSD 与 PNG 预览图...")
    save_5layer_psd_and_png(img_a_rgb, clean_bg, img_b_rgb, aligned_body_b, head_a, M, out_psd, out_png, geom_a, geom_b)
    print(">> 任务全部成功！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="生成标准 5 图层 PSD 的全自动头身对齐工具")
    parser.add_argument("--target", required=True, help="目标照片A路径")
    parser.add_argument("--donor", required=True, help="身体照片B路径")
    parser.add_argument("--target_person", type=int, default=None, help="目标图A人物序号(从左到右1, 2...，默认居中主体)")
    parser.add_argument("--donor_person", type=int, default=None, help="身体图B人物序号(从左到右1, 2...，默认居中主体)")
    parser.add_argument("--output", default="final_5layers_swap", help="输出文件前缀")
    args = parser.parse_args()

    run_swap(args.target, args.donor, args.output, target_person=args.target_person, donor_person=args.donor_person)
