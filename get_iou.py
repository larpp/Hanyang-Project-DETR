# IoU Matrix
def get_iou(pred_box, gt_box):
	# input box shape : [[x, y, w, h], [x, y, w, h], [...], ...]
    iou_matrix = []
    for box1 in pred_box:
        iou_li = []
        for box2 in gt_box:
            # 좌표 값 추출
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
               
            # 교집합의 좌상단과 우하단 좌표 계산
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
                
            # 교집합 영역의 너비와 높이 계산
            intersection_width = max(0, x_right - x_left)
            intersection_height = max(0, y_bottom - y_top)
                
            # 교집합 영역의 면적 계산
            intersection_area = intersection_width * intersection_height

            if intersection_area <= 0: iou = 0
            else:
                # 각 박스의 면적 계산
                area_box1 = w1 * h1
                area_box2 = w2 * h2
                    
                # 합집합 영역의 면적 계산
                union_area = area_box1 + area_box2 - intersection_area
                    
                # IoU 계산
                iou = intersection_area / union_area
            
            iou_li.append(iou)
            
        
        iou_matrix.append(iou_li)
    return iou_matrix