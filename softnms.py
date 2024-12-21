import torch
import math

class BoxIOU:
    def __init__(self, GIoU=False, DIoU=False, CIoU=False, SIoU=False, EIou=False, eps=1e-7):
        self.GIoU = GIoU
        self.DIoU = DIoU
        self.CIoU = CIoU
        self.SIoU = SIoU
        self.EIou = EIou
        self.eps = eps

    def calculate_iou(self, box1, box2):
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(self.eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(self.eps)

        # Intersection area
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
                (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

        # Union Area
        union = w1 * h1 + w2 * h2 - inter + self.eps

        # IoU
        iou = inter / union
        if self.CIoU or self.DIoU or self.GIoU or self.EIou:
            cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
            ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
            if self.CIoU or self.DIoU or self.EIou:  # Distance or Complete IoU
                c2 = cw ** 2 + ch ** 2 + self.eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
                if self.CIoU:
                    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + self.eps))
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
                elif self.EIou:
                    rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                    rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                    cw2 = cw ** 2 + self.eps
                    ch2 = ch ** 2 + self.eps
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)
                return iou - rho2 / c2  # DIoU
            c_area = cw * ch + self.eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        elif self.SIoU:
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + self.eps
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + self.eps
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            return iou - 0.5 * (distance_cost + shape_cost)
        return iou

class SoftNMS:
    def __init__(self, iou_thresh=0.5, sigma=0.5, score_threshold=0.25):
        self.iou_thresh = iou_thresh
        self.sigma = sigma
        self.score_threshold = score_threshold

    def apply(self, bboxes, scores):
        order = torch.arange(0, scores.size(0)).to(bboxes.device)
        keep = []

        while order.numel() > 1:
            if order.numel() == 1:
                keep.append(order[0])
                break
            else:
                i = order[0]
                keep.append(i)

            iou = BoxIOU().calculate_iou(bboxes[i], bboxes[order[1:]]).squeeze()

            idx = (iou > self.iou_thresh).nonzero().squeeze()
            if idx.numel() > 0:
                iou = iou[idx]
                newScores = torch.exp(-torch.pow(iou, 2) / self.sigma)
                scores[order[idx + 1]] *= newScores

            newOrder = (scores[order[1:]] > self.score_threshold).nonzero().squeeze()
            if newOrder.numel() == 0:
                break
            else:
                maxScoreIndex = torch.argmax(scores[order[newOrder + 1]])
                if maxScoreIndex != 0:
                    newOrder[[0, maxScoreIndex], ] = newOrder[[maxScoreIndex, 0], ]
                order = order[newOrder + 1]

        return torch.LongTensor(keep)
