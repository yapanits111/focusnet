# thesis_evaluation.py
# Comprehensive evaluation metrics for FocusNet thesis
# Implements all required metrics: Precision, Recall, F1-Score, mAP, IoU
# Includes statistical analysis: Wilcoxon signed-rank test for model comparison
# 
# Purpose: Generate all evaluation results needed for thesis validation
# Models: FocusNet vs Baseline SSD comparison

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import precision_recall_curve, average_precision_score
import json

def decode_predictions(cls_logits, box_deltas, anchors, score_thresh=0.5, nms_thresh=0.45):
    """
    Decode model predictions into bounding boxes and scores
    
    Args:
        cls_logits: Classification logits [B, A, C]
        box_deltas: Box regression deltas [B, A, 4]
        anchors: Anchor boxes [A, 4]
        score_thresh: Confidence threshold
        nms_thresh: NMS IoU threshold
    
    Returns:
        list: Predictions for each image [(boxes, labels, scores), ...]
    """
    batch_size = cls_logits.size(0)
    cls_scores = F.softmax(cls_logits, dim=-1)
    
    results = []
    
    for b in range(batch_size):
        # Get scores and labels
        scores_b = cls_scores[b]  # [A, C]
        deltas_b = box_deltas[b]  # [A, 4]
        
        # Get max scores and predicted classes (exclude background)
        max_scores, pred_labels = scores_b[:, 1:].max(dim=-1)  # Exclude background class 0
        pred_labels = pred_labels + 1  # Adjust for background
        
        # Filter by score threshold
        valid_mask = max_scores > score_thresh
        
        if not valid_mask.any():
            results.append((torch.empty(0, 4), torch.empty(0), torch.empty(0)))
            continue
        
        valid_scores = max_scores[valid_mask]
        valid_labels = pred_labels[valid_mask]
        valid_deltas = deltas_b[valid_mask]
        valid_anchors = anchors[valid_mask]
        
        # Decode boxes
        pred_boxes = decode_boxes(valid_anchors, valid_deltas)
        
        # Apply NMS per class
        keep_indices = []
        for class_id in valid_labels.unique():
            class_mask = valid_labels == class_id
            if not class_mask.any():
                continue
            
            class_boxes = pred_boxes[class_mask]
            class_scores = valid_scores[class_mask]
            
            # Apply NMS
            keep = torchvision.ops.nms(class_boxes, class_scores, nms_thresh)
            class_indices = torch.where(class_mask)[0][keep]
            keep_indices.extend(class_indices.tolist())
        
        if keep_indices:
            keep_indices = torch.tensor(keep_indices, dtype=torch.long)
            final_boxes = pred_boxes[keep_indices]
            final_labels = valid_labels[keep_indices]  
            final_scores = valid_scores[keep_indices]
        else:
            final_boxes = torch.empty(0, 4)
            final_labels = torch.empty(0)
            final_scores = torch.empty(0)
        
        results.append((final_boxes, final_labels, final_scores))
    
    return results

def decode_boxes(anchors, deltas, center_variance=0.1, size_variance=0.2):
    """Decode box deltas to actual coordinates"""
    # Decode center and size
    cxcy = deltas[..., :2] * center_variance * anchors[..., 2:] + anchors[..., :2]
    wh = torch.exp(deltas[..., 2:] * size_variance) * anchors[..., 2:]
    
    # Convert to x1y1x2y2 format
    x1y1 = cxcy - wh / 2
    x2y2 = cxcy + wh / 2
    
    return torch.cat([x1y1, x2y2], dim=-1)

def calculate_iou(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes
    
    Args:
        boxes1: [N, 4] in x1y1x2y2 format
        boxes2: [M, 4] in x1y1x2y2 format
    
    Returns:
        torch.Tensor: IoU matrix [N, M]
    """
    # Calculate intersection
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]
    
    # Calculate union
    union = area1[:, None] + area2 - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-6)
    
    return iou

def evaluate_detection(model, data_loader, device, iou_thresh=0.5, conf_thresh=0.5, class_names=None):
    """
    Comprehensive detection evaluation for thesis
    
    Args:
        model: Detection model (FocusNet or Baseline SSD)
        data_loader: Validation/test data loader
        device: CUDA device
        iou_thresh: IoU threshold for positive detection
        conf_thresh: Confidence threshold
        class_names: List of class names
    
    Returns:
        dict: Complete evaluation metrics
    """
    model.eval()
    
    # Storage for all predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    per_class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'precisions': [], 'recalls': []})
    
    print(f"🔍 Evaluating model on {len(data_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if batch_idx % 50 == 0:
                print(f"   Batch {batch_idx}/{len(data_loader)}")
            
            images = images.to(device)
            
            # Get model predictions
            cls_logits, box_deltas, anchors = model(images)
            predictions = decode_predictions(cls_logits, box_deltas, anchors, 
                                           score_thresh=conf_thresh, nms_thresh=0.45)
            
            # Process each image in batch
            for i, (pred_boxes, pred_labels, pred_scores) in enumerate(predictions):
                # Ground truth for this image
                gt_boxes = targets[i]['boxes']
                gt_labels = targets[i]['labels']
                
                # Convert normalized coordinates to absolute
                if len(gt_boxes) > 0:
                    img_h, img_w = images[i].shape[1], images[i].shape[2]
                    gt_boxes = gt_boxes * torch.tensor([img_w, img_h, img_w, img_h])
                
                if len(pred_boxes) > 0:
                    img_h, img_w = images[i].shape[1], images[i].shape[2]  
                    pred_boxes = pred_boxes * torch.tensor([img_w, img_h, img_w, img_h])
                
                # Store for mAP calculation
                all_predictions.append({
                    'boxes': pred_boxes.cpu(),
                    'labels': pred_labels.cpu(),
                    'scores': pred_scores.cpu()
                })
                all_ground_truth.append({
                    'boxes': gt_boxes.cpu(),
                    'labels': gt_labels.cpu()
                })
                
                # Calculate per-class metrics
                if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                    # Calculate IoU matrix
                    iou_matrix = calculate_iou(pred_boxes.cpu(), gt_boxes.cpu())
                    
                    # Match predictions to ground truth
                    for class_id in torch.unique(gt_labels):
                        class_id_int = class_id.item()
                        
                        # Ground truth for this class
                        gt_class_mask = gt_labels == class_id
                        gt_class_boxes = gt_boxes[gt_class_mask]
                        
                        # Predictions for this class
                        pred_class_mask = pred_labels.cpu() == class_id
                        pred_class_boxes = pred_boxes.cpu()[pred_class_mask]
                        pred_class_scores = pred_scores.cpu()[pred_class_mask]
                        
                        # True positives and false positives
                        if len(pred_class_boxes) > 0 and len(gt_class_boxes) > 0:
                            class_iou = calculate_iou(pred_class_boxes, gt_class_boxes)
                            max_iou, max_indices = class_iou.max(dim=1)
                            
                            tp_mask = max_iou >= iou_thresh
                            tp_count = tp_mask.sum().item()
                            fp_count = len(pred_class_boxes) - tp_count
                            
                            per_class_stats[class_id_int]['tp'] += tp_count
                            per_class_stats[class_id_int]['fp'] += fp_count
                        else:
                            per_class_stats[class_id_int]['fp'] += len(pred_class_boxes)
                        
                        # False negatives
                        if len(gt_class_boxes) > 0:
                            if len(pred_class_boxes) == 0:
                                per_class_stats[class_id_int]['fn'] += len(gt_class_boxes)
                            else:
                                # Count unmatched ground truth
                                class_iou = calculate_iou(pred_class_boxes, gt_class_boxes)
                                max_iou_per_gt, _ = class_iou.max(dim=0)
                                fn_count = (max_iou_per_gt < iou_thresh).sum().item()
                                per_class_stats[class_id_int]['fn'] += fn_count
                
                elif len(gt_boxes) > 0:
                    # No predictions but ground truth exists
                    for class_id in torch.unique(gt_labels):
                        class_count = (gt_labels == class_id).sum().item()
                        per_class_stats[class_id.item()]['fn'] += class_count
                
                elif len(pred_boxes) > 0:
                    # Predictions but no ground truth
                    for class_id in torch.unique(pred_labels):
                        class_count = (pred_labels == class_id).sum().item()
                        per_class_stats[class_id.item()]['fp'] += class_count
    
    # Calculate final metrics
    results = calculate_final_metrics(per_class_stats, all_predictions, all_ground_truth, 
                                    class_names, iou_thresh)
    
    return results

def calculate_final_metrics(per_class_stats, all_predictions, all_ground_truth, class_names=None, iou_thresh=0.5):
    """Calculate final evaluation metrics for thesis"""
    
    print("📊 Calculating final evaluation metrics...")
    
    # Per-class metrics
    per_class_metrics = {}
    overall_tp, overall_fp, overall_fn = 0, 0, 0
    
    for class_id, stats in per_class_stats.items():
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        
        # Precision, Recall, F1-Score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class_{class_id}"
        
        per_class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
        
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
    
    # Overall metrics
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    # Calculate mAP (Mean Average Precision)
    map_scores = calculate_map(all_predictions, all_ground_truth, iou_thresh, class_names)
    
    # Average IoU calculation
    avg_iou = calculate_average_iou(all_predictions, all_ground_truth, iou_thresh)
    
    results = {
        'per_class': per_class_metrics,
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'mAP': map_scores['mAP'],
            'average_iou': avg_iou
        },
        'map_details': map_scores,
        'configuration': {
            'iou_threshold': iou_thresh,
            'total_images': len(all_predictions)
        }
    }
    
    return results

def calculate_map(predictions, ground_truth, iou_thresh=0.5, class_names=None):
    """Calculate Mean Average Precision (mAP) for thesis evaluation"""
    
    if not class_names:
        # Determine unique classes from ground truth
        all_classes = set()
        for gt in ground_truth:
            all_classes.update(gt['labels'].tolist())
        class_names = [f"Class_{i}" for i in sorted(all_classes)]
    
    class_aps = {}
    
    for class_idx, class_name in enumerate(class_names):
        class_id = class_idx + 1  # Assuming classes start from 1 (0 is background)
        
        # Collect all predictions and ground truth for this class
        class_predictions = []
        class_ground_truth = []
        
        for pred, gt in zip(predictions, ground_truth):
            # Predictions for this class
            class_mask = pred['labels'] == class_id
            if class_mask.any():
                class_pred_boxes = pred['boxes'][class_mask]
                class_pred_scores = pred['scores'][class_mask]
                
                for box, score in zip(class_pred_boxes, class_pred_scores):
                    class_predictions.append({
                        'box': box,
                        'score': score.item(),
                        'image_id': len(class_predictions)  # Simple image ID
                    })
            
            # Ground truth for this class
            gt_class_mask = gt['labels'] == class_id
            if gt_class_mask.any():
                class_gt_boxes = gt['boxes'][gt_class_mask]
                for box in class_gt_boxes:
                    class_ground_truth.append({
                        'box': box,
                        'image_id': len(class_ground_truth)
                    })
        
        # Calculate AP for this class
        if len(class_predictions) > 0 and len(class_ground_truth) > 0:
            ap = calculate_class_ap(class_predictions, class_ground_truth, iou_thresh)
            class_aps[class_name] = ap
        else:
            class_aps[class_name] = 0.0
    
    # Calculate mAP
    map_score = np.mean(list(class_aps.values())) if class_aps else 0.0
    
    return {
        'mAP': map_score,
        'per_class_AP': class_aps
    }

def calculate_class_ap(predictions, ground_truth, iou_thresh):
    """Calculate Average Precision for a single class"""
    if not predictions or not ground_truth:
        return 0.0
    
    # Sort predictions by confidence score (descending)
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    
    # Group ground truth by image
    gt_by_image = defaultdict(list)
    for gt in ground_truth:
        gt_by_image[gt['image_id']].append(gt['box'])
    
    # Process each prediction
    for i, pred in enumerate(predictions):
        pred_box = pred['box']
        img_id = pred['image_id']
        
        if img_id in gt_by_image:
            gt_boxes = gt_by_image[img_id]
            max_iou = 0.0
            
            for gt_box in gt_boxes:
                iou = calculate_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0))[0, 0].item()
                max_iou = max(max_iou, iou)
            
            if max_iou >= iou_thresh:
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    
    # Calculate precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(ground_truth)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for r in np.arange(0, 1.1, 0.1):
        precisions_above_r = precisions[recalls >= r]
        if len(precisions_above_r) > 0:
            ap += np.max(precisions_above_r)
    
    return ap / 11.0

def calculate_average_iou(predictions, ground_truth, iou_thresh=0.5):
    """Calculate average IoU for thesis evaluation"""
    total_iou = 0.0
    num_matches = 0
    
    for pred, gt in zip(predictions, ground_truth):
        if len(pred['boxes']) > 0 and len(gt['boxes']) > 0:
            iou_matrix = calculate_iou(pred['boxes'], gt['boxes'])
            max_iou_per_pred, _ = iou_matrix.max(dim=1)
            
            # Only count IoUs above threshold
            valid_ious = max_iou_per_pred[max_iou_per_pred >= iou_thresh]
            if len(valid_ious) > 0:
                total_iou += valid_ious.sum().item()
                num_matches += len(valid_ious)
    
    return total_iou / num_matches if num_matches > 0 else 0.0

def statistical_comparison(focusnet_results, baseline_results, alpha=0.05):
    """
    Perform Wilcoxon signed-rank test for thesis statistical analysis
    
    Args:
        focusnet_results: Results from FocusNet model
        baseline_results: Results from Baseline SSD model
        alpha: Significance level
    
    Returns:
        dict: Statistical comparison results
    """
    print("📈 Performing statistical comparison (Wilcoxon signed-rank test)...")
    
    # Extract per-class F1 scores for comparison
    focusnet_f1_scores = []
    baseline_f1_scores = []
    
    for class_name in focusnet_results['per_class'].keys():
        if class_name in baseline_results['per_class']:
            focusnet_f1_scores.append(focusnet_results['per_class'][class_name]['f1_score'])
            baseline_f1_scores.append(baseline_results['per_class'][class_name]['f1_score'])
    
    # Perform Wilcoxon signed-rank test
    if len(focusnet_f1_scores) > 0 and len(baseline_f1_scores) > 0:
        statistic, p_value = stats.wilcoxon(focusnet_f1_scores, baseline_f1_scores)
        
        # Determine significance
        is_significant = p_value < alpha
        
        # Calculate effect size (mean difference)
        mean_diff = np.mean(focusnet_f1_scores) - np.mean(baseline_f1_scores)
        
        comparison_results = {
            'test': 'Wilcoxon signed-rank test',
            'statistic': statistic,
            'p_value': p_value,
            'alpha': alpha,
            'is_significant': is_significant,
            'mean_difference': mean_diff,
            'focusnet_mean_f1': np.mean(focusnet_f1_scores),
            'baseline_mean_f1': np.mean(baseline_f1_scores),
            'interpretation': get_statistical_interpretation(p_value, alpha, mean_diff)
        }
    else:
        comparison_results = {
            'error': 'Insufficient data for statistical comparison'
        }
    
    return comparison_results

def get_statistical_interpretation(p_value, alpha, mean_diff):
    """Generate interpretation for thesis writing"""
    if p_value < alpha:
        if mean_diff > 0:
            return f"FocusNet significantly outperforms Baseline SSD (p={p_value:.4f} < {alpha})"
        else:
            return f"Baseline SSD significantly outperforms FocusNet (p={p_value:.4f} < {alpha})"
    else:
        return f"No significant difference between models (p={p_value:.4f} >= {alpha})"

def generate_thesis_report(focusnet_results, baseline_results, statistical_results, save_path=None):
    """
    Generate comprehensive thesis evaluation report
    
    Args:
        focusnet_results: FocusNet evaluation results
        baseline_results: Baseline SSD evaluation results  
        statistical_results: Statistical comparison results
        save_path: Path to save the report
    
    Returns:
        str: Formatted thesis report
    """
    report = []
    report.append("=" * 80)
    report.append("FOCUSNET THESIS EVALUATION REPORT")
    report.append("=" * 80)
    
    # Model comparison table
    report.append("\n📊 MODEL COMPARISON SUMMARY")
    report.append("-" * 50)
    report.append(f"{'Metric':<20} {'FocusNet':<15} {'Baseline SSD':<15} {'Improvement':<15}")
    report.append("-" * 50)
    
    metrics = ['precision', 'recall', 'f1_score', 'mAP', 'average_iou']
    for metric in metrics:
        focusnet_val = focusnet_results['overall'][metric]
        baseline_val = baseline_results['overall'][metric]
        improvement = ((focusnet_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
        
        report.append(f"{metric.replace('_', ' ').title():<20} {focusnet_val:<15.4f} {baseline_val:<15.4f} {improvement:+.2f}%")
    
    # Per-class results
    report.append(f"\n📋 PER-CLASS RESULTS")
    report.append("-" * 70)
    report.append(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Model':<10}")
    report.append("-" * 70)
    
    for class_name in focusnet_results['per_class'].keys():
        # FocusNet results
        fn_metrics = focusnet_results['per_class'][class_name]
        report.append(f"{class_name:<15} {fn_metrics['precision']:<12.4f} {fn_metrics['recall']:<12.4f} {fn_metrics['f1_score']:<12.4f} {'FocusNet':<10}")
        
        # Baseline results
        if class_name in baseline_results['per_class']:
            bs_metrics = baseline_results['per_class'][class_name]
            report.append(f"{'':<15} {bs_metrics['precision']:<12.4f} {bs_metrics['recall']:<12.4f} {bs_metrics['f1_score']:<12.4f} {'Baseline':<10}")
        
        report.append("")
    
    # Statistical analysis
    report.append(f"\n📈 STATISTICAL ANALYSIS")
    report.append("-" * 50)
    if 'error' not in statistical_results:
        report.append(f"Test: {statistical_results['test']}")
        report.append(f"P-value: {statistical_results['p_value']:.6f}")
        report.append(f"Significance level (α): {statistical_results['alpha']}")
        report.append(f"Result: {statistical_results['interpretation']}")
        report.append(f"Mean F1 difference: {statistical_results['mean_difference']:+.4f}")
    else:
        report.append(f"Error: {statistical_results['error']}")
    
    # Thesis conclusions
    report.append(f"\n🎓 THESIS CONCLUSIONS")
    report.append("-" * 50)
    
    overall_improvement = ((focusnet_results['overall']['f1_score'] - baseline_results['overall']['f1_score']) 
                          / baseline_results['overall']['f1_score'] * 100) if baseline_results['overall']['f1_score'] > 0 else 0
    
    if overall_improvement > 0:
        report.append(f"✅ FocusNet demonstrates superior performance with {overall_improvement:.2f}% F1-Score improvement")
        report.append(f"✅ CBAM attention mechanism effectively enhances road hazard detection")
        report.append(f"✅ Integration of attention with MobileNetV3 backbone proves beneficial")
    else:
        report.append(f"⚠️ FocusNet shows {abs(overall_improvement):.2f}% lower F1-Score than baseline")
        report.append(f"💭 Further optimization of CBAM integration may be needed")
    
    report.append(f"\n📊 KEY METRICS FOR THESIS:")
    report.append(f"   • mAP: {focusnet_results['overall']['mAP']:.4f}")
    report.append(f"   • Average IoU: {focusnet_results['overall']['average_iou']:.4f}")
    report.append(f"   • Overall F1-Score: {focusnet_results['overall']['f1_score']:.4f}")
    
    # Save report if path provided
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"📄 Thesis report saved to: {save_path}")
    
    return report_text

# Main evaluation function for thesis
def complete_thesis_evaluation(focusnet_model, baseline_model, test_loader, device, 
                              class_names=None, save_dir="/content/drive/MyDrive/thesis_results"):
    """
    Complete evaluation pipeline for FocusNet thesis
    
    Args:
        focusnet_model: Trained FocusNet model
        baseline_model: Trained Baseline SSD model
        test_loader: Test dataset loader
        device: CUDA device
        class_names: List of class names
        save_dir: Directory to save results
    
    Returns:
        dict: Complete evaluation results
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("🎓 STARTING COMPLETE THESIS EVALUATION")
    print("=" * 60)
    
    # Evaluate FocusNet
    print("🔍 Evaluating FocusNet...")
    focusnet_results = evaluate_detection(focusnet_model, test_loader, device, 
                                        iou_thresh=0.5, conf_thresh=0.5, class_names=class_names)
    
    # Evaluate Baseline SSD  
    print("🔍 Evaluating Baseline SSD...")
    baseline_results = evaluate_detection(baseline_model, test_loader, device,
                                        iou_thresh=0.5, conf_thresh=0.5, class_names=class_names)
    
    # Statistical comparison
    print("📊 Performing statistical analysis...")
    statistical_results = statistical_comparison(focusnet_results, baseline_results)
    
    # Generate thesis report
    print("📄 Generating thesis report...")
    report = generate_thesis_report(focusnet_results, baseline_results, statistical_results,
                                   os.path.join(save_dir, "thesis_evaluation_report.txt"))
    
    # Save detailed results
    detailed_results = {
        'focusnet_results': focusnet_results,
        'baseline_results': baseline_results,
        'statistical_comparison': statistical_results,
        'thesis_report': report
    }
    
    results_path = os.path.join(save_dir, "detailed_evaluation_results.json")
    with open(results_path, 'w') as f:
        # Convert tensors to lists for JSON serialization
        json_results = convert_tensors_to_lists(detailed_results)
        json.dump(json_results, f, indent=2)
    
    print(f"✅ Complete evaluation saved to: {save_dir}")
    print("🎓 Thesis evaluation completed successfully!")
    
    return detailed_results

def convert_tensors_to_lists(obj):
    """Convert tensors to lists for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_lists(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Example usage
if __name__ == "__main__":
    print("🎓 FocusNet Thesis Evaluation Toolkit")
    print("✅ All evaluation functions loaded successfully!")
    print("📊 Ready for comprehensive model comparison and statistical analysis")