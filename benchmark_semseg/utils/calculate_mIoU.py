import numpy as np
from sklearn.metrics import confusion_matrix

ep = 0.000001
#s_matrix = np.array([[1.0, 0.0, 0.0, 0.0],
#                     [0.0, 1.0, 1.0, 0.0],
#                     [0.0, 1.0, 1.0, 0.0],
#                     [0.0, 0.0, 0.0, 1.0]])

# false positive and false negative computation for SIOU
def fp_fn_sim(conf_matrix, s_matrix):
    return conf_matrix * s_matrix

# true positive computation for SIOU
def tp_sim(conf_matrix, s_matrix):
    # Multiply confusion matrix columns by s matrix columns
    column_products = np.sum(conf_matrix * s_matrix, axis=0)
    # Multiply confusion matrix rows by s matrix rows
    row_products = np.sum(conf_matrix * s_matrix, axis=1)
    # Sum the products to get the true positive
    tp = (column_products + row_products) / 2
    return tp

def calculate_siou(conf_matrix, s_matrix, d_matrix):
    # Similarity matrix IOU calculation
    tp = tp_sim(conf_matrix, s_matrix)
    fp_fn = fp_fn_sim(conf_matrix, d_matrix)
    fp = np.sum(fp_fn, axis=1)
    fn = np.sum(fp_fn, axis=0)
    siou = tp/(tp + fp + fn + ep)
    msiou = np.mean(siou)
    return siou, msiou

def metric_evaluate(predicted_label, gt_label, NUM_CLASS):
    """
    Evaluate segmentation metrics: Overall Accuracy (OA), Mean IoU, per-class IoU, mSIOU, and SIOU.
    
    :param predicted_label: (B, N) numpy array or torch tensor
    :param gt_label: (B, N) numpy array or torch tensor
    :param NUM_CLASS: int
    :return: oa, mean_IoU, iou_list, msiou, siou
    """
    print("[metric_evaluate] Starting evaluation...")

    # Convert to NumPy arrays if they are torch tensors
    if not isinstance(predicted_label, np.ndarray):
        predicted_label = predicted_label.cpu().numpy()
    if not isinstance(gt_label, np.ndarray):
        gt_label = gt_label.cpu().numpy()

    print("[metric_evaluate] Flattening inputs...")
    gt_flat = gt_label.flatten()
    pred_flat = predicted_label.flatten()

    print("[metric_evaluate] Computing confusion matrix...")
    conf_matrix = confusion_matrix(gt_flat, pred_flat, labels=list(range(NUM_CLASS)))

    print("[metric_evaluate] Computing per-class IoU...")
    intersection = np.diag(conf_matrix)
    union = conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - intersection
    iou_list = intersection / (union + 1e-6)

    print("[metric_evaluate] Computing mean IoU and overall accuracy...")
    mean_IoU = np.mean(iou_list)
    oa = intersection.sum() / conf_matrix.sum()

    print("[metric_evaluate] Computing siou and msiou...")
    s_matrix = np.eye(NUM_CLASS)
    d_matrix = 1 - s_matrix
    siou, msiou = calculate_siou(conf_matrix, s_matrix, d_matrix)

    print("[metric_evaluate] Done.")
    print(f"  - Overall Accuracy: {oa:.4f}")
    print(f"  - Mean IoU: {mean_IoU:.4f}")
    print(f"  - Per-class IoU: {[round(float(i), 4) for i in iou_list]}")

    try:
        print(f"  - SIOU: {float(siou):.4f}")
        print(f"  - MSIOU: {float(msiou):.4f}")
    except Exception as e:
        print(f"[WARNING] Failed to print SIOU/MSIOU: {e}")
        print(f"  - SIOU raw: {siou}")
        print(f"  - MSIOU raw: {msiou}")


    return oa, mean_IoU, iou_list.tolist(), msiou, siou
