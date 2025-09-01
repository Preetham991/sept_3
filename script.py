import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import LabelBinarizer
from scipy.special import softmax
from scipy.stats import entropy, pearsonr, spearmanr, kendalltau
from scipy.optimize import minimize_scalar
import warnings
import os
warnings.filterwarnings('ignore')

# Create figures directory
os.makedirs('./figures', exist_ok=True)

def generate_dataset(n_samples=500, n_classes=5, random_state=42):
    """Generate synthetic email classification dataset with imbalanced classes."""
    np.random.seed(random_state)
    
    # Class names
    classes = ['Spam', 'Promotions', 'Social', 'Updates', 'Forums']
    
    # Simulate imbalanced class distribution
    class_probs = [0.15, 0.25, 0.20, 0.30, 0.10]  # Imbalanced
    true_labels = np.random.choice(n_classes, n_samples, p=class_probs)
    
    # Generate feature vectors (simulating email embeddings)
    features = np.random.randn(n_samples, 128)
    
    # Add class-specific patterns to features
    for i in range(n_classes):
        mask = (true_labels == i)
        if np.any(mask):
            features[mask] += np.random.randn(128) * 0.5
    
    # Simulate model predictions with some miscalibration
    raw_logits = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        # Base prediction influenced by features
        raw_logits[i] = features[i, :n_classes] + np.random.randn(n_classes) * 0.3
        # Add bias toward correct class
        raw_logits[i, true_labels[i]] += 1.5 + np.random.randn() * 0.5
    
    # Add systematic miscalibration (overconfidence)
    raw_logits *= 1.8  # Temperature > 1 makes model overconfident
    
    probabilities = softmax(raw_logits, axis=1)
    predicted_labels = np.argmax(probabilities, axis=1)
    
    return {
        'features': features,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'raw_logits': raw_logits,
        'probabilities': probabilities,
        'classes': classes,
        'class_probs': class_probs
    }

def confidence_scores(probabilities, logits=None):
    """Compute various confidence scores."""
    scores = {}
    
    # Maximum Softmax Probability (MSP)
    scores['msp'] = np.max(probabilities, axis=1)
    
    # Entropy-based confidence
    scores['entropy'] = -np.array([entropy(p) for p in probabilities])
    scores['entropy_normalized'] = scores['entropy'] / np.log(probabilities.shape[1])
    
    # Top-1 vs Top-2 margin
    sorted_probs = np.sort(probabilities, axis=1)
    scores['margin'] = sorted_probs[:, -1] - sorted_probs[:, -2]
    
    # Energy score (if logits available)
    if logits is not None:
        scores['energy'] = -np.log(np.sum(np.exp(logits), axis=1))
    
    # Variance across logits
    if logits is not None:
        scores['logit_variance'] = -np.var(logits, axis=1)
    
    return scores

def calibration_methods(probabilities, true_labels, predicted_labels, method='temperature'):
    """Apply post-hoc calibration methods."""
    if method == 'temperature':
        # Temperature scaling
        def temperature_nll(T):
            scaled_probs = softmax(np.log(probabilities + 1e-12) / T, axis=1)
            return log_loss(true_labels, scaled_probs)
        
        result = minimize_scalar(temperature_nll, bounds=(0.1, 10.0), method='bounded')
        optimal_T = result.x
        calibrated_probs = softmax(np.log(probabilities + 1e-12) / optimal_T, axis=1)
        
        return calibrated_probs, {'temperature': optimal_T}
    
    elif method == 'platt':
        # Platt scaling using LogisticRegression
        confidence = np.max(probabilities, axis=1).reshape(-1, 1)
        correct = (predicted_labels == true_labels).astype(int)
        
        platt_model = LogisticRegression()
        platt_model.fit(confidence, correct)
        
        calibrated_confidence = platt_model.predict_proba(confidence)[:, 1]
        
        # Convert back to full probability matrix
        calibrated_probs = probabilities.copy()
        for i in range(len(calibrated_probs)):
            if calibrated_confidence[i] < probabilities[i, predicted_labels[i]]:
                # Reduce confidence in predicted class
                factor = calibrated_confidence[i] / probabilities[i, predicted_labels[i]]
                calibrated_probs[i, predicted_labels[i]] *= factor
                # Redistribute remaining probability mass
                remaining = (1 - calibrated_probs[i, predicted_labels[i]]) / (probabilities.shape[1] - 1)
                for j in range(probabilities.shape[1]):
                    if j != predicted_labels[i]:
                        calibrated_probs[i, j] = remaining
        
        return calibrated_probs, {'platt_model': platt_model}
    
    elif method == 'isotonic':
        # Isotonic regression
        confidence = np.max(probabilities, axis=1)
        correct = (predicted_labels == true_labels).astype(int)
        
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(confidence, correct)
        
        calibrated_confidence = iso_reg.predict(confidence)
        
        # Convert back to full probability matrix (simplified approach)
        calibrated_probs = probabilities.copy()
        for i in range(len(calibrated_probs)):
            if calibrated_confidence[i] != confidence[i]:
                factor = calibrated_confidence[i] / confidence[i] if confidence[i] > 0 else 1
                calibrated_probs[i] *= factor
                calibrated_probs[i] /= np.sum(calibrated_probs[i])  # Renormalize
        
        return calibrated_probs, {'isotonic_model': iso_reg}

def compute_metrics(probabilities, true_labels, predicted_labels, confidence_scores_dict):
    """Compute comprehensive evaluation metrics."""
    metrics = {}
    n_classes = probabilities.shape[1]
    
    # Basic accuracy
    metrics['accuracy'] = accuracy_score(true_labels, predicted_labels)
    
    # Negative Log-Likelihood
    metrics['nll'] = log_loss(true_labels, probabilities)
    
    # Brier Score
    lb = LabelBinarizer()
    lb.fit(range(n_classes))
    true_binary = lb.transform(true_labels)
    metrics['brier'] = np.mean(np.sum((probabilities - true_binary) ** 2, axis=1))
    
    # Expected Calibration Error (ECE)
    def compute_ece(probs, labels, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = predictions == labels
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    metrics['ece'] = compute_ece(probabilities, true_labels)
    
    # Maximum Calibration Error (MCE)
    def compute_mce(probs, labels, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = predictions == labels
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    metrics['mce'] = compute_mce(probabilities, true_labels)
    
    # ROC-AUC for confidence vs correctness
    correct = (predicted_labels == true_labels).astype(int)
    if 'msp' in confidence_scores_dict:
        try:
            metrics['confidence_auroc'] = roc_auc_score(correct, confidence_scores_dict['msp'])
        except:
            metrics['confidence_auroc'] = 0.5
    
    # Correlations between confidence and correctness
    if 'msp' in confidence_scores_dict:
        metrics['pearson_corr'], _ = pearsonr(confidence_scores_dict['msp'], correct)
        metrics['spearman_corr'], _ = spearmanr(confidence_scores_dict['msp'], correct)
        metrics['kendall_corr'], _ = kendalltau(confidence_scores_dict['msp'], correct)
    
    return metrics

def generate_visualizations(data, confidence_dict, metrics_dict, calibrated_data=None):
    """Generate all 14 required plots."""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Overall Reliability Diagram
    plt.figure(figsize=(8, 6))
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(data['probabilities'], axis=1)
    predictions = np.argmax(data['probabilities'], axis=1)
    accuracies = predictions == data['true_labels']
    
    bin_centers = []
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_sizes.append(prop_in_bin)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, label='Observed accuracy')
    plt.plot(bin_centers, bin_confs, 'ro-', label='Mean confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram (Overall)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figures/reliability_overall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class Reliability Diagram
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for class_idx in range(len(data['classes'])):
        ax = axes[class_idx]
        
        class_mask = data['predicted_labels'] == class_idx
        if np.sum(class_mask) > 0:
            class_confidences = confidences[class_mask]
            class_correct = (data['true_labels'][class_mask] == class_idx).astype(int)
            
            # Simple binning for per-class
            if len(class_confidences) > 5:
                n_bins_class = min(5, len(class_confidences) // 2)
                hist, bin_edges = np.histogram(class_confidences, bins=n_bins_class)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                bin_accs_class = []
                bin_confs_class = []
                for i in range(len(bin_centers)):
                    mask = (class_confidences >= bin_edges[i]) & (class_confidences < bin_edges[i+1])
                    if np.sum(mask) > 0:
                        bin_accs_class.append(class_correct[mask].mean())
                        bin_confs_class.append(class_confidences[mask].mean())
                    else:
                        bin_accs_class.append(0)
                        bin_confs_class.append(bin_centers[i])
                
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                ax.bar(bin_centers, bin_accs_class, width=(bin_edges[1]-bin_edges[0])*0.8, alpha=0.7)
                ax.plot(bin_centers, bin_confs_class, 'ro-')
        
        ax.set_title(f'{data["classes"][class_idx]}')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig('./figures/reliability_per_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Adaptive Reliability Diagram
    plt.figure(figsize=(8, 6))
    sorted_indices = np.argsort(confidences)
    sorted_confidences = confidences[sorted_indices]
    sorted_accuracies = accuracies[sorted_indices]
    
    # Use quantile-based bins
    n_bins = 10
    bin_indices = np.array_split(range(len(sorted_confidences)), n_bins)
    
    bin_centers = []
    bin_accs = []
    bin_confs = []
    
    for indices in bin_indices:
        if len(indices) > 0:
            bin_accs.append(sorted_accuracies[indices].mean())
            bin_confs.append(sorted_confidences[indices].mean())
            bin_centers.append(len(indices))
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.scatter(bin_confs, bin_accs, s=[c*5 for c in bin_centers], alpha=0.7, label='Adaptive bins')
    plt.plot(bin_confs, bin_accs, 'r-', alpha=0.5)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Adaptive Reliability Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figures/reliability_adaptive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Boxplot Agreement
    plt.figure(figsize=(10, 6))
    correct_mask = (data['predicted_labels'] == data['true_labels'])
    confidence_data = [confidences[correct_mask], confidences[~correct_mask]]
    
    box_plot = plt.boxplot(confidence_data, labels=['Correct', 'Incorrect'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightgreen')
    box_plot['boxes'][1].set_facecolor('lightcoral')
    
    plt.ylabel('Confidence Score')
    plt.title('Confidence Distribution by Correctness')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figures/boxplot_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Heatmap Confidence vs Correctness
    plt.figure(figsize=(8, 6))
    conf_bins = np.linspace(0, 1, 11)
    correct_counts = np.zeros((10, 2))  # [incorrect, correct]
    
    for i in range(10):
        mask = (confidences >= conf_bins[i]) & (confidences < conf_bins[i+1])
        if np.sum(mask) > 0:
            correct_counts[i, 0] = np.sum(~accuracies[mask])
            correct_counts[i, 1] = np.sum(accuracies[mask])
    
    plt.imshow(correct_counts.T, aspect='auto', cmap='Blues', origin='lower')
    plt.colorbar(label='Count')
    plt.xlabel('Confidence Bin')
    plt.ylabel('Prediction Outcome')
    plt.title('Confidence vs Correctness Heatmap')
    plt.yticks([0, 1], ['Incorrect', 'Correct'])
    plt.xticks(range(10), [f'{conf_bins[i]:.1f}' for i in range(10)])
    plt.tight_layout()
    plt.savefig('./figures/heatmap_confidence_correctness.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Confidence Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(confidences[accuracies], bins=20, alpha=0.7, label='Correct', color='green', density=True)
    plt.hist(confidences[~accuracies], bins=20, alpha=0.7, label='Incorrect', color='red', density=True)
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Confidence Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figures/confidence_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Violin Plot Per Class
    plt.figure(figsize=(12, 6))
    class_confidences = []
    class_labels = []
    
    for i, class_name in enumerate(data['classes']):
        mask = data['predicted_labels'] == i
        if np.sum(mask) > 0:
            class_confidences.extend(confidences[mask])
            class_labels.extend([class_name] * np.sum(mask))
    
    df_violin = pd.DataFrame({'Confidence': class_confidences, 'Class': class_labels})
    sns.violinplot(data=df_violin, x='Class', y='Confidence')
    plt.title('Confidence Distribution by Predicted Class')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figures/violin_per_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Confidence-Error Curve
    plt.figure(figsize=(8, 6))
    sorted_indices = np.argsort(-confidences)  # Sort by decreasing confidence
    sorted_confidences = confidences[sorted_indices]
    sorted_errors = (~accuracies[sorted_indices]).astype(int)
    
    # Cumulative error rate
    cumulative_errors = np.cumsum(sorted_errors) / np.arange(1, len(sorted_errors) + 1)
    coverage = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    
    plt.plot(coverage, cumulative_errors, 'b-', label='Error rate')
    plt.xlabel('Coverage (fraction of samples)')
    plt.ylabel('Error rate')
    plt.title('Confidence-Error Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./figures/confidence_error_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. Temperature Sweep
    plt.figure(figsize=(8, 6))
    temperatures = np.logspace(-1, 1, 20)
    eces = []
    nlls = []
    
    for T in temperatures:
        temp_probs = softmax(np.log(data['probabilities'] + 1e-12) / T, axis=1)
        temp_ece = compute_ece(temp_probs, data['true_labels'])
        temp_nll = log_loss(data['true_labels'], temp_probs)
        eces.append(temp_ece)
        nlls.append(temp_nll)
    
    plt.subplot(1, 2, 1)
    plt.semilogx(temperatures, eces, 'b-o')
    plt.xlabel('Temperature')
    plt.ylabel('ECE')
    plt.title('ECE vs Temperature')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogx(temperatures, nlls, 'r-o')
    plt.xlabel('Temperature')
    plt.ylabel('NLL')
    plt.title('NLL vs Temperature')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figures/temperature_sweep.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 10. Risk-Coverage Curve
    plt.figure(figsize=(8, 6))
    # Sort by confidence (descending)
    sorted_indices = np.argsort(-confidences)
    sorted_errors = (~accuracies[sorted_indices]).astype(int)
    
    coverages = []
    risks = []
    
    for i in range(1, len(sorted_indices) + 1):
        coverage = i / len(sorted_indices)
        risk = np.sum(sorted_errors[:i]) / i
        coverages.append(coverage)
        risks.append(risk)
    
    plt.plot(coverages, risks, 'b-', label='Risk-Coverage')
    plt.xlabel('Coverage')
    plt.ylabel('Risk (Error Rate)')
    plt.title('Risk-Coverage Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./figures/risk_coverage_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 11-14. ROC, PR, Cumulative Gain, Lift Chart
    correct = accuracies.astype(int)
    
    # 11. ROC Overlay
    from sklearn.metrics import roc_curve
    plt.figure(figsize=(8, 6))
    
    fpr, tpr, _ = roc_curve(correct, confidences)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Confidence vs Correctness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figures/roc_overlay.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 12. PR Overlay
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(correct, confidences)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.axhline(y=correct.mean(), color='k', linestyle='--', label='Random')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve: Confidence vs Correctness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figures/pr_overlay.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 13. Cumulative Gain
    plt.figure(figsize=(8, 6))
    sorted_indices = np.argsort(-confidences)
    sorted_correct = correct[sorted_indices]
    
    cumulative_correct = np.cumsum(sorted_correct)
    total_correct = np.sum(correct)
    
    x_values = np.arange(1, len(cumulative_correct) + 1) / len(cumulative_correct)
    y_values = cumulative_correct / total_correct
    
    plt.plot(x_values, y_values, 'b-', label='Cumulative Gain')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('Fraction of Dataset')
    plt.ylabel('Fraction of Positive Cases Found')
    plt.title('Cumulative Gain Chart')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figures/cumulative_gain.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 14. Lift Chart
    plt.figure(figsize=(8, 6))
    baseline_rate = correct.mean()
    
    # Calculate lift for deciles
    n_deciles = 10
    decile_size = len(confidences) // n_deciles
    lifts = []
    deciles = []
    
    for i in range(n_deciles):
        start_idx = i * decile_size
        end_idx = min((i + 1) * decile_size, len(sorted_indices))
        decile_indices = sorted_indices[start_idx:end_idx]
        
        decile_rate = correct[decile_indices].mean()
        lift = decile_rate / baseline_rate if baseline_rate > 0 else 1
        lifts.append(lift)
        deciles.append(i + 1)
    
    plt.bar(deciles, lifts, alpha=0.7, color='skyblue')
    plt.axhline(y=1, color='k', linestyle='--', label='Baseline')
    plt.xlabel('Decile (by Confidence)')
    plt.ylabel('Lift')
    plt.title('Lift Chart')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figures/lift_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(data, metrics, calibrated_metrics, confidence_dict, output_file='Email_Confidence_Results.txt'):
    """Save comprehensive results to text file."""
    
    with open(output_file, 'w') as f:
        f.write("EMAIL CONFIDENCE SCORE EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        # Dataset Summary
        f.write("DATASET SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total samples: {len(data['true_labels'])}\n")
        f.write(f"Number of classes: {len(data['classes'])}\n")
        f.write(f"Classes: {', '.join(data['classes'])}\n\n")
        
        # Class distribution
        f.write("Class Distribution:\n")
        for i, class_name in enumerate(data['classes']):
            count = np.sum(data['true_labels'] == i)
            percentage = count / len(data['true_labels']) * 100
            f.write(f"  {class_name}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Agreement ratios
        accuracy = accuracy_score(data['true_labels'], data['predicted_labels'])
        f.write(f"Overall Accuracy: {accuracy:.3f}\n\n")
        
        # Raw Model Metrics
        f.write("RAW MODEL METRICS\n")
        f.write("-" * 20 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric.upper()}: {value:.4f}\n")
        f.write("\n")
        
        # Calibrated Model Metrics
        f.write("CALIBRATED MODEL METRICS (Temperature Scaling)\n")
        f.write("-" * 50 + "\n")
        for metric, value in calibrated_metrics.items():
            f.write(f"{metric.upper()}: {value:.4f}\n")
        f.write("\n")
        
        # Confidence Score Analysis
        f.write("CONFIDENCE SCORE ANALYSIS\n")
        f.write("-" * 30 + "\n")
        for conf_type, scores in confidence_dict.items():
            f.write(f"{conf_type.upper()}:\n")
            f.write(f"  Mean: {np.mean(scores):.4f}\n")
            f.write(f"  Std: {np.std(scores):.4f}\n")
            f.write(f"  Min: {np.min(scores):.4f}\n")
            f.write(f"  Max: {np.max(scores):.4f}\n\n")
        
        # Correlation Analysis
        f.write("CORRELATION ANALYSIS\n")
        f.write("-" * 25 + "\n")
        f.write("Confidence vs Correctness correlations:\n")
        if 'pearson_corr' in metrics:
            f.write(f"  Pearson: {metrics['pearson_corr']:.4f}\n")
            f.write(f"  Spearman: {metrics['spearman_corr']:.4f}\n")
            f.write(f"  Kendall: {metrics['kendall_corr']:.4f}\n")
        f.write("\n")
        
        # Per-class Performance
        f.write("PER-CLASS PERFORMANCE\n")
        f.write("-" * 25 + "\n")
        for i, class_name in enumerate(data['classes']):
            class_mask = data['predicted_labels'] == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean((data['true_labels'][class_mask] == i))
                class_conf = np.mean(np.max(data['probabilities'][class_mask], axis=1))
                f.write(f"{class_name}:\n")
                f.write(f"  Predictions: {np.sum(class_mask)}\n")
                f.write(f"  Accuracy: {class_acc:.3f}\n")
                f.write(f"  Avg Confidence: {class_conf:.3f}\n\n")
        
        # Summary Comparison
        f.write("CALIBRATION IMPROVEMENT SUMMARY\n")
        f.write("-" * 35 + "\n")
        ece_improvement = (metrics['ece'] - calibrated_metrics['ece']) / metrics['ece'] * 100
        nll_improvement = (metrics['nll'] - calibrated_metrics['nll']) / metrics['nll'] * 100
        f.write(f"ECE Improvement: {ece_improvement:.1f}%\n")
        f.write(f"NLL Improvement: {nll_improvement:.1f}%\n")
        f.write("\n")
        
        f.write("VISUALIZATION FILES GENERATED\n")
        f.write("-" * 30 + "\n")
        plot_files = [
            "reliability_overall.png", "reliability_per_class.png", "reliability_adaptive.png",
            "boxplot_agreement.png", "heatmap_confidence_correctness.png", "confidence_histogram.png",
            "violin_per_class.png", "confidence_error_curve.png", "temperature_sweep.png",
            "risk_coverage_curve.png", "roc_overlay.png", "pr_overlay.png", 
            "cumulative_gain.png", "lift_chart.png"
        ]
        for i, filename in enumerate(plot_files, 1):
            f.write(f"{i:2d}. {filename}\n")

def run_experiment():
    """Main experiment pipeline."""
    print("Generating synthetic email classification dataset...")
    
    # Generate dataset
    data = generate_dataset()
    
    print("Computing confidence scores...")
    
    # Compute confidence scores
    conf_scores = confidence_scores(data['probabilities'], data['raw_logits'])
    
    print("Computing baseline metrics...")
    
    # Compute baseline metrics
    metrics = compute_metrics(data['probabilities'], data['true_labels'], 
                            data['predicted_labels'], conf_scores)
    
    print("Applying calibration methods...")
    
    # Apply temperature scaling
    calibrated_probs, calib_params = calibration_methods(
        data['probabilities'], data['true_labels'], data['predicted_labels'], 'temperature'
    )
    
    # Compute calibrated confidence scores and metrics
    calib_conf_scores = confidence_scores(calibrated_probs)
    calibrated_metrics = compute_metrics(calibrated_probs, data['true_labels'], 
                                       data['predicted_labels'], calib_conf_scores)
    
    print("Generating visualizations...")
    
    # Generate all visualizations
    generate_visualizations(data, conf_scores, metrics)
    
    print("Saving results...")
    
    # Save results
    save_results(data, metrics, calibrated_metrics, conf_scores)
    
    print("Experiment completed successfully!")
    print(f"Temperature scaling optimal T: {calib_params['temperature']:.3f}")
    print(f"ECE improvement: {((metrics['ece'] - calibrated_metrics['ece']) / metrics['ece'] * 100):.1f}%")
    print(f"Results saved to: Email_Confidence_Results.txt")
    print(f"Figures saved to: ./figures/ directory")
    
    return data, metrics, calibrated_metrics, conf_scores

# Helper function for ECE calculation
def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

if __name__ == "__main__":
    # Run the complete experiment
    data, metrics, calibrated_metrics, conf_scores = run_experiment()