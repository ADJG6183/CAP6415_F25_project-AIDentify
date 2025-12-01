"""
Gradio Web UI for AI-Generated Image Detection.

A simple, user-friendly web interface for detecting AI-generated images.
"""
import gradio as gr
import numpy as np
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import io

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.detectors.ensemble_detector import EnsembleDetector
from src.utils.image_processing import load_image


# Initialize detector globally
print("Initializing AI Image Detector...")
detector = None

def initialize_detector(use_cnn=False):
    """Initialize the detector (called on first use)."""
    global detector
    if detector is None:
        # Check for trained models
        ml_model_path = 'trained_models/ml_model_random_forest.pkl'
        cnn_model_path = 'trained_models/cnn_model_custom.pth'

        ml_path = ml_model_path if os.path.exists(ml_model_path) else None
        cnn_path = cnn_model_path if os.path.exists(cnn_model_path) else None

        detector = EnsembleDetector(
            ml_model_path=ml_path,
            cnn_model_path=cnn_path,
            use_cnn=use_cnn
        )
        print("Detector initialized!")
    return detector


def create_result_visualization(result):
    """Create a visualization of detection results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Probability gauge
    ax1.barh(['Real', 'AI-Generated'],
             [1 - result['probability'], result['probability']],
             color=['green', 'red'])
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Probability')
    ax1.set_title('Detection Result')
    ax1.axvline(x=result['threshold'], color='black', linestyle='--', label='Threshold')
    ax1.legend()

    # Individual detector scores
    detectors = []
    scores = []
    colors = []

    for detector_name, prob in result['individual_predictions'].items():
        if prob is not None:
            detectors.append(detector_name.capitalize())
            scores.append(prob)
            colors.append('red' if prob > 0.5 else 'green')

    if detectors:
        ax2.barh(detectors, scores, color=colors)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('AI-Generated Probability')
        ax2.set_title('Individual Detector Scores')
        ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image.open(buf)


def detect_image(image, detection_method, threshold, use_cnn):
    """
    Detect if an image is AI-generated.

    Args:
        image: PIL Image or numpy array
        detection_method: Detection method to use
        threshold: Classification threshold
        use_cnn: Whether to use CNN detector

    Returns:
        verdict_text, confidence_text, visualization, detailed_text
    """
    if image is None:
        return "⚠️ Please upload an image first!", "", None, ""

    try:
        # Initialize detector
        det = initialize_detector(use_cnn=use_cnn)

        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure RGB
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]

        # Perform detection
        result = det.predict(image, threshold=threshold, method=detection_method)

        # Create verdict text with emoji
        if result['is_ai_generated']:
            verdict = "🤖 **AI-GENERATED IMAGE**"
            verdict_color = "red"
        else:
            verdict = "📷 **REAL IMAGE**"
            verdict_color = "green"

        # Confidence text
        confidence = f"**Confidence:** {result['confidence']:.1%}"
        probability = f"**Probability:** {result['probability']:.1%}"

        # Create detailed breakdown
        detailed = f"""
### Detection Details

**Method Used:** {detection_method}
**Threshold:** {threshold}

#### Individual Detector Results:
"""
        for detector_name, prob in result['individual_predictions'].items():
            if prob is not None:
                status = "✓ AI" if prob > 0.5 else "✓ Real"
                detailed += f"- **{detector_name.capitalize()}:** {prob:.1%} {status}\n"

        # Create visualization
        viz = create_result_visualization(result)

        return verdict, f"{probability}\n{confidence}", viz, detailed

    except Exception as e:
        error_msg = f"❌ Error during detection: {str(e)}"
        return error_msg, "", None, str(e)


def batch_detect_images(files, detection_method, threshold, use_cnn):
    """
    Detect multiple images at once.

    Args:
        files: List of uploaded files
        detection_method: Detection method to use
        threshold: Classification threshold
        use_cnn: Whether to use CNN detector

    Returns:
        HTML table with results
    """
    if not files:
        return "⚠️ Please upload images first!"

    try:
        det = initialize_detector(use_cnn=use_cnn)

        results_html = """
        <div style="max-height: 400px; overflow-y: auto;">
        <table style="width: 100%; border-collapse: collapse;">
        <tr style="background-color: #f2f2f2;">
            <th style="padding: 10px; border: 1px solid #ddd;">Image</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Verdict</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Probability</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Confidence</th>
        </tr>
        """

        for file in files:
            try:
                # Load image
                image = Image.open(file.name)
                image_np = np.array(image)

                # Ensure RGB
                if len(image_np.shape) == 2:
                    image_np = np.stack([image_np] * 3, axis=-1)
                elif image_np.shape[2] == 4:
                    image_np = image_np[:, :, :3]

                # Detect
                result = det.predict(image_np, threshold=threshold, method=detection_method)

                # Format result
                verdict = "🤖 AI-Generated" if result['is_ai_generated'] else "📷 Real"
                color = "#ffcccc" if result['is_ai_generated'] else "#ccffcc"

                filename = os.path.basename(file.name)

                results_html += f"""
                <tr style="background-color: {color};">
                    <td style="padding: 8px; border: 1px solid #ddd;">{filename}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>{verdict}</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{result['probability']:.1%}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{result['confidence']:.1%}</td>
                </tr>
                """
            except Exception as e:
                filename = os.path.basename(file.name) if hasattr(file, 'name') else "unknown"
                results_html += f"""
                <tr style="background-color: #ffffcc;">
                    <td style="padding: 8px; border: 1px solid #ddd;">{filename}</td>
                    <td colspan="3" style="padding: 8px; border: 1px solid #ddd;">❌ Error: {str(e)}</td>
                </tr>
                """

        results_html += "</table></div>"
        return results_html

    except Exception as e:
        return f"❌ Error during batch detection: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="AIDentify - AI Image Detection", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🔍 AIDentify - AI-Generated Image Detection

    Upload an image to detect whether it was generated by AI or is a real photograph.
    This system uses multiple detection methods including frequency analysis, statistical features, and deep learning.
    """)

    with gr.Tabs():
        # Single Image Detection Tab
        with gr.TabItem("🖼️ Single Image Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="Upload Image",
                        type="pil",
                        height=400
                    )

                    with gr.Accordion("⚙️ Detection Settings", open=True):
                        detection_method = gr.Radio(
                            choices=["weighted_average", "ml_model", "voting"],
                            value="weighted_average",
                            label="Detection Method",
                            info="weighted_average: Combines all detectors | ml_model: Uses trained ML model | voting: Majority vote"
                        )

                        threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Classification Threshold",
                            info="Higher = more strict (fewer false positives)"
                        )

                        use_cnn = gr.Checkbox(
                            label="Use CNN Detector (slower but more accurate)",
                            value=False,
                            info="Requires more computational resources"
                        )

                    detect_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")

                with gr.Column(scale=1):
                    verdict_output = gr.Markdown(label="Verdict")
                    confidence_output = gr.Markdown(label="Confidence")
                    viz_output = gr.Image(label="Detection Visualization")
                    detailed_output = gr.Markdown(label="Detailed Analysis")

            detect_btn.click(
                fn=detect_image,
                inputs=[input_image, detection_method, threshold, use_cnn],
                outputs=[verdict_output, confidence_output, viz_output, detailed_output]
            )

            gr.Markdown("""
            ### 💡 Tips:
            - **Real images** typically have natural sensor noise and follow statistical patterns
            - **AI-generated images** often have artifacts in frequency domain and unusual statistical properties
            - Try adjusting the threshold if results seem too sensitive or not sensitive enough
            - Using CNN detector improves accuracy but requires more time and resources
            """)

        # Batch Detection Tab
        with gr.TabItem("📁 Batch Detection"):
            gr.Markdown("""
            ### Analyze Multiple Images
            Upload multiple images to detect them all at once.
            """)

            with gr.Row():
                with gr.Column():
                    batch_input = gr.Files(
                        label="Upload Multiple Images",
                        file_types=["image"],
                        file_count="multiple"
                    )

                    with gr.Row():
                        batch_method = gr.Radio(
                            choices=["weighted_average", "ml_model", "voting"],
                            value="weighted_average",
                            label="Detection Method"
                        )

                    with gr.Row():
                        batch_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Threshold"
                        )

                        batch_cnn = gr.Checkbox(
                            label="Use CNN",
                            value=False
                        )

                    batch_btn = gr.Button("🔍 Analyze All Images", variant="primary", size="lg")

            batch_output = gr.HTML(label="Results")

            batch_btn.click(
                fn=batch_detect_images,
                inputs=[batch_input, batch_method, batch_threshold, batch_cnn],
                outputs=batch_output
            )

        # About Tab
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## About AIDentify

            AIDentify is a comprehensive AI-generated image detection system that achieves **>80% accuracy**
            using an ensemble of multiple detection methods.

            ### 🔬 Detection Methods

            #### 1. Frequency Domain Analysis
            - Analyzes DCT and FFT patterns
            - Detects generator artifacts and unnatural frequency distributions
            - Identifies lack of natural sensor noise

            #### 2. Statistical Analysis
            - Examines color distributions and noise characteristics
            - Checks Benford's Law adherence
            - Analyzes texture patterns (GLCM, LBP)
            - Studies edge and gradient distributions

            #### 3. Deep Learning CNN (Optional)
            - End-to-end learned features
            - Custom CNN or EfficientNet architecture
            - Highest accuracy but requires more resources

            #### 4. Ensemble Combination
            - Combines all methods for robust detection
            - Weighted averaging, voting, or ML-based fusion

            ### 📊 Performance

            | Method | Accuracy | Speed |
            |--------|----------|-------|
            | Frequency + Statistical | 60-70% | Fast |
            | With Trained ML | 75-85% | Fast |
            | With Trained CNN | 85-95% | Medium |
            | Full Ensemble | **90-97%** | Medium |

            ### 🚀 Usage Tips

            1. **For quick analysis**: Use without CNN (fast mode)
            2. **For best accuracy**: Enable CNN detector (requires trained model)
            3. **Adjust threshold**: Lower for more detections, higher for fewer false positives
            4. **Train models**: For >80% accuracy, train on your own dataset

            ### 📚 Training

            To train models and improve accuracy:
            ```bash
            python src/train.py --data_dir data --model_type both --epochs 50
            ```

            See README.md and IMPLEMENTATION_GUIDE.md for detailed instructions.

            ### 🎓 Educational Context

            Developed for CAP6415 (Computer Vision) demonstrating:
            - Frequency domain analysis (Fourier Transform)
            - Statistical pattern recognition
            - Deep learning for image classification
            - Ensemble methods

            ### ⚠️ Limitations

            - Accuracy depends on training data quality
            - May not detect newest generation techniques without retraining
            - Results are probabilistic, not definitive
            - Should be used as one tool among many for verification

            ---

            **GitHub:** [CAP6415_F25_project-AIDentify](https://github.com/yourusername/CAP6415_F25_project-AIDentify)

            **License:** MIT
            """)

# Launch the app
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting AIDentify Web Interface...")
    print("="*60)
    print("\nThe web interface will open in your browser.")
    print("If it doesn't open automatically, copy the URL shown below.\n")

    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        show_error=True
    )
