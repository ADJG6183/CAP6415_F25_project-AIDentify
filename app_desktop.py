"""
Tkinter Desktop GUI for AI-Generated Image Detection.

A simple desktop application for detecting AI-generated images.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import sys
import os
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.detectors.ensemble_detector import EnsembleDetector
from src.utils.image_processing import load_image


class AIDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AIDentify - AI Image Detection")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        # Initialize detector as None (lazy loading)
        self.detector = None
        self.current_image = None
        self.current_image_path = None

        # Setup GUI
        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="🔍 AIDentify - AI-Generated Image Detection",
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Left Panel - Image Display
        left_frame = ttk.LabelFrame(main_frame, text="Image", padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Image canvas
        self.image_canvas = tk.Canvas(left_frame, width=400, height=400, bg='gray')
        self.image_canvas.pack(fill=tk.BOTH, expand=True)

        # Buttons frame
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=10)

        self.load_btn = ttk.Button(button_frame, text="📁 Load Image", command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.detect_btn = ttk.Button(
            button_frame,
            text="🔍 Detect",
            command=self.detect_image,
            state=tk.DISABLED
        )
        self.detect_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ttk.Button(button_frame, text="🗑️ Clear", command=self.clear_image)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Right Panel - Settings and Results
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Settings
        settings_frame = ttk.LabelFrame(right_frame, text="⚙️ Detection Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=5)

        # Detection method
        ttk.Label(settings_frame, text="Detection Method:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.method_var = tk.StringVar(value="weighted_average")
        method_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.method_var,
            values=["weighted_average", "ml_model", "voting"],
            state="readonly"
        )
        method_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)

        # Threshold
        ttk.Label(settings_frame, text="Threshold:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.threshold_var = tk.DoubleVar(value=0.5)
        threshold_scale = ttk.Scale(
            settings_frame,
            from_=0.0,
            to=1.0,
            variable=self.threshold_var,
            orient=tk.HORIZONTAL
        )
        threshold_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)

        self.threshold_label = ttk.Label(settings_frame, text="0.50")
        self.threshold_label.grid(row=1, column=2, pady=5)
        threshold_scale.configure(command=self.update_threshold_label)

        # Use CNN
        self.use_cnn_var = tk.BooleanVar(value=False)
        cnn_check = ttk.Checkbutton(
            settings_frame,
            text="Use CNN Detector (slower, more accurate)",
            variable=self.use_cnn_var
        )
        cnn_check.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)

        settings_frame.columnconfigure(1, weight=1)

        # Results
        results_frame = ttk.LabelFrame(right_frame, text="📊 Detection Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Verdict
        self.verdict_label = ttk.Label(
            results_frame,
            text="No detection performed yet",
            font=('Arial', 14, 'bold'),
            foreground='gray'
        )
        self.verdict_label.pack(pady=10)

        # Details
        details_canvas = tk.Canvas(results_frame, height=300)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=details_canvas.yview)
        scrollable_frame = ttk.Frame(details_canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: details_canvas.configure(scrollregion=details_canvas.bbox("all"))
        )

        details_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        details_canvas.configure(yscrollcommand=scrollbar.set)

        self.details_text = tk.Text(
            scrollable_frame,
            height=15,
            width=40,
            wrap=tk.WORD,
            font=('Courier', 9)
        )
        self.details_text.pack(fill=tk.BOTH, expand=True)

        details_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))

    def update_threshold_label(self, value):
        """Update threshold label."""
        self.threshold_label.config(text=f"{float(value):.2f}")

    def load_image(self):
        """Load an image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                # Load and display image
                self.current_image_path = file_path
                img = Image.open(file_path)

                # Resize for display
                display_size = (400, 400)
                img.thumbnail(display_size, Image.Resampling.LANCZOS)

                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(img)

                # Display on canvas
                self.image_canvas.delete("all")
                self.image_canvas.create_image(
                    200, 200,
                    image=photo,
                    anchor=tk.CENTER
                )
                self.image_canvas.image = photo  # Keep a reference

                # Load actual image for detection
                self.current_image = load_image(file_path)

                # Enable detect button
                self.detect_btn.config(state=tk.NORMAL)

                self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
                self.verdict_label.config(text="Image loaded. Click 'Detect' to analyze.", foreground='gray')
                self.details_text.delete('1.0', tk.END)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
                self.status_var.set("Error loading image")

    def clear_image(self):
        """Clear the current image."""
        self.image_canvas.delete("all")
        self.current_image = None
        self.current_image_path = None
        self.detect_btn.config(state=tk.DISABLED)
        self.verdict_label.config(text="No detection performed yet", foreground='gray')
        self.details_text.delete('1.0', tk.END)
        self.status_var.set("Ready")

    def initialize_detector(self):
        """Initialize detector if not already initialized."""
        if self.detector is None:
            self.status_var.set("Initializing detector...")
            self.root.update()

            # Check for trained models
            ml_model_path = 'trained_models/ml_model_random_forest.pkl'
            cnn_model_path = 'trained_models/cnn_model_custom.pth'

            ml_path = ml_model_path if os.path.exists(ml_model_path) else None
            cnn_path = cnn_model_path if os.path.exists(cnn_model_path) else None

            self.detector = EnsembleDetector(
                ml_model_path=ml_path,
                cnn_model_path=cnn_path,
                use_cnn=self.use_cnn_var.get()
            )

            self.status_var.set("Detector initialized")

    def detect_image(self):
        """Detect if current image is AI-generated."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first!")
            return

        # Run detection in separate thread to avoid freezing GUI
        thread = threading.Thread(target=self._run_detection, daemon=True)
        thread.start()

    def _run_detection(self):
        """Run detection (called in separate thread)."""
        try:
            # Update UI - disable buttons
            self.root.after(0, lambda: self.detect_btn.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.load_btn.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.progress.start(10))
            self.root.after(0, lambda: self.status_var.set("Analyzing image..."))

            # Initialize detector if needed
            self.initialize_detector()

            # Perform detection
            result = self.detector.predict(
                self.current_image,
                threshold=self.threshold_var.get(),
                method=self.method_var.get()
            )

            # Update UI with results
            self.root.after(0, lambda: self._display_results(result))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Detection failed:\n{str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Detection failed"))

        finally:
            # Re-enable buttons
            self.root.after(0, lambda: self.detect_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.load_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.progress.stop())

    def _display_results(self, result):
        """Display detection results."""
        # Update verdict
        if result['is_ai_generated']:
            verdict_text = "🤖 AI-GENERATED IMAGE"
            color = 'red'
        else:
            verdict_text = "📷 REAL IMAGE"
            color = 'green'

        self.verdict_label.config(text=verdict_text, foreground=color)

        # Update details
        details = f"""
═══════════════════════════════════════
        DETECTION RESULTS
═══════════════════════════════════════

Verdict: {verdict_text}

Probability (AI-generated): {result['probability']:.1%}
Confidence: {result['confidence']:.1%}

Method: {result['method']}
Threshold: {result['threshold']}

───────────────────────────────────────
Individual Detector Scores:
───────────────────────────────────────
"""

        for detector_name, prob in result['individual_predictions'].items():
            if prob is not None:
                status = "✓ AI" if prob > 0.5 else "✓ Real"
                details += f"\n{detector_name.capitalize():15s}: {prob:6.1%}  {status}"

        details += "\n\n═══════════════════════════════════════\n"

        self.details_text.delete('1.0', tk.END)
        self.details_text.insert('1.0', details)

        self.status_var.set("Detection complete")


def main():
    """Run the desktop GUI application."""
    root = tk.Tk()
    app = AIDetectorGUI(root)

    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    print("\n" + "="*60)
    print("AIDentify Desktop Application Started")
    print("="*60)
    print("\nClose the window to exit.\n")

    root.mainloop()


if __name__ == '__main__':
    main()
