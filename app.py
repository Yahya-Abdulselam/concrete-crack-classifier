"""Gradio web UI for crack classification inference."""

import os
import numpy as np
import gradio as gr
from tensorflow import keras

import config
from src.preprocessing import preprocess_for_inference

# Load model once at startup
MODEL_PATH = os.path.join(config.MODEL_DIR, "best_model.keras")
model = None


def load_model():
    """Load the trained model."""
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run the training notebook first."
        )
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")


def predict(image_path: str) -> tuple[dict, str]:
    """Run inference on a single image.

    Args:
        image_path: Path to the uploaded image.

    Returns:
        Tuple of (class_confidences_dict, status_message).
    """
    if model is None:
        return {}, "Model not loaded. Please check the model file."

    # Preprocess
    img = preprocess_for_inference(image_path)

    # Predict
    predictions = model.predict(img, verbose=0)[0]

    # Build confidence dict for Gradio Label output
    confidences = {
        config.CLASS_NAMES[i]: float(predictions[i])
        for i in range(config.NUM_CLASSES)
    }

    # Determine top prediction
    top_idx = np.argmax(predictions)
    top_class = config.CLASS_NAMES[top_idx]
    top_conf = predictions[top_idx]

    # Status message
    if top_conf < config.CONFIDENCE_THRESHOLD:
        status = (
            f"**{top_class}** ({top_conf:.1%})\n\n"
            f"Warning: Low confidence prediction (below {config.CONFIDENCE_THRESHOLD:.0%}). "
            f"Consider manual inspection."
        )
    else:
        status = f"**{top_class}** ({top_conf:.1%})"

    return confidences, status


def create_app() -> gr.Blocks:
    """Create the Gradio interface."""
    with gr.Blocks(
        title="Crack Classification System",
        theme=gr.themes.Soft(),
    ) as app:
        class_display = " | ".join(
            name.replace("_", " ").title() for name in config.CLASS_NAMES
        )
        gr.Markdown(
            "# Crack Classification System\n"
            "Upload a concrete image to classify the type of crack.\n\n"
            f"**Classes:** {class_display}"
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="filepath",
                    label="Upload Image",
                )
                classify_btn = gr.Button("Classify", variant="primary")

            with gr.Column(scale=1):
                label_output = gr.Label(
                    num_top_classes=config.NUM_CLASSES,
                    label="Prediction Confidence",
                )
                status_output = gr.Markdown(label="Result")

        classify_btn.click(
            fn=predict,
            inputs=image_input,
            outputs=[label_output, status_output],
        )

        gr.Markdown(
            "---\n"
            "*UREP 32-0210-250078 | Qatar University*"
        )

    return app


if __name__ == "__main__":
    load_model()
    app = create_app()
    app.launch(share=False)
