"""Gradio web UI for crack classification inference (PyTorch)."""

import os

import gradio as gr
import torch

import config
from src.model import InceptionV3Classifier
from src.preprocessing import preprocess_for_inference
from src.device import get_device


def load_model():
    """Load the best trained model."""
    device = get_device()
    model = InceptionV3Classifier().to(device)

    model_path = os.path.join(config.OUTPUT_DIR, "inceptionv3", "models", "best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(config.MODEL_DIR, "best_model.pt")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Model loaded from: {model_path}")
    else:
        print(f"WARNING: No model found at {model_path}")

    model.eval()
    return model, device


MODEL, DEVICE = load_model()


def predict(image_path: str) -> tuple[dict, str]:
    """Run inference on a single image."""
    if image_path is None:
        return {}, "No image provided."

    tensor = preprocess_for_inference(image_path, size=config.IMG_SIZE, normalize="imagenet")
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        output = MODEL(tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    confidences = {
        config.CLASS_NAMES[i]: float(probabilities[i])
        for i in range(config.NUM_CLASSES)
    }

    top_class = max(confidences, key=confidences.get)
    top_conf = confidences[top_class]

    if top_conf < config.CONFIDENCE_THRESHOLD:
        status = (f"Low confidence: {top_class} ({top_conf:.1%}). "
                  f"Consider manual review.")
    else:
        status = f"Prediction: {top_class} ({top_conf:.1%})"

    return confidences, status


def create_app() -> gr.Blocks:
    """Create the Gradio interface."""
    with gr.Blocks(title="Crack Classifier") as app:
        gr.Markdown("# Concrete Crack Classifier")
        gr.Markdown("Upload an image of a concrete surface to classify crack type.")

        with gr.Row():
            image_input = gr.Image(type="filepath", label="Upload Image")

        classify_btn = gr.Button("Classify", variant="primary")

        with gr.Row():
            label_output = gr.Label(num_top_classes=6, label="Classification")
            status_output = gr.Textbox(label="Status", interactive=False)

        classify_btn.click(
            fn=predict,
            inputs=[image_input],
            outputs=[label_output, status_output],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
