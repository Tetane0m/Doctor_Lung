import logging
import io
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model



# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


class CancerPredictor:
    def __init__(self):
        # Load models
        model = tf.keras.applications.ResNet50(weights='imagenet')
        self.resnet = Model(inputs=model.input, outputs=model.output)
        self.lstm_model = tf.keras.models.load_model('lstm_model_98.h5')
        self.class_names = ["Benign", "Malignant", "Normal"]

    async def predict_image(self, image_file):
        """Process image and make prediction"""
        try:
            # ----------------------------------------

            image = Image.open(io.BytesIO(await image_file.download_as_bytearray()))

            # Resize the image to 224x224 for model input
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            resized_image_cv = cv2.resize(image_cv, (224, 224), interpolation=cv2.INTER_LINEAR)

            # Convert image to array
            image_array = tf.keras.preprocessing.image.img_to_array(resized_image_cv)

            # Expand dimensions to match model input shape
            image_array = np.expand_dims(image_array, axis=0)  # Shape becomes (1, 224, 224, 3)
            image_array = preprocess_input(image_array)  # Apply ResNet preprocessing

            print("Image shape before ResNet:", image_array.shape)

            # Extract features using the model
            resnet_vector = self.resnet.predict(image_array)
            resnet_vector = resnet_vector.reshape(1, 1, resnet_vector.shape[1])

            # ----------------------------------------

            # Make prediction
            prediction = self.lstm_model.predict(resnet_vector)
            class_idx = np.argmax(prediction)
            confidence = np.max(prediction)

            prediction = predictor.lstm_model.predict(resnet_vector)
            print("Raw Prediction Output:", prediction)

            return self.class_names[class_idx], confidence
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return None, None


# Initialize predictor
predictor = CancerPredictor()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message"""
    await update.message.reply_text(
        "🩺 Welcome to Lung Cancer Screening Bot!\n\n"
        "Send a chest CT scan image (PNG/JPG) for analysis.\n\n"
        "⚠️ Disclaimer: This is an AI screening tool and should NOT replace professional medical diagnosis."
    )


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process user-sent image"""
    user = update.message.from_user
    logging.info(f"Image received from {user.first_name}")

    # Get highest resolution photo
    photo_file = await update.message.photo[-1].get_file()

    # Show processing status
    await update.message.reply_chat_action(action="typing")

    # Make prediction
    class_name, confidence = await predictor.predict_image(photo_file)

    if not class_name:
        await update.message.reply_text("❌ Error processing image. Please try again.")
        return

    # Generate advice
    advice = generate_medical_advice(class_name, confidence)

    # Format response
    response = (
        f"🔍 Prediction Result: {class_name}\n"
        f"🟢 Confidence: {confidence:.2%}\n\n"
        f"📌 Medical Advice:\n{advice}"
    )

    await update.message.reply_text(response)


def generate_medical_advice(class_name, confidence):
    """Generate appropriate medical recommendations"""
    advice = {
        "Benign": [
            "No signs of malignancy detected",
            "Recommend routine monitoring (6-12 month follow-up)",
            "Consult a pulmonologist for confirmation"
        ],
        "Malignant": [
            "Potential malignancy detected",
            "Urgently consult an oncologist",
            "Recommended tests: Biopsy, PET-CT scan",
            "Do not delay medical evaluation"
        ],
        "Normal": [
            "Healthy lung tissue observed",
            "Maintain regular health checkups",
            "Practice preventive care"
        ]
    }[class_name]

    # Add confidence note
    if class_name == "Malignant" and confidence < 0.8:
        advice.append("\n⚠️ Note: Moderate confidence prediction - additional tests strongly recommended")

    return "\n• ".join(advice)


def main():
    """Run the bot"""
    # Replace with your bot token
    TOKEN = "7884638558:AAF-qw8pk64toLRUTKeq5iOeKDEwgt9ZC54"

    # Create Application
    application = Application.builder().token(TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    # Run bot
    application.run_polling()


if __name__ == "__main__":
    main()