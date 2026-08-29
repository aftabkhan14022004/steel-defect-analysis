import streamlit as st
from PIL import Image
from model_utils import CLASS_NAMES, load_model, preprocess_image_from_array
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import os

st.set_page_config(page_title="Steel Defect Inspection", layout="wide")

# Clear any cached model from previous runs
st.cache_resource.clear()

CONFIDENCE_THRESHOLD = 85.0

st.title("🏭 Steel Surface Defect Inspection System")
st.write("Automated visual quality inspection using deep learning.")

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "MobileNetV2 transfer learning model trained on NEU Steel Surface Defect Dataset.\n\n"
    "**Test Accuracy:** 98.89%\n\n"
    "**Cross-Validation:** 99.56% ± 0.38%"
)

st.sidebar.header("Defect Types")
defect_info = {
    "crazing": "Network of fine cracks",
    "inclusion": "Foreign material embedded",
    "patches": "Large irregular dark areas",
    "pitted_surface": "Small holes or pits",
    "rolled-in_scale": "Scale marks from rolling",
    "scratches": "Thin linear marks"
}
for defect, desc in defect_info.items():
    st.sidebar.markdown(f"**{defect}:** {desc}")


def get_model():
    return load_model()


def predict_with_all_probs(image, model):
    img_array = preprocess_image_from_array(image)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=0)[0]
    return predictions


def get_decision(confidence):
    if confidence >= 95:
        return "✅ Accept", "🟢"
    elif confidence >= CONFIDENCE_THRESHOLD:
        return "🟡 Review", "🟡"
    else:
        return "🔴 Reject", "🔴"


model = get_model()

# Inspection details
st.sidebar.header("Inspection Details")
batch_id = st.sidebar.text_input("Batch ID", value="B00001")
line_number = st.sidebar.selectbox("Production Line", ["Line 1", "Line 2", "Line 3"])
inspector_id = st.sidebar.text_input("Inspector ID", value="AUTO_INSPECTION")

# Mode selection
inspection_mode = st.radio("Select Mode:", ["Single Image Inspection", "Batch Inspection"])

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

if inspection_mode == "Single Image Inspection":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png', 'bmp'])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            inspection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption=f"Captured Image | Batch: {batch_id}", width=350)

            predictions = predict_with_all_probs(image, model)
            predicted_idx = np.argmax(predictions)
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = predictions[predicted_idx] * 100

            decision, icon = get_decision(confidence)

            with col2:
                st.write(f"**Inspection Time:** {inspection_time}")
                st.write(f"**Production Line:** {line_number}")
                st.write(f"**Inspector:** {inspector_id}")
                st.write(f"**Batch ID:** {batch_id}")
                st.write("---")
                st.write(f"### {icon} Quality Decision: {decision}")

                if confidence < CONFIDENCE_THRESHOLD:
                    st.error("### ⚠️ Manual Inspection Required")
                    st.write(
                        f"This prediction has **{confidence:.2f}%** confidence, "
                        f"below the acceptance threshold of **{CONFIDENCE_THRESHOLD}%**.\n\n"
                        f"**Recommended action:** Route to human inspector for verification."
                    )
                else:
                    st.success(f"### ✅ Automated Inspection Passed")
                    st.write(f"Prediction confidence exceeds acceptance threshold.")

                st.write(f"**Predicted Defect:** {predicted_class}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.progress(int(confidence) / 100)

            with st.expander("📊 Class Probabilities"):
                probs = {CLASS_NAMES[i]: float(predictions[i]) * 100 for i in range(len(CLASS_NAMES))}
                for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"**{class_name}:** {prob:.2f}%")
                    st.progress(int(prob) / 100)

            with st.expander("🔧 Preprocessing Details"):
                st.write(
                    "1. Resized to 224×224 pixels\n"
                    "2. Converted to RGB (3 channels)\n"
                    "3. Normalized with MobileNetV2 ImageNet preprocessing\n"
                    "4. Expanded to batch dimension"
                )

            # Log to history
            inspection_no = len(st.session_state.history) + 1

            st.session_state.history.append({
                'No.': inspection_no,
                'Time': inspection_time,
                'File': uploaded_file.name,
                'Defect': predicted_class,
                'Confidence': f"{confidence:.2f}%",
                'Decision': decision,
                'Batch': batch_id,
                'Line': line_number
            })

        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.write("### Batch Inspection Setup")
    st.write("Select defect classes, then choose individual images for inspection.")

    # Relative path to validation images
    VALID_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "raw", "NEU-DET", "validation", "images"
    )

    # Select all classes option
    select_all_classes = st.checkbox("✅ Select ALL defect classes")

    if select_all_classes:
        selected_classes = CLASS_NAMES.copy()
    else:
        selected_classes = st.multiselect(
            "Select defect classes:",
            CLASS_NAMES,
            default=['crazing']
        )

    # Number of images to show per class
    max_images_to_show = st.number_input(
        "Max images to display per class:",
        min_value=1,
        max_value=60,
        value=30
    )

    # Show images with checkboxes
    selected_images = []

    if selected_classes:
        for class_name in selected_classes:
            class_path = os.path.join(VALID_PATH, class_name)
            if os.path.isdir(class_path):
                st.write(f"### {class_name}")
                all_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]

                st.write(
                    f"**{len(all_files)} images available, showing first {min(len(all_files), max_images_to_show)}**"
                )

                # Select all images for this class
                select_all_images = st.checkbox(
                    f"✅ Select ALL images in {class_name}", key=f"select_all_{class_name}"
                )

                display_files = all_files[:max_images_to_show]

                cols = st.columns(5)
                for idx, f in enumerate(display_files):
                    with cols[idx % 5]:
                        img_path = os.path.join(class_path, f)
                        img = Image.open(img_path)
                        st.image(img, width=80, caption=f.split('.')[0])

                        if select_all_images:
                            selected_images.append({
                                'file': f,
                                'path': img_path,
                                'actual_class': class_name
                            })
                            st.checkbox("Selected", value=True, key=f"auto_{class_name}_{f}", disabled=True)
                        else:
                            if st.checkbox("Select", key=f"{class_name}_{f}"):
                                selected_images.append({
                                    'file': f,
                                    'path': img_path,
                                    'actual_class': class_name
                                })

    if st.button("🚀 Run Inspection", type="primary"):
        if not selected_images:
            st.warning("Please select at least one image.")
        else:
            st.success(f"Processing {len(selected_images)} selected images...")

            results = []
            inspection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            progress_bar = st.progress(0)

            for idx, img_info in enumerate(selected_images):
                image = Image.open(img_info['path'])
                predictions = predict_with_all_probs(image, model)
                predicted_idx = np.argmax(predictions)
                confidence = predictions[predicted_idx] * 100
                predicted_class = CLASS_NAMES[predicted_idx]

                decision, icon = get_decision(confidence)

                inspection_no = len(st.session_state.history) + 1

                result = {
                    'No.': inspection_no,
                    'Time': inspection_time,
                    'File': img_info['file'],
                    'Actual': img_info['actual_class'],
                    'Predicted': predicted_class,
                    'Confidence': f"{confidence:.2f}%",
                    'Decision': decision,
                    'Batch': batch_id,
                    'Line': line_number
                }

                results.append(result)
                st.session_state.history.append(result)

                progress_bar.progress((idx + 1) / len(selected_images))

            df = pd.DataFrame(results)

            st.write(f"### Inspection Results | Batch: {batch_id} | Line: {line_number}")
            st.dataframe(df, use_container_width=True)

            # Summary
            accept_count = sum(1 for r in results if "Accept" in r['Decision'])
            review_count = sum(1 for r in results if "Review" in r['Decision'])
            reject_count = sum(1 for r in results if "Reject" in r['Decision'])

            col1, col2, col3 = st.columns(3)
            col1.metric("✅ Accept", accept_count)
            col2.metric("🟡 Review", review_count)
            col3.metric("🔴 Reject", reject_count)

            # Line status
            st.write("---")
            if reject_count > 0:
                st.error(f"🚨 **Line Status: CRITICAL** — {reject_count} images rejected")
            elif review_count > 0:
                st.warning(f"⚠️ **Line Status: ATTENTION NEEDED** — {review_count} images flagged for review")
            else:
                st.success("✅ **Line Status: NORMAL** — all inspections passed")

            # Flagged images
            flagged_df = df[df['Decision'] != '✅ Accept']
            st.write("### 🚨 Flagged for Manual Review")
            if len(flagged_df) > 0:
                st.dataframe(flagged_df, use_container_width=True)
                st.warning(f"⚠️ {len(flagged_df)} images require human inspection")
            else:
                st.success("✅ All images passed automated inspection")

            # Defect distribution
            st.write("### 📊 Predicted Defect Distribution")
            defect_counts = df['Predicted'].value_counts()
            st.bar_chart(defect_counts)

            # Export CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Export Results as CSV",
                data=csv,
                file_name=f"inspection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Inspection History
with st.expander("📜 Inspection History (Session)"):
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        st.write(f"**Total Inspections:** {len(history_df)}")
    else:
        st.write("No inspections logged yet.")

st.markdown("---")
st.markdown("*Industrial Quality Inspection Demo — Built with TensorFlow, Keras, and Streamlit*")