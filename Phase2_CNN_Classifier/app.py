import streamlit as st
from PIL import Image
from model_utils import CLASS_NAMES, load_model, preprocess_image_from_array, make_gradcam_heatmap, overlay_gradcam, \
    log_inspection
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import os
import mysql.connector

st.set_page_config(page_title="Steel Defect Inspection", layout="wide")

st.cache_resource.clear()

CONFIDENCE_THRESHOLD = 85.0
LOW_CONFIDENCE_THRESHOLD = 50.0

st.title("🏭 Steel Surface Defect Inspection System")
st.write("Automated visual quality inspection using deep learning.")

st.sidebar.title("About")
st.sidebar.info(
    "MobileNetV2 transfer learning model trained on NEU Steel Surface Defect Dataset.\n\n"
    "**Test Accuracy:** 98.89%\n\n"
    "**Cross-Validation:** 99.56% ± 0.38%"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Details:**")
st.sidebar.write("Architecture: MobileNetV2 Transfer Learning")
st.sidebar.write("Input: 224×224×3")
st.sidebar.write("Classes: 6")

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


def get_top_k_predictions(predictions, k=3):
    top_indices = np.argsort(predictions)[::-1][:k]
    return [(CLASS_NAMES[i], predictions[i] * 100) for i in top_indices]


def get_decision(confidence):
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return "⚫ Low-Confidence Prediction", "⚫"
    elif confidence >= 95:
        return "✅ Accept", "🟢"
    elif confidence >= CONFIDENCE_THRESHOLD:
        return "🟡 Review", "🟡"
    else:
        return "🔴 Reject", "🔴"


model = get_model()

st.sidebar.header("Inspection Details")
batch_id = st.sidebar.text_input("Batch ID", value="B00001")
line_number = st.sidebar.selectbox("Production Line", ["Line 1", "Line 2", "Line 3"])
inspector_id = st.sidebar.text_input("Inspector ID", value="AUTO_INSPECTION")

inspection_mode = st.radio("Select Mode:", ["Single Image Inspection", "Batch Inspection"], index=None)

if inspection_mode is None:
    st.info("👆 Please select a mode to begin.")
    st.stop()

if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'current_filename' not in st.session_state:
    st.session_state.current_filename = None


def process_single_image(image, filename, batch_id, line_number, inspector_id):
    inspection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption=f"Captured Image | Batch: {batch_id}", width=350)

    predictions = predict_with_all_probs(image, model)
    predicted_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = predictions[predicted_idx] * 100
    decision, icon = get_decision(confidence)
    top_3 = get_top_k_predictions(predictions, k=3)

    with col2:
        st.write(f"**Inspection Time:** {inspection_time}")
        st.write(f"**Production Line:** {line_number}")
        st.write(f"**Inspector:** {inspector_id}")
        st.write(f"**Batch ID:** {batch_id}")
        st.write("---")
        st.write(f"### {icon} Quality Decision: {decision}")

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            st.warning(
                f"### ⚫ Low confidence across all defect classes ({confidence:.2f}% top match)\n\n"
                f"The model is not confident this image matches any of the 6 trained defect "
                f"types well. This may be a genuinely ambiguous defect case, or an image "
                f"outside the model's training distribution — this threshold alone cannot "
                f"distinguish between the two. Route to manual inspection."
            )
        elif confidence < CONFIDENCE_THRESHOLD:
            st.error("### ⚠️ Manual Inspection Required")
            st.write(
                f"This prediction has **{confidence:.2f}%** confidence, "
                f"below the acceptance threshold of **{CONFIDENCE_THRESHOLD}%**.\n\n"
                f"**Recommended action:** Route to human inspector for verification."
            )
        else:
            st.success(f"### ✅ Automated Inspection Passed")
            st.write(f"Prediction confidence exceeds acceptance threshold.")

        if confidence >= LOW_CONFIDENCE_THRESHOLD:
            st.metric("Confidence", f"{confidence:.2f}%")

        st.write("**Top 3 Predictions:**")
        for cls, prob in top_3:
            st.write(f"- {cls}: {prob:.2f}%")

    with st.expander("📊 Class Probabilities"):
        probs = {CLASS_NAMES[i]: float(predictions[i]) * 100 for i in range(len(CLASS_NAMES))}
        for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            st.write(f"**{class_name}:** {prob:.2f}%")
            st.progress(int(prob) / 100)

    with st.expander("🔧 Preprocessing Details"):
        st.write(
            "1. Resized to 224×224 pixels (nearest-neighbor interpolation)\n"
            "2. Converted to RGB (3 channels)\n"
            "3. Normalized with MobileNetV2 ImageNet preprocessing\n"
            "4. Expanded to batch dimension"
        )

    with st.expander("🔥 Grad-CAM: What the Model Looked At"):
        img_array_for_cam = preprocess_image_from_array(image)
        img_array_for_cam = tf.expand_dims(img_array_for_cam, 0)
        heatmap = make_gradcam_heatmap(img_array_for_cam, model)
        gradcam_image = overlay_gradcam(image, heatmap)
        st.image(gradcam_image, caption="Red/yellow regions influenced the prediction most", width=350)

    log_inspection(
        batch_id=batch_id,
        image_file=filename,
        actual_defect="unknown",
        predicted_defect=predicted_class,
        confidence=confidence,
        decision=decision,
        line_number=line_number
    )

    st.toast(f"✅ Inspection logged: {predicted_class} ({confidence:.2f}%)")

    inspection_no = len(st.session_state.history) + 1
    st.session_state.history.append({
        'No.': inspection_no,
        'Time': inspection_time,
        'File': filename,
        'Defect': predicted_class,
        'Confidence': f"{confidence:.2f}%",
        'Decision': decision,
        'Batch': batch_id,
        'Line': line_number
    })


if inspection_mode == "Single Image Inspection":
    source_choice = st.radio("Choose image source:", ["Upload My Own Image", "Use NEU Sample Image"], index=None)

    if source_choice is None:
        st.info("👆 Please select an image source.")
        st.stop()

    if source_choice == "Upload My Own Image":
        uploaded_file = st.file_uploader("Upload any steel defect image", type=['jpg', 'jpeg', 'png', 'bmp'])
        if uploaded_file is not None:
            st.image(Image.open(uploaded_file), width=250)
            if st.button("🚀 Run Inspection", type="primary"):
                image = Image.open(uploaded_file)
                process_single_image(image, uploaded_file.name, batch_id, line_number, inspector_id)

    else:
        VALID_PATH_SAMPLE = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "raw", "NEU-DET", "validation", "images"
        )

        col1, col2 = st.columns(2)
        with col1:
            sample_class = st.selectbox("Select defect class:", CLASS_NAMES)
        with col2:
            if os.path.exists(VALID_PATH_SAMPLE):
                class_path = os.path.join(VALID_PATH_SAMPLE, sample_class)
                all_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
                sample_file = st.selectbox("Select image:", all_files)

        if st.button("🚀 Run Inspection", type="primary"):
            sample_path = os.path.join(class_path, sample_file)
            image = Image.open(sample_path)
            process_single_image(image, sample_file, batch_id, line_number, inspector_id)

else:
    source_choice = st.radio("Choose image source:", ["Upload My Own Images", "Use NEU Dataset Images"], index=None)

    if source_choice is None:
        st.info("👆 Please select an image source.")
        st.stop()

    if source_choice == "Upload My Own Images":
        uploaded_batch_files = st.file_uploader(
            "Upload multiple steel defect images",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            key="batch_upload"
        )

        if uploaded_batch_files:
            if st.button("🚀 Run Inspection on Uploaded Images", type="primary"):
                st.success(f"Processing {len(uploaded_batch_files)} uploaded images...")

                results = []
                inspection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                progress_bar = st.progress(0)

                for idx, uploaded_file in enumerate(uploaded_batch_files):
                    image = Image.open(uploaded_file)
                    predictions = predict_with_all_probs(image, model)
                    predicted_idx = np.argmax(predictions)
                    confidence = predictions[predicted_idx] * 100
                    predicted_class = CLASS_NAMES[predicted_idx]

                    decision, icon = get_decision(confidence)
                    top_3 = get_top_k_predictions(predictions, k=3)
                    top_3_str = ", ".join(f"{cls} ({prob:.1f}%)" for cls, prob in top_3)

                    inspection_no = len(st.session_state.history) + 1

                    result = {
                        'No.': inspection_no,
                        'Time': inspection_time,
                        'File': uploaded_file.name,
                        'Actual': "unknown",
                        'Predicted': predicted_class,
                        'Confidence': f"{confidence:.2f}%",
                        'Top 3': top_3_str,
                        'Decision': decision,
                        'Batch': batch_id,
                        'Line': line_number
                    }

                    results.append(result)
                    st.session_state.history.append(result)

                    log_inspection(
                        batch_id=batch_id,
                        image_file=uploaded_file.name,
                        actual_defect="unknown",
                        predicted_defect=predicted_class,
                        confidence=confidence,
                        decision=decision,
                        line_number=line_number
                    )

                    progress_bar.progress((idx + 1) / len(uploaded_batch_files))

                df = pd.DataFrame(results)
                st.write("### Uploaded Images Inspection Results")
                st.dataframe(df, use_container_width=True)

                accept_count = sum(1 for r in results if "Accept" in r['Decision'])
                review_count = sum(1 for r in results if "Review" in r['Decision'])
                reject_count = sum(1 for r in results if "🔴 Reject" in r['Decision'])
                low_conf_count = sum(1 for r in results if "Low-Confidence" in r['Decision'])

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("✅ Accept", accept_count)
                col2.metric("🟡 Review", review_count)
                col3.metric("🔴 Reject", reject_count)
                col4.metric("⚫ Low-Conf", low_conf_count)

                avg_conf = df['Confidence'].str.rstrip('%').astype(float).mean()
                st.metric("📊 Avg Confidence", f"{avg_conf:.2f}%")

                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Export Results as CSV",
                    data=csv,
                    file_name=f"uploaded_inspection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

                st.toast(f"✅ Batch complete: {len(results)} images processed")

    else:
        VALID_PATH = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "raw", "NEU-DET", "validation", "images"
        )

        select_all_classes = st.checkbox("✅ Select ALL defect classes")

        if select_all_classes:
            selected_classes = CLASS_NAMES.copy()
        else:
            selected_classes = st.multiselect(
                "Select defect classes:",
                CLASS_NAMES,
                default=['crazing']
            )

        max_images_to_show = st.number_input(
            "Max images to display per class:",
            min_value=1,
            max_value=60,
            value=30
        )

        selected_images = []

        if selected_classes:
            for class_name in selected_classes:
                class_path = os.path.join(VALID_PATH, class_name)
                if os.path.isdir(class_path):
                    st.write(f"### {class_name}")
                    all_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]

                    st.write(
                        f"**{len(all_files)} images available, showing first {min(len(all_files), max_images_to_show)}**")

                    select_all_images = st.checkbox(f"✅ Select ALL images in {class_name}",
                                                    key=f"select_all_{class_name}")

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

        if st.button("🚀 Run Inspection on NEU Images", type="primary"):
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
                    top_3 = get_top_k_predictions(predictions, k=3)
                    top_3_str = ", ".join(f"{cls} ({prob:.1f}%)" for cls, prob in top_3)

                    inspection_no = len(st.session_state.history) + 1

                    result = {
                        'No.': inspection_no,
                        'Time': inspection_time,
                        'File': img_info['file'],
                        'Actual': img_info['actual_class'],
                        'Predicted': predicted_class,
                        'Confidence': f"{confidence:.2f}%",
                        'Top 3': top_3_str,
                        'Decision': decision,
                        'Batch': batch_id,
                        'Line': line_number
                    }

                    results.append(result)
                    st.session_state.history.append(result)

                    log_inspection(
                        batch_id=batch_id,
                        image_file=img_info['file'],
                        actual_defect=img_info['actual_class'],
                        predicted_defect=predicted_class,
                        confidence=confidence,
                        decision=decision,
                        line_number=line_number
                    )

                    progress_bar.progress((idx + 1) / len(selected_images))

                df = pd.DataFrame(results)

                st.write(f"### Inspection Results | Batch: {batch_id} | Line: {line_number}")
                st.dataframe(df, use_container_width=True)

                accept_count = sum(1 for r in results if "Accept" in r['Decision'])
                review_count = sum(1 for r in results if "Review" in r['Decision'])
                reject_count = sum(1 for r in results if "🔴 Reject" in r['Decision'])
                low_conf_count = sum(1 for r in results if "Low-Confidence" in r['Decision'])

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("✅ Accept", accept_count)
                col2.metric("🟡 Review", review_count)
                col3.metric("🔴 Reject", reject_count)
                col4.metric("⚫ Low-Conf", low_conf_count)

                avg_conf = df['Confidence'].str.rstrip('%').astype(float).mean()
                st.metric("📊 Avg Confidence", f"{avg_conf:.2f}%")

                st.write("---")
                if reject_count > 0:
                    st.error(f"🚨 **Line Status: CRITICAL** — {reject_count} images rejected")
                elif low_conf_count > 0:
                    st.warning(
                        f"⚠️ **Line Status: NEEDS ATTENTION** — {low_conf_count} images returned low-confidence predictions")
                elif review_count > 0:
                    st.warning(f"⚠️ **Line Status: ATTENTION NEEDED** — {review_count} images flagged for review")
                else:
                    st.success("✅ **Line Status: NORMAL** — all inspections passed")

                flagged_df = df[df['Decision'] != '✅ Accept']
                st.write("### 🚨 Flagged for Manual Review")
                if len(flagged_df) > 0:
                    st.dataframe(flagged_df, use_container_width=True)
                    st.warning(f"⚠️ {len(flagged_df)} images require human inspection")
                else:
                    st.success("✅ All images passed automated inspection")

                st.write("### 📊 Predicted Defect Distribution")
                defect_counts = df['Predicted'].value_counts()
                st.bar_chart(defect_counts)

                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Export Results as CSV",
                    data=csv,
                    file_name=f"inspection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

                st.toast(f"✅ Batch complete: {len(results)} images processed")

with st.expander("📜 Inspection History (Session)"):
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        st.write(f"**Total Inspections:** {len(history_df)}")
    else:
        st.write("No inspections logged yet.")

with st.expander("🗄️ Inspection Logs (Database)"):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="steel_defects"
        )
        query = "SELECT * FROM inspection_log ORDER BY inspection_id DESC LIMIT 20"
        db_df = pd.read_sql(query, conn)
        conn.close()
        st.dataframe(db_df, use_container_width=True)
        st.write(f"**Total logged inspections:** {len(db_df)}")
    except Exception as e:
        st.write(f"Database error: {e}")

if st.button("🗑️ Clear Session History"):
    st.session_state.history = []
    st.rerun()

st.markdown("---")
st.markdown("*Industrial Quality Inspection Demo — Built with TensorFlow, Keras, and Streamlit*")