import tempfile
import cv2 as cv
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from sklearn.cluster import KMeans

CONST_KMEANS = 'Kmeans'

SELECT_CLUSTER_NO = 10
SELECT_RANDOM_STATE = 41
SELECT_STEP_SIZE = 10
SELECT_MODEL = CONST_KMEANS


def main():
    global SELECT_CLUSTER_NO, SELECT_RANDOM_STATE, SELECT_STEP_SIZE, SELECT_MODEL
    global CONST_KMEANS

    st.set_page_config(
        page_title="Media pixels clustering",
        page_icon="🌄",
        layout='wide',
    )

    st.title("🌄 Media Pixels Clustering")
    st.write("""* This Streamlit app lets users upload images or videos and apply KMeans clustering to reduce the number of colors.

* Users can configure the number of clusters, step size (sampling), and random state.

* For images, it processes and displays both the original and clustered versions side by side.

* For videos, it processes each frame with KMeans and outputs a new clustered video.""")

    st.write("""Made with ❤ by Youssef Samy Youssef""")

    st.divider()

    st.subheader("Model Config")

    main_model_cols = st.columns(2)

    SELECT_MODEL = main_model_cols[0].selectbox(
        "Select Model",
        (CONST_KMEANS,),
    )

    SELECT_STEP_SIZE = main_model_cols[1].number_input("Enter Step Size:", max_value=300, min_value=1,
                                                       value=SELECT_STEP_SIZE)

    if SELECT_MODEL == CONST_KMEANS:
        st.write('Kmeans Config')
        kmeans_cols = st.columns(2)

        SELECT_CLUSTER_NO = kmeans_cols[0].number_input("Enter clusters number:", max_value=100, min_value=1,
                                                        value=SELECT_CLUSTER_NO,
                                                        disabled=SELECT_MODEL != CONST_KMEANS)
        SELECT_RANDOM_STATE = kmeans_cols[1].number_input("Random State:", max_value=100, min_value=1,
                                                          value=SELECT_RANDOM_STATE,
                                                          disabled=SELECT_MODEL != CONST_KMEANS)

    st.divider()

    st.subheader("Select Media")
    media_tabs = st.tabs(['Images', 'Video'])

    with media_tabs[0]:
        build_image_tab()
    with media_tabs[1]:
        build_video_tab()


def build_image_tab():
    uploaded_file = st.file_uploader("Pick an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")  # ensures it's 3 channels
        image = ImageOps.exif_transpose(image)  # Fix orientation

        image = np.array(image)

        convert_btn = st.button("Convert image")
        image_cols = st.columns(2)

        image_cols[0].image(image, width=400)

        if convert_btn:
            with st.spinner("Converting"):
                image_tuned = apply_image_tuning(image)
            image_cols[1].image(image_tuned, width=400)


def build_video_tab():
    uploaded_file = st.file_uploader("Pick a video", type=["mp4", "wav"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        temp_path = tfile.name
        tfile.close()

        convert_btn = st.button("Convert video")

        loading_bar = st.progress(0, text="...")
        loading_bar.empty()

        videos_cols = st.columns(2)

        videos_cols[0].video(uploaded_file)

        if convert_btn:
            # Use a temporary file for the output video
            tmp_outfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            output_video_path = tmp_outfile.name
            tmp_outfile.close()

            cap = cv.VideoCapture(temp_path)
            if not cap.isOpened():
                raise IOError("Cannot open video file")

            frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv.CAP_PROP_FPS)
            total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

            fourcc = cv.VideoWriter_fourcc(*'avc1')
            out = cv.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            frame_counter = 0

            while True:
                loading_bar.progress(100 * frame_counter // total_frames)
                frame_counter += 1

                ret, frame = cap.read()
                if not ret:
                    break
                processed = apply_image_tuning(frame)

                if len(processed.shape) == 2:
                    processed = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)

                out.write(processed)

            cap.release()
            out.release()
            loading_bar.empty()

            videos_cols[1].video(output_video_path)


def model_apply_kmeans(image_scaled_pixels):
    global SELECT_CLUSTER_NO, SELECT_STEP_SIZE, SELECT_RANDOM_STATE

    kmeans_img = KMeans(
        n_clusters=SELECT_CLUSTER_NO,
        random_state=SELECT_RANDOM_STATE,
    )

    kmeans_img.fit(image_scaled_pixels[::SELECT_STEP_SIZE])
    img_pixels_clusters = kmeans_img.predict(image_scaled_pixels)
    img_pixels_tuned = kmeans_img.cluster_centers_[img_pixels_clusters, :]

    return img_pixels_tuned


def apply_image_tuning(image):
    image_scaled = image

    PIXELS_LIST_SHAPE = (image_scaled.shape[0] * image_scaled.shape[1], image_scaled.shape[2])

    image_scaled_pixels = image_scaled.reshape(PIXELS_LIST_SHAPE)

    image_pixels_tuned = model_apply_kmeans(image_scaled_pixels)

    image_tuned = image_pixels_tuned.reshape(image_scaled.shape)

    return image_tuned.astype(image.dtype)


if __name__ == "__main__":
    main()
