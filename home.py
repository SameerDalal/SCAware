import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import wfdb
import tensorflow as tf
import argparse

model_segment_length = 10000
step_size = 5000
num_channels = 2


def process_ecg(file_path):

    model = tf.keras.models.load_model('./models/model_3/keras/scd_model.keras')

    record = wfdb.rdrecord(file_path)

    ecg_signal = record.p_signal
    ecg_signal = ecg_signal.astype(np.float32)
    ecg_signal = ecg_signal[:, :num_channels]

    frequency = record.fs

    st.set_page_config(initial_sidebar_state="collapsed")
    st.title("SCA Prediction")
    graph = go.Figure()
    graph_placeholder = st.empty()
    col1, col2, col3 = st.columns(3)

    with col1:
        prediction_placeholder = st.empty()
        prediction_placeholder.markdown("**Making Prediction ...**")
    with col2:
        confidence_placeholder = st.empty()
        confidence_placeholder.markdown("**Calculating Confidence ...**")
    with col3:
        updated_at_placeholder = st.empty()
        updated_at_placeholder.markdown("**Last Updated At ...**")

    graph.add_trace(go.Scatter(y=[], mode='lines', name='Channel 1'))
    graph.add_trace(go.Scatter(y=[], mode='lines', name='Channel 2'))

    graph.update_layout(
        title="Real-time ECG Data",
        xaxis=dict(title="Samples", range=[0, frequency]),
        yaxis_title="Amplitude",
        template="plotly_dark"
    )

    model_data_list = []
    graph_data_list = []
    prediction_list = []

    graph_placeholder.plotly_chart(graph, use_container_width=True, key="ecg_graph")

    for i in range(0, len(ecg_signal), frequency):
        data = ecg_signal[i:i+frequency]
        model_data_list.extend(data.tolist())

        graph_data_list.extend(data.tolist())
        if len(graph_data_list) > frequency:
            graph_data_list = graph_data_list[len(graph_data_list)-frequency:]

        x_range = list(range(len(graph_data_list)))

        graph.data[0].x = x_range
        graph.data[0].y = [val[0] for val in graph_data_list]
        #second channel
        graph.data[1].x = x_range
        graph.data[1].y = [val[1] for val in graph_data_list]

        graph.update_layout()

        graph_placeholder.plotly_chart(graph, use_container_width=True, key=f"ecg_graph_{i}")

        if len(model_data_list) >= model_segment_length:
            model_np_data_list = np.array(model_data_list[:model_segment_length])
            model_input_data = np.expand_dims(model_np_data_list, axis=0)

            prediction = model.predict(model_input_data)
            prediction_binary = 1 if prediction >= 0.5 else 0
            
            average_confidence = calculate_confidence(prediction_binary, prediction_list)
            prediction_list.append(prediction_binary)

            prediction_placeholder.markdown(f"**Prediction:** {'Condition Detected' if prediction_binary else 'No Condition'}")
            confidence_placeholder.markdown(f"**Average Confidence:** {average_confidence:.2f}%")
            updated_at_placeholder.markdown(f"**Last Updated At:** {time.strftime("%I:%M:%S %p", time.localtime())}")

            model_data_list = model_data_list[step_size:]

            if prediction_binary == 1 and average_confidence > 50:
                st.switch_page("./pages/chatbot.py")
                return
        time.sleep(1)
    return

# fix: also determine confidence based on number of predictions 
def calculate_confidence(latest_prediction, prediction_list):
    alpha = 0
    beta = 0
    for prediction in prediction_list:
        if prediction == latest_prediction:
            alpha += 1
        else:
            beta += 1
    
    if(alpha == 0 and beta == 0):
        return 50.0
    return alpha / (alpha + beta) * 100.0

def arg_parser():

    parser = argparse.ArgumentParser(description="Indicate which data you want to simulate (normal or SCA ECG)")
    parser.add_argument('--data_type', type=str, choices=['normal', 'sca'], required=True)
    return parser.parse_args().data_type


def main():
    arg = arg_parser()
    if(arg == 'normal'):
        process_ecg('./data/test_data/normal/18177')
    if(arg == 'sca'):
        process_ecg('./data/test_data/SCD/P0016')
    print("Faulty argument passed")
    return

main()