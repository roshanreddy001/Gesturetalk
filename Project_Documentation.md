# Project Documentation: GestureTalk

## 1. Abstract
The "GestureTalk" project is a real-time hand gesture recognition and translation system designed to break communication barriers for individuals with speech and hearing impairments. The system captures live video feeds to detect hand gestures using MediaPipe's landmark extraction. It employs a temporal Deep Neural Network (DNN) with 1D Convolutional layers via TensorFlow Lite to accurately classify these gestures into 34 distinct classes, ranging from daily greetings to medical emergencies. To ensure seamless communication, the system incorporates NVIDIA Riva Neural Machine Translation (NMT) for translating recognized sentences from English into multiple targeted languages. Furthermore, it integrates a multi-lingual Text-to-Speech (TTS) synthesis engine that dynamically vocalizes the translated text in real-time. A Flask-based web interface provides users with real-time visual feedback, language selection, and customizable speech settings, bridging the gap between sign language and spoken languages globally.

## 2. Objectives
* **Real-Time Gesture Recognition:** Develop a low-latency, highly accurate system capable of detecting and classifying 34 diverse hand gestures using standard webcam input.
* **Contextual Sentence Building:** Implement a temporal buffering mechanism that logically strings continuous raw gestures together into meaningful, punctuated English sentences.
* **Multi-lingual Translation:** Integrate NVIDIA Riva NMT for high-accuracy, real-time translation of generated sentences into various user-selected languages, coupled with an offline fallback mechanism.
* **Vocal Output Generation:** Synthesize natural-sounding speech from the translated text using multi-lingual TTS engines (gTTS/pyttsx3) to facilitate instant auditory communication.
* **Interactive User Interface:** Design a responsive web application (using Flask) that provides users with a live video feed, real-time prediction overlays, and intuitive controls for language and audio adjustments.
* **Optimized Inference for Edge-Deployment:** Ensure high performance and low computational overhead by operating a lightweight quantized TFLite model, suitable for edge devices.

## 3. Models and Architecture
### 3.1. Feature Extraction Module (MediaPipe Hands)
* **Process:** Captures 42 (x, y) landmark coordinates across both hands per frame, generating an 84-dimensional continuous feature vector.
* **Normalization:** Applies spatial normalization (wrist-relative translation and scale-invariance based on palm size) to handle variance in camera distance and user hand sizes.

### 3.2. Temporal Preprocessor
* **Process:** Maintains a stateful rolling window buffer of the last 10 frames to capture essential spatial-temporal dynamics of moving gestures, mapping static frames into sequential sequences.

### 3.3. Gesture Classification Engine (TensorFlow Lite DNN)
* **Architecture (`src/train_model.py`):**
  * `Input Layer`: Shape (10 time steps, 84 features).
  * `Conv1D Layer 1`: 128 filters, kernel size 3, ReLU activation, followed by Batch Normalization and 20% Dropout.
  * `Conv1D Layer 2`: 64 filters, kernel size 3, ReLU activation, followed by Batch Normalization and 20% Dropout.
  * `GlobalAveragePooling1D`: Reduces sequences to a fixed 1D vector.
  * `Dense Hidden Layer`: 64 units, ReLU activation.
  * `Output Layer`: Dense 34 units with Softmax activation for multi-class probability distribution.
* **Optimization:** The trained model is quantized and converted into a `.tflite` format using representative dataset calibration. This drastically reduces the memory footprint and speeds up inference without significant accuracy trade-offs.

### 3.4. AI Translation Engine (NVIDIA Riva NMT)
* **Process:** Acts as the primary backend for sophisticated English-to-Target language Neural Machine Translation.
* **Resilience:** Integrates an offline dictionary fallback (`src/offline_dict.py`) to guarantee functionality even during network or API disruptions.

### 3.5. Concurrent Processing & Web Backend
* **Process:** Built on Flask and multi-threading. Video inference runs continuously while text-enhancement, AI translation, and TTS generation execute as concurrent background jobs, eliminating UI and video freeze loops.

---

## 4. PowerPoint Presentation (PPT) Structure
*You can use the following slide-by-slide structure to build your project presentation.*

### Slide 1: Title Slide
* **Title:** GestureTalk: Real-Time Hand Gesture-to-Speech Translation
* **Subtitle:** Bridging Communication Gaps using Deep Learning & NVIDIA Riva AI
* **Presenters:** [Your Names/Batch Details]
* **Course/Sem:** [CSE-4006] / Semester 6

### Slide 2: Introduction & Problem Statement
* **Problem:** Millions of individuals with speech and hearing impairments face daily barriers in verbal communication.
* **Current limitations:** Most people don't understand sign language, and human translators are not always available.
* **Solution (GestureTalk):** An automated AI system translating continuous hand gestures into spoken sentences in multiple languages in real-time using standard cameras.

### Slide 3: Project Objectives
* Enable real-time, low-latency gesture tracking using regular webcams.
* Translate continuous gestures into grammatically correct sentences.
* Provide multi-lingual support via text translation.
* Deliver auditory feedback using multi-lingual Text-to-Speech (TTS).
* Ensure smooth performance with lightweight edge-optimized AI models.

### Slide 4: System Architecture overview
* **Frontend:** Browser-based UI (HTML/Flask) to capture video and playback audio.
* **Feature Extraction:** MediaPipe Hands extracts 84 continuous 2D coordinates.
* **Deep Learning Model:** TFLite 1D-CNN temporal sequence model for classification.
* **NLP & Translation:** Sentence buffering followed by NVIDIA Riva Neural Machine Translation.
* **Audio Synthesis:** Multi-lingual TTS pipelines translating text into `base64` audio for immediate playback.

### Slide 5: The Deep Learning Model
* **Model Type:** 1D Convolutional Neural Network (CNN)
* **Input Window:** 10 consecutive frames per sequence.
* **Layers:**
  * Two `Conv1D` blocks with Batch Normalization & Dropout to prevent overfitting.
  * `GlobalAveragePooling1D` for temporal abstraction.
  * Fully Connected `Dense` network routing to 34 classification nodes.
* **Efficiency:** Model is quantized into a `.tflite` format, dramatically reducing RAM usage and increasing inference speed.

### Slide 6: NVIDIA Riva Integration
* Why NVIDIA Riva? Real-time performance and superior Neural Machine Translation (NMT).
* Gesture logic yields raw English words (e.g., "Hello" + "Friend").
* Text is polished to "Hello my friend".
* NVIDIA Riva dynamically translates this base English into languages like Spanish, Hindi, French, etc., enabling global accessibility.

### Slide 7: Unique Features & Resilience
* **Multi-threading:** Video capture and inference never freeze while waiting for server audio/translations.
* **Offline Fallback:** System falls back to a local pre-defined dictionary translation map if AI API limits or network drops out.
* **Smoothing Buffers:** Employs majority-vote prediction smoothing to remove noise or hand jitters mid-action.

### Slide 8: Future Scopes & Enhancements
* Incorporating facial emotion recognition to dynamically shape sentence tone (e.g. asking a question vs stating a fact).
* Expanding vocabulary dynamically beyond 34 gestures.
* Native mobile app deployment utilizing on-device hardware acceleration.

### Slide 9: Conclusion & QA
* **Conclusion:** GestureTalk successfully demonstrates how lightweight Deep Learning and cloud-based AI can harmoniously create an accessible, inclusive tech environment.
* **Thank You!**
* **Questions?**
