# PodFlow
# PodFlow: AI-Powered Podcast Segmentation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Flask](https://img.shields.io/badge/Flask-Backend-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue)

**PodFlow** is an automated end-to-end machine learning pipeline that identifies topic changes in long-form audio. It ingests podcast RSS feeds, processes audio using a custom Convolutional Recurrent Neural Network (CRNN), and serves timestamped chapters via a responsive web interface.

## Key Features

* **Automated Ingestion:** Daily cron jobs scrape RSS feeds for new episodes from top sports podcasts.
* **Deep Learning Segmentation:** Uses a custom PyTorch model to detect transition points (chapters) in audio without human intervention.
* **Memory-Optimized Processing:** Implements chunked audio loading and sliding-window inference to process 3+ hour audio files on low-memory (2GB RAM) cloud instances.
* **Full-Stack Deployment:** Deployed on DigitalOcean using Nginx, Gunicorn, and PostgreSQL with SSL encryption.

---

##  Architecture

The system operates on a cyclical 24-hour schedule:

1.  **Ingestion:** `ingest_podcasts.py` parses RSS feeds and updates the PostgreSQL database with new episode metadata.
2.  **Processing:** `batch_process.py` performs the heavy lifting:
    * Downloads new MP3 files.
    * Extracts acoustic features (MFCCs, Spectral Contrast, Energy) using `librosa`.
    * Runs inference using the CRNN model to generate chapter timestamps.
3.  **Serving (24/7):** A Flask API serves the episode and chapter data to a frontend web player.
4.  **Cleanup (Weekly):** A maintenance script prunes data older than 30 days to optimize storage costs.

---

## The Model (CRNN)

The core of this project is a **Convolutional Recurrent Neural Network** trained to recognize audio boundaries (silence, music transitions, and topic shifts).

### Architecture Details
* **Input:** Sequence of audio feature vectors (MFCCs, Chroma, Spectral Flux).
* **CNN Layers:** 1D Convolutional layers extract local patterns and acoustic textures.
* **Bi-LSTM Layers:** Bidirectional Long Short-Term Memory layers analyze temporal context (past and future frames) to understand the flow of conversation.
* **Output:** A sigmoid activation function outputs a probability score (0-1) for every time step, indicating the likelihood of a chapter boundary.

### Engineering Constraints
Running this model on a standard VPS (2GB RAM) required significant optimization. The pipeline uses **chunked processing** and **Python generators** to handle large audio arrays without triggering Out-Of-Memory (OOM) kills.

---

## 🛠️ Local Installation

Follow these steps to run the pipeline on your local machine.

### Prerequisites
* Python 3.8+
* PostgreSQL installed locally

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/podflow.git](https://github.com/YOUR_USERNAME/podflow.git)
cd podflow
