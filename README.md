# 🎙️ VOIP Speaker Verification for Indian Languages

A deep-learning based **speaker verification framework for Indian-language speech over Voice over IP (VOIP) communication channels**.

This repository investigates the robustness and adaptation of modern speaker-embedding models for VOIP speech, with experiments based on **ResNet293** and **ECAPA-TDNN** architectures. The project also explores **fine-tuning on VOIP speech data** to improve speaker verification performance under channel and language variability.

---

## 📌 Overview

**Speaker verification** is the task of determining whether a speech signal belongs to a claimed speaker.

Unlike speaker identification, where the system determines *who* is speaking, speaker verification answers:

> **"Is this speech really from the claimed speaker?"**

This project focuses on the challenges involved in speaker verification when speech is transmitted through **VOIP communication systems**, particularly for **Indian languages**.

VOIP speech can introduce channel-related variations such as:

* Compression artifacts
* Network-related distortions
* Variable recording conditions
* Microphone and device differences
* Background noise
* Language and accent variations

The repository evaluates deep speaker-embedding models under these conditions and investigates whether **VOIP-specific fine-tuning** can improve verification performance.

---

## 🎯 Objectives

The main objectives of this project are:

1. To investigate deep-learning approaches for speaker verification using VOIP speech.
2. To evaluate **ResNet293** and **ECAPA-TDNN** based speaker representations.
3. To study the effect of VOIP-channel characteristics on speaker verification.
4. To investigate fine-tuning of pretrained speaker models using VOIP speech.
5. To analyze performance across Indian-language speech conditions.
6. To evaluate speaker verification systems using appropriate performance metrics.
7. To study cross-language and cross-condition speaker verification performance.

---

## 🧠 Models

### 1. ResNet293

The project uses a **ResNet293-based speaker recognition/verification architecture** for extracting discriminative speaker representations from speech.

ResNet-based speaker encoders use deep residual connections to learn robust acoustic representations while allowing very deep neural networks to be trained effectively.

Implementation:

```text
resnet293_on_VOIP.py
resnet293_on_VOIP_finetuned.py
```

---

### 2. ECAPA-TDNN

The project also investigates **ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN)**, a widely used architecture for speaker representation learning.

ECAPA-TDNN is designed to produce speaker embeddings that capture speaker-discriminative characteristics while being robust to variations in speech.

Implementation:

```text
ecapa.py
finetune_ecapa_voip.py
```

The repository therefore provides an experimental basis for comparing different deep speaker-embedding approaches for VOIP speech.

---

## 🔬 Research Workflow

The overall experimental workflow can be summarized as:

```text
                 VOIP Speech Data
                        │
                        ▼
               Data Organization
                        │
                        ▼
              Audio Pre-processing
                        │
                        ▼
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
        ResNet293              ECAPA-TDNN
             │                     │
             ▼                     ▼
      Speaker Embeddings     Speaker Embeddings
             │                     │
             └──────────┬──────────┘
                        ▼
                 Verification
                        │
                        ▼
              Similarity Scoring
                        │
                        ▼
                 Thresholding
                        │
                        ▼
               Performance Metrics
                        │
                        ▼
              Model Comparison
                        │
                        ▼
             VOIP Fine-tuning Study
```

---

## 📊 Speaker Verification

The system follows the standard speaker-verification paradigm.

Given:

* **Enrollment speech** from a claimed speaker
* **Test speech** from an unknown utterance

the system generates speaker embeddings and calculates their similarity.

Conceptually:

```text
Enrollment Audio
       │
       ▼
Speaker Encoder
       │
       ▼
Speaker Embedding
       │
       │
       │ Similarity
       ▼
Test Audio ──► Speaker Encoder ──► Test Embedding
       │
       ▼
Similarity Score
       │
       ▼
Threshold
       │
   ┌───┴────┐
   │        │
Accept    Reject
```

A higher similarity score indicates greater similarity between the enrollment and test speaker representations.

---

## 📈 Evaluation Metrics

Speaker verification performance can be evaluated using metrics such as:

### Equal Error Rate (EER)

**EER** is one of the commonly used metrics for speaker verification.

It is the operating point where:

```text
False Acceptance Rate (FAR)
              =
False Rejection Rate (FRR)
```

Lower EER indicates fewer verification errors at this operating point.

---

### False Acceptance Rate (FAR)

The percentage of impostor trials incorrectly accepted as genuine speakers.

```text
FAR = False Acceptances / Total Impostor Trials
```

---

### False Rejection Rate (FRR)

The percentage of genuine speaker trials incorrectly rejected.

```text
FRR = False Rejections / Total Genuine Trials
```

---

### ROC / DET Analysis

The verification scores can also be analyzed by varying the decision threshold and examining the trade-off between genuine acceptance and impostor acceptance.

This provides a more complete view of system behavior than reporting a single threshold.

---

## 🌏 Indian-Language Focus

A major motivation of this work is the evaluation of speaker verification in **Indian-language speech conditions**.

Indian speech presents additional challenges due to:

* Large linguistic diversity
* Multiple regional accents
* Code-switching and multilingual speakers
* Different phonetic characteristics
* Speaker-dependent pronunciation patterns
* Variations in recording and communication channels

The repository can therefore serve as a research framework for investigating **language-dependent and cross-language effects on speaker verification**.

---

## 🔄 VOIP Fine-Tuning

A key component of this project is the investigation of **fine-tuning speaker-verification models using VOIP speech**.

The motivation is that a model trained on conventional speech recordings may encounter a distribution shift when it receives speech transmitted through VOIP channels.

The fine-tuning process can be viewed as:

```text
Pretrained Speaker Model
          │
          ▼
   VOIP Training Data
          │
          ▼
      Fine-tuning
          │
          ▼
VOIP-adapted Speaker Model
          │
          ▼
     Evaluation
```

The repository includes dedicated implementations for ECAPA-TDNN and ResNet293 VOIP experiments.

---

## 📁 Repository Structure

```text
VOIP-Speaker-Verification-for-Indian-Languages/
│
├── README.md
│
├── ecapa.py
├── finetune_ecapa_voip.py
│
├── resnet293_on_VOIP.py
├── resnet293_on_VOIP_finetuned.py
│
├── features.py
├── Segregate.py
├── split.py
│
├── check.py
├── count.py
├── count1.py
├── count2.py
│
├── Wavelem.py
├── quanta.py
├── scam.py
├── te.py
├── tecross.py
├── tecrosswft.py
├── tecrosswft_fixed.py
├── tq.py
├── tqcross.py
├── tqcrosswft_fixed.py
├── try.py
└── try2.py
```

The repository currently contains scripts for model implementation, fine-tuning, feature processing, data organization, evaluation, and experimental analysis. ([GitHub][1])

---

## 🛠️ Technologies

The project is implemented primarily in **Python** and uses deep-learning and speech-processing tools for experimentation.

Typical components of the experimental environment include:

* Python
* PyTorch
* Torchaudio
* NumPy
* SciPy
* Librosa
* Scikit-learn
* Matplotlib
* Pandas

> The exact package versions should be specified in a `requirements.txt` file for complete reproducibility.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/satishchikkamath/VOIP-Speaker-Verification-for-Indian-Languages.git

cd VOIP-Speaker-Verification-for-Indian-Languages
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it using:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchaudio numpy scipy librosa scikit-learn matplotlib pandas
```

For reproducible experiments, it is recommended to maintain a project-specific:

```text
requirements.txt
```

file containing the exact versions used in the experiments.

---

## ▶️ Running the Experiments

### ResNet293

The ResNet293 implementation can be explored through:

```bash
python resnet293_on_VOIP.py
```

For the fine-tuned version:

```bash
python resnet293_on_VOIP_finetuned.py
```

### ECAPA-TDNN

The ECAPA-TDNN implementation is available through:

```bash
python ecapa.py
```

For VOIP-specific fine-tuning:

```bash
python finetune_ecapa_voip.py
```

> Dataset paths, model checkpoints, GPU configuration, and other experiment-specific parameters may need to be modified according to the local environment.

---

## 🧪 Experimental Components

The repository contains several supporting scripts for the experimental pipeline.

| Component                            | Purpose                                           |
| ------------------------------------ | ------------------------------------------------- |
| `ecapa.py`                           | ECAPA-TDNN based speaker-verification experiments |
| `finetune_ecapa_voip.py`             | ECAPA-TDNN fine-tuning using VOIP speech          |
| `resnet293_on_VOIP.py`               | ResNet293 experiments on VOIP speech              |
| `resnet293_on_VOIP_finetuned.py`     | Fine-tuned ResNet293 experiments                  |
| `features.py`                        | Feature extraction / processing                   |
| `Segregate.py`                       | Data organization / segregation                   |
| `split.py`                           | Dataset splitting                                 |
| `count.py`, `count1.py`, `count2.py` | Dataset/statistical analysis                      |
| `Wavelem.py`                         | Waveform-related processing                       |
| `te*.py`                             | Experimental evaluation / testing scripts         |
| `tq*.py`                             | Experimental evaluation / analysis scripts        |

The repository contains multiple experimental scripts reflecting different stages and configurations of the research work. ([GitHub][1])

---

## 📌 Research Contributions

This repository provides an experimental framework for studying:

* Deep-learning based speaker verification for VOIP speech
* ResNet293 speaker representations
* ECAPA-TDNN speaker representations
* VOIP-specific model adaptation
* Fine-tuning of pretrained speaker models
* Indian-language speech conditions
* Cross-language speaker verification
* Speaker verification evaluation using threshold-based metrics

---

## ⚠️ Dataset and Reproducibility

The audio datasets used in the experiments are **not included in this repository**.

Users should obtain the relevant datasets from their original sources and comply with their respective:

* Licensing conditions
* Terms of use
* Privacy requirements
* Research-use restrictions

Dataset paths should be updated in the scripts before running experiments.

For reproducible research, it is recommended to document:

1. Dataset version
2. Number of speakers
3. Number of utterances
4. Language distribution
5. Train/validation/test split
6. Sampling rate
7. Audio preprocessing
8. Model checkpoint
9. Training parameters
10. Evaluation protocol

---

## 📚 Research Context

Speaker verification is an important component of modern voice-based authentication and speech technologies. However, models developed using conventional speech corpora may experience performance degradation when deployed in different acoustic, linguistic, or communication environments.

This project therefore focuses specifically on the intersection of:

**Speaker Verification + VOIP Speech + Indian Languages + Deep Learning**

The use of both **ResNet293 and ECAPA-TDNN** enables investigation of different speaker-embedding architectures, while VOIP-specific fine-tuning provides a mechanism for adapting pretrained models to the target communication environment.

---

## 🔮 Future Work

Potential extensions of this work include:

* Evaluation on a larger number of Indian languages
* Cross-language speaker verification experiments
* Cross-accent evaluation
* Noise and channel robustness experiments
* Data augmentation for VOIP conditions
* Comparison with additional speaker encoders such as WavLM and x-vector systems
* Calibration of verification scores
* DET and ROC curve analysis
* Language-independent speaker embeddings
* Anti-spoofing and replay-attack detection
* Evaluation on real-world telephony and VOIP channels
* Deployment as a real-time speaker-verification service

---

## 👨‍💻 Author

**Satish Chikkamath**

Department of Electronics and Communication Engineering
KLE Technological University, India

---

## 📄 Citation

If you use this repository or the associated experimental work in your research, please cite the corresponding publication.

```bibtex
@article{chikkamath_voip_speaker_verification,
  title   = {VOIP Speaker Verification for Indian Languages},
  author  = {Chikkamath, Satish},
  year    = {2026},
  note    = {Research implementation and experimental repository}
}
```

> Replace the BibTeX entry above with the final bibliographic information once the associated paper is published.

---

## ⭐ Acknowledgements

This work is part of research on **speech processing, speaker verification, and AI-based voice technologies for Indian-language communication systems**.

---

## 📜 License

Please add an appropriate open-source license before distributing or reusing this repository.

For example:

```text
MIT License
```

or another license appropriate to the datasets, pretrained models, and code used in the project.

---

## 🔗 Repository

**GitHub:**
[https://github.com/satishchikkamath/VOIP-Speaker-Verification-for-Indian-Languages](https://github.com/satishchikkamath/VOIP-Speaker-Verification-for-Indian-Languages)

A couple of things I **deliberately did not invent**: specific EER/accuracy numbers, exact dataset size, exact Indian languages, training hyperparameters, or claims about which model performs better. Your current repository establishes the use of **ResNet293, ECAPA-TDNN and VOIP fine-tuning**, but the README does not document those experimental results yet. ([GitHub][2])

**One important improvement:** if this repository is connected to your **speaker-verification research paper**, I would make the README even stronger by adding a **Results** section with your actual EER, FAR, FRR, accuracy, language-wise results, and a **ResNet293 vs ECAPA-TDNN comparison table**. That would make the GitHub repository look much more like a reproducible research artifact rather than just a collection of Python files.

[1]: https://github.com/satishchikkamath/VOIP-Speaker-Verification-for-Indian-Languages "GitHub - satishchikkamath/VOIP-Speaker-Verification-for-Indian-Languages · GitHub"
[2]: https://github.com/satishchikkamath/VOIP-Speaker-Verification-for-Indian-Languages/blob/main/README.md "VOIP-Speaker-Verification-for-Indian-Languages/README.md at main · satishchikkamath/VOIP-Speaker-Verification-for-Indian-Languages · GitHub"
