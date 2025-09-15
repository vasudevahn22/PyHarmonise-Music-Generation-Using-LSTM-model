# 🎵 Music Generation using LSTM

## 📌 Overview
This project explores **AI-driven music composition** using a **Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM)** architecture.  
By treating melody creation as a **time-series prediction problem**, the model captures long-term dependencies in musical sequences and generates **coherent, human-like melodies**.  

The pipeline leverages **Keras, Music21, and MuseScore** to preprocess symbolic music, train the LSTM model, and produce MIDI outputs.

---

## 🎯 Objectives
- Preprocess and encode symbolic music data (Kern files).  
- Transpose songs into a common key (C major / A minor) for easier learning.  
- Convert symbolic representations into integer sequences suitable for neural networks.  
- Build and train an **LSTM-based music generation model**.  
- Generate melodies using dynamic sampling techniques (temperature scaling & symbol weighting).  
- Convert generated outputs into **MIDI files** for playback and analysis.  

---

## ⚙️ Approach

### 1. Data Preprocessing
- **Loading songs:** Used `music21` to parse `.krn` files into symbolic streams.  
- **Filtering durations:** Kept only acceptable note durations (e.g., quarter, half notes).  
- **Transposing:** Standardized all songs into **C major** or **A minor**.  

### 2. Encoding & Mapping
- Encoded notes and rests into symbolic sequences (MIDI numbers, rests, prolongation markers).  
- Combined all encoded songs into a single dataset with delimiters.  
- Created a **mapping dictionary** (symbols → integers) for training.  

### 3. Sequence Preparation
- Converted encoded songs into integer lists.  
- Generated **fixed-length training sequences** and **one-hot encoded** them for supervised learning.  

### 4. Model Architecture
- **Input Layer:** Encoded sequences.  
- **LSTM Layer:** 256 units, capturing temporal dependencies.  
- **Dropout Layer:** Prevented overfitting.  
- **Dense Output Layer:** Softmax activation, predicting next musical event.  

### 5. Training
- 50 epochs, batch size = 64.  
- Model saved as `model.h5` for inference.  

### 6. Melody Generation
- Used **seed sequences** as input.  
- Applied **entropy-driven temperature scaling** to adjust randomness.  
- Added **symbol weighting** (rests penalized, notes prioritized) for more musical outputs.  
- Generated symbolic sequences → converted into **MIDI** using `music21`.  

---

## 📈 Results
- Generated melodies were **musically diverse, coherent, and structured**.  
- **Dynamic temperature adjustment** balanced randomness and structure, avoiding monotony or chaos.  
- **Symbol biasing** reduced excessive rests and improved rhythmic flow.  
- Produced MIDI files that can be played back and analyzed in **MuseScore**.  

---

## ⚠️ Challenges & Learnings
- Fixed temperature → caused repetitive or chaotic outputs → solved with **entropy-based adjustment**.  
- Equal symbol probabilities → led to too many rests → solved with **biasing mechanism**.  
- Short training sequences → limited long-term learning → solved by **increasing sequence length**.  

---

## 🎵 Value of Research
This project highlights the **intersection of AI and creativity**, showing how deep learning can aid in:  
- **Automated composition** for musicians.  
- **Therapeutic soundscapes** in healthcare.  
- **Dynamic soundtracks** for games and films.  
- **AI-driven creativity** in cultural industries.  

---

## 🛠️ Tech Stack
- **Languages:** Python  
- **Libraries:** TensorFlow/Keras, NumPy, Pandas, Music21, Matplotlib  
- **Tools:** MuseScore (for analyzing MIDI outputs)  

---
