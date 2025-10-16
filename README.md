# SpeechRecognition-CTC-Wav2Vec2

This project was completed as part of the **Deep Learning for Speech Signals** course (Technion, Spring 2025).  
It includes two main tasks: implementing the CTC forward probability algorithm and building an ASR system with Wav2Vec2 and a KenLM language model.

---

## 📑 Project Overview
1. **CTC Forward Algorithm**  
   - Implemented in `ex_5_part1.py`.  
   - Computes the probability of a transcription sequence given a matrix of per-frame symbol probabilities `y[t,k]`.  
   - Tested with the provided `mat1.npy` file and sample transcriptions.

2. **Automatic Speech Recognition with Wav2Vec2 + KenLM**  
   - Implemented in `ex_5_part2.py`.  
   - Uses HuggingFace’s Wav2Vec2 pretrained acoustic model.  
   - Decodes outputs with a beam search decoder enhanced by a **KenLM n-gram language model** (`kenlm.arpa`).  
   - Input: train/test audio files with transcripts.  
   - Output: predicted text sequences, aligned with digits and letters.

---

## 📂 Project Structure
```
DLSpeech_Ex5/
├─ docs/
│ ├─ DLSpeech_Ex5_046747.pdf # Original assignment instructions
│ ├─ DLSpeech_Ex5_046747_updated.pdf # Updated instructions
│ ├─ ex_5.pdf # Submitted report
├─ data/ #most data is missing, will be delivered in the future
│ ├─ train_transcription.txt # Training transcriptions
│ ├─ lexicon.txt # Lexicon mapping for beam search
│ ├─ mat1.npy # Probability matrix (used in Part 1)
│ ├─ test/ # Test audio files (not included in repo)
│ └─ train/ # Training audio files (not included in repo)
├─ src/
│ ├─ ex_5_part1.py # CTC forward algorithm
│ ├─ ex_5_part2.py # Wav2Vec2 + KenLM ASR pipeline
│ └─ kenlm.arpa # KenLM language model (large file)
├─ results/
│ └─ output.txt # Example output predictions
├─ .gitignore
├─ LICENSE
└─ README.md
```

---

## 📊 Data

Due to size, the full datasets are not stored directly in this repository.  
They can be downloaded from Google Drive:

- [train.zip](https://drive.google.com/uc?id=1jNmYJoXHlCOTD5j5aXa9DpA7DTEflHbo)  
- [test.zip](https://drive.google.com/uc?id=1-nkIkgEyUWgIpYeoGAY-Ml2LoTnknTIN)  

Unzip them into the `data/` directory before running the code:
```
unzip train.zip -d data/train
unzip test.zip -d data/test
```
Other resources stored in this repo:
- `train_transcription.txt` – contains the training transcriptions.  
- `lexicon.txt` – defines the lexicon for beam search decoding.  
- `mat1.npy` – NumPy array of probabilities for Part 1 (CTC).

---

## 📜 Results
- `output.txt` – Example system output, mapping spoken digits/letters into text.  
- Evaluation included WER (Word Error Rate) and CER (Character Error Rate) using the **jiwer** library.  

---

## ⚙️ Requirements
- Python 3.9+  
- PyTorch + Torchaudio  
- HuggingFace `transformers`  
- `kenlm`  
- `flashlight-text`  
- `numpy`, `pandas`, `tqdm`, `jiwer`  

Install dependencies:
```bash
pip install torch torchaudio transformers numpy pandas tqdm jiwer
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install flashlight-text

## ▶️ How to Run

### Part 1 – CTC Forward Probability

```bash
python src/ex_5_part1.py data/mat1.npy "aaabb" "abc"
```

Outputs the forward probability of the sequence.

---

### Part 2 – ASR with Wav2Vec2 + KenLM

1. Train / fine-tune Wav2Vec2 on the `train/` data.  

2. Run decoding with beam search and KenLM:
   ```bash
   python src/ex_5_part2.py --data data/test --lexicon data/lexicon.txt --lm src/kenlm.arpa
   ```

3. Results are written to `results/output.txt`.
