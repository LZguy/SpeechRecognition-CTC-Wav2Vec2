# Deep Learning for Speech Signals – Exercise 5

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
