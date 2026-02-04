from typing import Optional

import librosa
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from phonemizer.phonemize import phonemize
from phonemizer.separator import Separator
import re

from pathbench.evaluator import Evaluator
from pathbench.string_clean import clean_text


class ArticulatoryPrecisionEvaluatorOld(Evaluator):
    """An evaluator that uses a wav2vec 2.0 model to compute articulatory precision."""

    def __init__(self, model_id: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Phonetic model '{model_id}' loaded on {self.device}.")

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
        **kwargs,
    ) -> Optional[float]:
        """
        Computes the articulatory precision score.
        """
        try:
            speech, sample_rate = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Error reading audio file {audio_path}: {e}")
            return None

        if len(speech) < 400:
            print(f"Warning: Skipping audio file {audio_path} because it is too short ({len(speech)} samples).")
            return None

        # Process audio
        input_values = self.processor(
            speech, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.device)

        # Get model outputs
        with torch.no_grad():
            logits = self.model(input_values).logits

        # 1. Phonemize the ground truth transcription.
        separator = Separator(phone=" ", word="|")
        phonemized_reference = phonemize(
            clean_text(transcription),
            language=language,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            separator=separator
        )
        phonemized_reference = re.sub(r"\s+", " ", phonemized_reference.replace("|", " ")).strip()
        print(f"Phonemized reference for {utterance_id}: {phonemized_reference}")
        if not phonemized_reference:
            print(f"Warning: Could not phonemize reference transcription for {utterance_id}.")
            return None

        # 2. Get the mapping from phonemes to model vocab indices.
        vocab = self.processor.tokenizer.get_vocab()
        # reverse mapping of this
        vocab_reverse = {v: k for k, v in vocab.items()}

        #print("vocab", vocab)
        #print("vocab_reverse", vocab_reverse)
        
        # The model seems to have different symbols than the phonemizer.
        # For example, phonemizer might produce 'ə' but the model has 'ə'.
        # Let's assume for now they are compatible.
        target_phonemes = phonemized_reference.split()
        print(target_phonemes)
        
        # ʲ phonemes are not in so remove
        target_phonemes = [p.replace("ʲ", "") for p in target_phonemes]
        target_phonemes = [p.replace("dz", "z") for p in target_phonemes]
        
        for p in target_phonemes:
            if p not in vocab:
                # Remove
                print(f"Warning: Phoneme {p} not in model vocabulary for {audio_path}.")
                target_phonemes.remove(p)
        try:
            target_ids = [vocab[p] for p in target_phonemes]
        except KeyError as e:
            print(f"Phoneme {e} not in model vocabulary.")
            return None

        # 3. Forced alignment
        # Based on https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html
        
        emissions = torch.log_softmax(logits, dim=-1)
        emissions = torch.exp(emissions)
        
        # torchaudio.functional.forced_align requires CPU tensors
        emissions = emissions.cpu()
        targets = torch.tensor(target_ids, dtype=torch.int32).unsqueeze(0)

        print(emissions.shape)
        print(targets.shape)
        print(targets)
        try:
            aligned_path, scores = torchaudio.functional.forced_align(
                emissions, targets, blank=vocab.get(self.processor.tokenizer.pad_token, 0)
            )
        except Exception as e:
            print(f"Forced alignment failed for {utterance_id}: {e}")
            return None

        # 4. Calculate Articulatory Precision
        # The following section first shows the segmentation of the raw model output
        # (including <pad> tokens), and then calculates the articulatory precision score
        # based on the forced alignment with the reference transcription.
        # The score is calculated using average probabilities, not log probabilities.

        # Get the raw segmentation from argmax, which includes <pad> tokens
        print("\n--- Raw Model Output Segmentation (including <pad>) ---")
        best_path = torch.argmax(emissions, dim=-1)[0]
        
        change_points_raw = (best_path.diff() != 0).nonzero(as_tuple=True)[0]
        segments_raw = torch.cat([
            torch.tensor([0], device=best_path.device),
            change_points_raw + 1,
            torch.tensor([best_path.shape[0]], device=best_path.device)
        ])

        probabilities = emissions[0]

        total_prob = 0
        num_phonemes = 0 
        for i, (start, end) in enumerate(zip(segments_raw[:-1], segments_raw[1:])):
            token_id = best_path[start].item()
            avg_prob = probabilities[start:end, token_id].mean().item()
            token_str = vocab_reverse.get(token_id, "UNK")

            # If <pad>, skip from calculation
            if not token_str == self.processor.tokenizer.pad_token:
                num_phonemes += 1
                total_prob += avg_prob
            #print(f"  Segment {i} ({token_str}): frames {start}-{end-1}, avg_prob={avg_prob:.4f}")


        if num_phonemes > 0:
            artp_score = total_prob / num_phonemes
        else:
            artp_score = 0.0 # Or handle as an error
        
        #print(f"\n--- Final Score ---")
        #print(f"Utterance ID: {utterance_id}")
        #print(f"Transcription: {transcription}")
        #print(f"Phonemized Reference: {phonemized_reference}")
        #print(f"Articulatory Precision Score: {artp_score}")
        #print("(Note: Score is based on forced alignment of reference text, not raw model output)")

        return artp_score
    
class ArticulatoryPrecisionEvaluator(Evaluator):
    """An evaluator that uses a wav2vec 2.0 model to compute articulatory precision."""

    def __init__(self, model_id: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Phonetic model '{model_id}' loaded on {self.device}.")

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
        **kwargs,
    ) -> Optional[float]:
        """
        Computes the articulatory precision score.
        """
        try:
            speech, sample_rate = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Error reading audio file {audio_path}: {e}")
            return None

        if len(speech) < 400:
            print(f"Warning: Skipping audio file {audio_path} because it is too short ({len(speech)} samples).")
            return None

        # Process audio
        input_values = self.processor(
            speech, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.device)

        # Get model outputs
        with torch.no_grad():
            logits = self.model(input_values).logits

        # 1. Phonemize the ground truth transcription.
        separator = Separator(phone=" ", word="|")
        phonemized_reference = phonemize(
            clean_text(transcription),
            language=language,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            separator=separator
        )
        phonemized_reference = re.sub(r"\s+", " ", phonemized_reference.replace("|", " ")).strip()
        print(f"Phonemized reference for {utterance_id}: {phonemized_reference}")
        if not phonemized_reference:
            print(f"Warning: Could not phonemize reference transcription for {utterance_id}.")
            return None

        # 2. Get the mapping from phonemes to model vocab indices.
        vocab = self.processor.tokenizer.get_vocab()
        # reverse mapping of this
        vocab_reverse = {v: k for k, v in vocab.items()}

        #print("vocab", vocab)
        #print("vocab_reverse", vocab_reverse)
        
        # The model seems to have different symbols than the phonemizer.
        # For example, phonemizer might produce 'ə' but the model has 'ə'.
        # Let's assume for now they are compatible.
        target_phonemes = phonemized_reference.split()
        print(target_phonemes)
        
        # ʲ phonemes are not in so remove
        target_phonemes = [p.replace("ʲ", "") for p in target_phonemes]
        target_phonemes = [p.replace("dz", "z") for p in target_phonemes]
        
        for p in target_phonemes:
            if p not in vocab:
                # Remove
                print(f"Warning: Phoneme {p} not in model vocabulary for {audio_path}.")
                target_phonemes.remove(p)
        try:
            target_ids = [vocab[p] for p in target_phonemes]
        except KeyError as e:
            print(f"Phoneme {e} not in model vocabulary.")
            return None

        # 3. Forced alignment
        # Based on https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html
        
        emissions = torch.log_softmax(logits, dim=-1)
        emissions = torch.exp(emissions)
        
        # torchaudio.functional.forced_align requires CPU tensors
        emissions = emissions.cpu()
        targets = torch.tensor(target_ids, dtype=torch.int32).unsqueeze(0)

        print(emissions.shape)
        print(targets.shape)
        print(targets)
        try:
            aligned_path, scores = torchaudio.functional.forced_align(
                emissions, targets, blank=vocab.get(self.processor.tokenizer.pad_token, 0)
            )
        except Exception as e:
            print(f"Forced alignment failed for {utterance_id}: {e}")
            return None

        # 4. Calculate Articulatory Precision
        # The following section first shows the segmentation of the raw model output
        # (including <pad> tokens), and then calculates the articulatory precision score
        # based on the forced alignment with the reference transcription.
        # The score is calculated using average probabilities, not log probabilities.

        # Get the raw segmentation from argmax, which includes <pad> tokens
        print("\n--- Raw Model Output Segmentation (including <pad>) ---")
        best_path = aligned_path
        
        #print("best_path.shape", best_path.shape)
        #print("emissions.shape", emissions.shape)
        #print("scores.shape", scores.shape)
        #print("best path:", best_path)
        #print("emisisons", emissions)
        #change_points_raw = (best_path.diff() != 0).nonzero(as_tuple=True)[0]
        #segments_raw = torch.cat([
        #    torch.tensor([0], device=best_path.device),
        #    change_points_raw + 1,
        #    torch.tensor([best_path.shape[0]], device=best_path.device)
        #])

        total_prob = 0
        num_phonemes = 0


        for i, (token, score) in enumerate(zip(best_path[0,:], scores[0,:])):
            #print(f"  Frame {i}: token_id={token}, score={score}")

            # If <pad>, skip from calculation
            if not token == vocab.get(self.processor.tokenizer.pad_token, 0):
                num_phonemes += 1
                total_prob += score

        if num_phonemes > 0:
            artp_score = float(total_prob / num_phonemes)
        else:
            artp_score = 0.0 # Or handle as an error
        print(artp_score)
        return artp_score