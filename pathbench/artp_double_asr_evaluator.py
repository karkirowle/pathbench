from typing import Optional, Dict, List

import librosa
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from phonemizer.phonemize import phonemize
from phonemizer.separator import Separator
import re
import os
import jiwer
from pyctcdecode import build_ctcdecoder


from pathbench.evaluator import Evaluator
from pathbench.string_clean import clean_text


class ArtPDoubleASREvaluator(Evaluator):
    """An evaluator that uses a wav2vec 2.0 model to compute articulatory precision."""

    def __init__(self, language: str, model_id: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"):
        self.phonetic_processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.phonetic_model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.phonetic_model.to(self.device)
        print(f"Phonetic model '{model_id}' loaded on {self.device}.")

        self.language = language
        model_ids = {
            "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
            "en-us": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
            "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
            "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
            "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
        }

        if language not in model_ids:
            raise ValueError(f"Language '{language}' is not supported for ArtPDoubleASREvaluator.")

        asr_model_id = model_ids[language]
        self.processor = Wav2Vec2Processor.from_pretrained(asr_model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(asr_model_id)
        self.model.to(self.device)
        print(f"ASR model '{asr_model_id}' for language '{language}' loaded on {self.device}.")

        # Assuming the 'lms' directory is at the project root.
        lms_dir = 'lms'
        lm_paths = {
            "en": os.path.join(lms_dir, "wiki_en_token.arpa"),
            "nl": os.path.join(lms_dir, "wiki_nl_token.arpa"),
            "es": os.path.join(lms_dir, "wiki_es_token.arpa.bin"),
            "it": os.path.join(lms_dir, "wiki_it_token.arpa.bin"),
        }

        lm_lang = language.split('-')[0]
        if lm_lang not in lm_paths:
            lm_lang = 'en' # Default to 'en' if no specific LM
        
        lm_path = lm_paths.get(lm_lang)
        self.decoder = None
        if lm_path and os.path.exists(lm_path):
            vocab_dict = self.processor.tokenizer.get_vocab()
            sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
            labels = list(sorted_vocab_dict.keys())
            
            self.decoder = build_ctcdecoder(
                labels,
                kenlm_model_path=lm_path,
            )
            print(f"CTC decoder for '{language}' with LM '{lm_path}' built.")
        else:
            print(f"Warning: Language model for '{language}' not found at '{lm_path}'. No decoder built.")


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
        if language != self.language:
            print(f"Warning: ArtPDoubleASREvaluator was initialized for language '{self.language}' but received a request for '{language}'. Skipping.")
            return None

        if not self.decoder:
            print(f"Error: No decoder available for language '{language}'.")
            return None
            
        try:
            speech, sample_rate = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Error reading audio file {audio_path}: {e}")
            return None

        if len(speech) < 400:
            print(f"Warning: Skipping audio file {audio_path} because it is too short ({len(speech)} samples).")
            return None

        # Get transcription from language-specific ASR with LM
        input_values_asr = self.processor(
            speech, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values
        input_values_asr = input_values_asr.to(self.device)

        with torch.no_grad():
            logits_asr = self.model(input_values_asr).logits
        
        logits_numpy = logits_asr.cpu().numpy()[0]
        lm_transcription = self.decoder.decode(logits_numpy)


        # Process audio with phonetic model
        input_values_phonetic = self.phonetic_processor(
            speech, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values
        input_values_phonetic = input_values_phonetic.to(self.device)

        # Get model outputs
        with torch.no_grad():
            logits = self.phonetic_model(input_values_phonetic).logits

        # 1. Phonemize the n-gram improved transcription.
        separator = Separator(phone=" ", word="|")
        phonemized_reference = phonemize(
            clean_text(lm_transcription),
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
        vocab = self.phonetic_processor.tokenizer.get_vocab()
        
        target_phonemes = phonemized_reference.split()
        
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
        emissions = torch.log_softmax(logits, dim=-1)
        emissions = torch.exp(emissions)
        
        emissions = emissions.cpu()
        targets = torch.tensor(target_ids, dtype=torch.int32).unsqueeze(0)

        try:
            aligned_path, scores = torchaudio.functional.forced_align(
                emissions, targets, blank=vocab.get(self.phonetic_processor.tokenizer.pad_token, 0)
            )
        except Exception as e:
            print(f"Forced alignment failed for {utterance_id}: {e}")
            return None

        # 4. Calculate Articulatory Precision
        best_path = aligned_path
        
        total_prob = 0
        num_phonemes = 0

        for i, (token, score) in enumerate(zip(best_path[0,:], scores[0,:])):
            if not token == vocab.get(self.phonetic_processor.tokenizer.pad_token, 0):
                num_phonemes += 1
                total_prob += score

        if num_phonemes > 0:
            artp_score = float(total_prob / num_phonemes)
        else:
            artp_score = 0.0
        
        return artp_score
