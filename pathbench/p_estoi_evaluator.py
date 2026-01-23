from typing import Optional, List

import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from phonemizer.phonemize import phonemize
from phonemizer.separator import Separator
import re
import numpy as np
import librosa

from pathbench.reference_evaluator import ReferenceEvaluator, STOI
from pathbench.string_clean import clean_text


class ForcedAlignmentPESTOIEvaluator(ReferenceEvaluator):
    """An evaluator that uses P-ESTOI to compute a score after trimming silence using forced alignment."""

    def __init__(self, model_id: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft", **kwargs):
        super().__init__(**kwargs)
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.cache = {}
        print(f"Phonetic model '{model_id}' loaded on {self.device}.")

    def _trim_audio(self, audio_path: str, transcription: str, language: str) -> Optional[np.ndarray]:
        """
        Trims silence from the beginning and end of an audio file using forced alignment.
        """
        cache_key = (audio_path, transcription, language)
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            speech, sample_rate = librosa.load(audio_path, sr=16000)
            #speech, sample_rate = sf.read(audio_path)
            print(speech, sample_rate)
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
        if not phonemized_reference:
            print(f"Warning: Could not phonemize reference transcription for {audio_path}.")
            return None

        # 2. Get the mapping from phonemes to model vocab indices.
        vocab = self.processor.tokenizer.get_vocab()
        target_phonemes = phonemized_reference.split()
        
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
                emissions, targets, blank=vocab.get(self.processor.tokenizer.pad_token, 0)
            )
        except Exception as e:
            print(f"Forced alignment failed for {audio_path}: {e}")
            return None

        # 4. Get start and end frames of speech

        
        # The frames are in terms of the model's output, which is every 20ms.
        # The model's output has a lower frame rate than the audio.
        # The ratio is sample_rate / (model_output_rate)
        # The model output rate is roughly 50 Hz (1000ms / 20ms).
        # So the ratio is 16000 / 50 = 320.
        
        # Find the index of first non-zero in aligned path
        for i, x in enumerate(aligned_path[0]):
            if x != 0:
                start_idx = i
                break
        
        # Find the index of last non-zero in aligned path
        for i, x in enumerate(reversed(aligned_path[0])):
            if x != 0:
                end_idx = len(aligned_path[0]) - i
                break

  
        ratio = speech.shape[0] / emissions.shape[1]
        #print("ratio", ratio)
        #print("start_idx", start_idx)
        #print("end_idx", end_idx)   
        start_frame = int(start_idx * ratio)
        end_frame = int((end_idx+1) * ratio)

        trimmed_audio = speech[start_frame:end_frame]
        print("trimmed_audio_length", trimmed_audio.shape)
        self.cache[cache_key] = trimmed_audio
        return trimmed_audio.astype(np.float64)


    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
        reference_audios: List[str],
        **kwargs,
    ) -> Optional[float]:
        """
        Computes the P-ESTOI score after trimming silence.
        """
        trimmed_audio = self._trim_audio(audio_path, transcription, language)
        #trimmed_audio = sf.read(audio_path)[0]
        if trimmed_audio is None:
            return None
    
        #reference_audios_data = [sf.read]
        reference_audios_data = [self._trim_audio(path, transcription, language) for path in reference_audios]
        #reference_audios_data = [sf.read(path)[0] for path in reference_audios]
        #print(trimmed_audio.shape, [ref.shape for ref in reference_audios_data])
        stoi_object = STOI(
            normalization_method='RMS',
            centroid_ind=0,
            frame_deletion=True,
            reference_words=reference_audios_data,
            test_words=[trimmed_audio],
            **self.stoi_kwargs
        )
        return stoi_object.estoi_val[0]
