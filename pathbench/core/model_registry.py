"""Global cache for HuggingFace model instances.

Ensures each unique model is loaded only once, regardless of how many
evaluators request it.  All models are placed in eval() mode and used
with torch.no_grad() -- they carry no mutable state that could leak
between callers.
"""

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2Processor

_ctc_models = {}   # model_id -> (processor, model, device)
_feat_models = {}  # (model_id, layer) -> featurizer_fn


def get_ctc_model(model_id: str):
    """Return a shared (processor, model, device) tuple for a Wav2Vec2ForCTC model."""
    if model_id not in _ctc_models:
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        _ctc_models[model_id] = (processor, model, device)
        print(f"Model '{model_id}' loaded on {device}.")
    return _ctc_models[model_id]


def get_featurizer(model_id: str, layer: int):
    """Return a shared featurizer function for a Wav2Vec2Model.

    The returned callable accepts a 1-D numpy audio array and returns
    a 2-D numpy feature matrix (frames x hidden_dim).
    """
    key = (model_id, layer)
    if key not in _feat_models:
        model = Wav2Vec2Model.from_pretrained(model_id)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        def _featurize(audio_data):
            if audio_data.dtype == np.float64:
                audio_data = audio_data.astype(np.float32)
            input_values = torch.from_numpy(audio_data).unsqueeze(0).to(device)
            with torch.no_grad():
                hidden_states = model(
                    input_values, output_hidden_states=True
                ).hidden_states
            return hidden_states[layer].squeeze(0).cpu().numpy()

        _feat_models[key] = _featurize
        print(f"Featurizer '{model_id}' (layer {layer}) loaded on {device}.")
    return _feat_models[key]
