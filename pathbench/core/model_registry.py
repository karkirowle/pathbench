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

# Hugging Face Hub commit of every model PathBench loads, pinned so that an
# upstream re-upload cannot silently change scores. These are the revisions
# behind the published results (each was still the upstream ``main`` on
# 2026-11-01; none had changed since 2023). Models not listed here are loaded
# from ``main``.
MODEL_REVISIONS = {
    "facebook/wav2vec2-large": "312b2410566b698c7a649068d413b2067848bd75",
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft": "2c733782da5604684829819a5eb744c193fe9398",
    "facebook/hubert-large-ll60k": "ff022d095678a2995f3c49bab18a96a9e553f782",
    "jonatasgrosman/wav2vec2-large-xlsr-53-english": "569a6236e92bd5f7652a0420bfe9bb94c5664080",
    "jonatasgrosman/wav2vec2-large-xlsr-53-spanish": "96d7e9b4e4a78af515a3c6d3cee7c0826045d276",
    "jonatasgrosman/wav2vec2-large-xlsr-53-dutch": "46f221381d200f7bef268309b3f02023ccf11fcc",
    "jonatasgrosman/wav2vec2-large-xlsr-53-italian": "dab04a3e00d8326052f3fb22a6ff276b822f6131",
    "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn": "99ccb2737be22b8bb50dcfcc39ad4d567fb90cfd",
}


def model_revision(model_id: str):
    """Pinned Hub revision for ``model_id``, or None (= ``main``) if unpinned."""
    return MODEL_REVISIONS.get(model_id)


def get_ctc_model(model_id: str):
    """Return a shared (processor, model, device) tuple for a Wav2Vec2ForCTC model."""
    if model_id not in _ctc_models:
        revision = model_revision(model_id)
        processor = Wav2Vec2Processor.from_pretrained(model_id, revision=revision)
        model = Wav2Vec2ForCTC.from_pretrained(model_id, revision=revision)
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
        model = Wav2Vec2Model.from_pretrained(model_id, revision=model_revision(model_id))
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
