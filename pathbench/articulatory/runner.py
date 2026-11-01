"""Articulatory-inversion inference wrapper.

Loads the upstream ``articulatory/articulatory`` (Wu et al., ICASSP 2023,
arXiv:2302.06774) speech-to-EMA model once and exposes
:meth:`ArticulatoryRunner.extract_ema` which maps a 16 kHz waveform to a
12-channel EMA trajectory at 100 Hz.

Pipeline (matches ``egs/ema/voc1/local/predict_ema.py`` in the upstream repo,
substituting HuggingFace's ``HubertModel`` for ``s3prl.hub.hubert_large_ll60k``
to keep dependencies light — both wrap the same Facebook HuBERT-Large-LL60K
checkpoint and the last-layer hidden states are the relevant signal):

1. 16 kHz waveform -> HuBERT-Large encoder -> ``(T_50, 1024)`` at 50 Hz.
2. 2x linear interpolation along time -> ``(T_100, 1024)`` at 100 Hz.
3. ``BiGRU.inference(feat, normalize_before=False)`` -> ``(T_100, 53)``.
4. Slice ``[:, :12]`` -> EMA at 100 Hz.

The 12 channels are anatomical (x, y) coordinates for, in order: lower
incisor, upper lip, lower lip, tongue tip, tongue body, tongue dorsum.

The ``articulatory.*`` code the BiGRU pipeline needs (``load_model``,
``BiGRU``, and the handful of layers they import) is vendored in-repo under
``pathbench/third_party/articulatory`` — a trimmed copy that drops the
upstream vocoders (HiFi-GAN, MelGAN, ...) and ESPnet nets PathBench never
uses. Those imports plus the HuggingFace ``HubertModel`` are deferred to
:meth:`load` so this module imports cleanly for static analysis and
audio-only evaluations.
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# Vendored minimal ``articulatory`` package lives here. Putting this dir on
# ``sys.path`` lets the upstream-style ``import articulatory.*`` in the vendored
# code (and in :meth:`load`) resolve without the full external clone under
# ``tools/articulatory``.
_VENDOR_ROOT = Path(__file__).resolve().parent / "third_party"


# Released checkpoint name; the speech-to-EMA Google Drive folder unpacks to
# this dir under ``checkpoints/`` inside ``tools/articulatory``.
DEFAULT_CHECKPOINT_SUBDIR = "checkpoints/hprc_no_m1f2_h2emaph_gru_joint_nogan_model"
DEFAULT_HUBERT_ID = "facebook/hubert-large-ll60k"

# Output channel layout of the BiGRU. First 12 are EMA; the remaining 41 are
# pitch / phoneme auxiliaries we ignore.
EMA_CHANNELS = 12
# 6 articulators x (x, y), in the order produced by the released checkpoint.
ARTICULATOR_NAMES = (
    "lower_incisor", "upper_lip", "lower_lip",
    "tongue_tip", "tongue_body", "tongue_dorsum",
)


class ArticulatoryRunner:
    """Loads the HuBERT + BiGRU pipeline once and exposes a single
    :meth:`extract_ema` call."""

    def __init__(self, repo_path: str,
                 checkpoint_subdir: str = DEFAULT_CHECKPOINT_SUBDIR,
                 hubert_id: str = DEFAULT_HUBERT_ID,
                 device: Optional[str] = None,
                 inversion_ckpt: Optional[str] = None,
                 ssl_kind: str = "hubert"):
        # ``repo_path`` is the parent of ``checkpoints/`` where the released
        # BiGRU weights + config live (e.g. ``tools/articulatory``). The
        # ``articulatory.*`` code itself is vendored in-repo (see
        # ``_VENDOR_ROOT``), so this no longer needs to be an importable clone.
        self.repo = Path(repo_path).resolve()
        self.checkpoint_dir = self.repo / checkpoint_subdir
        self.hubert_id = hubert_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Custom-backend overrides: when ``inversion_ckpt`` is given, load a
        # locally-trained BiGRU (``best.pkl`` from aai/train.py) on top of the
        # ``ssl_kind`` backbone instead of the released checkpoint. ``ssl_kind``
        # is "hubert" (hubert-large last layer) or "w2v10" (wav2vec2-large
        # hidden_states[10]). Released path is unchanged when ckpt is None.
        self.inversion_ckpt = inversion_ckpt
        self.ssl_kind = ssl_kind
        self._custom = inversion_ckpt is not None
        self._hubert = None
        self._hubert_processor = None
        self._ssl_model = None       # custom-backend SSL encoder
        self._ema_mean = None        # custom-backend denorm stats
        self._ema_std = None
        self._inv_model = None
        self._inv_config = None
        self._interp_factor = None

    def _ensure_loaded(self):
        if self._inv_model is None:
            self.load()

    def load(self) -> None:
        """Load HuBERT feature extractor + BiGRU inversion checkpoint."""
        if self._inv_model is not None:
            return

        if str(_VENDOR_ROOT) not in sys.path:
            sys.path.insert(0, str(_VENDOR_ROOT))

        if self._custom:
            self._load_custom()
            return

        import yaml
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
        from articulatory.utils import load_model

        # ---- HuBERT-Large (50 Hz, 1024-D last hidden state) ----
        self._hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained(self.hubert_id)
        self._hubert = HubertModel.from_pretrained(self.hubert_id).to(self.device).eval()

        # ---- BiGRU inversion (HuBERT 50 Hz -> EMA 100 Hz) ----
        ckpt = self.checkpoint_dir / "best_mel_ckpt.pkl"
        cfg = self.checkpoint_dir / "config.yml"
        with open(cfg) as f:
            self._inv_config = yaml.load(f, Loader=yaml.Loader)

        # Mirror predict_ema.py's exp_id-prefix heuristic: ``hprc_*`` checkpoints
        # use 2x linear interpolation (HuBERT 50 Hz -> 100 Hz BiGRU input),
        # everything else uses 4x (-> 200 Hz). The released speech-to-EMA model
        # is ``hprc_no_m1f2_h2emaph_gru_joint_nogan_model`` => factor 2.
        exp_id = self._inv_config.get("exp_id", self.checkpoint_dir.name)
        self._interp_factor = 2 if exp_id.startswith("hprc") else 4

        inv = load_model(str(ckpt), self._inv_config)
        inv.remove_weight_norm()
        self._inv_model = inv.eval().to(self.device)

    def _load_custom(self) -> None:
        """Load a locally-trained BiGRU (aai/train.py ``best.pkl``) on top of
        the ``ssl_kind`` backbone. ``best.pkl`` holds ``{"model": state_dict,
        "emean", "estd"}``; out_channels (12, or 12+n_phones for MTL) is
        inferred from the ``fc2`` output weight. Features are extracted at
        50 Hz then 2x linear-interpolated to 100 Hz, matching aai/features.py.
        """
        import torch as _torch
        from articulatory.models import BiGRU

        if self.ssl_kind == "hubert":
            from transformers import HubertModel
            self._ssl_model = HubertModel.from_pretrained(
                "facebook/hubert-large-ll60k").to(self.device).eval()
            self._ssl_layer = -1
        elif self.ssl_kind == "w2v10":
            from transformers import Wav2Vec2Model
            self._ssl_model = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-large").to(self.device).eval()
            self._ssl_layer = 10
        else:
            raise ValueError(f"Unknown ssl_kind: {self.ssl_kind}")

        # weights_only=False: checkpoint holds numpy emean/estd arrays. These
        # are our own locally-trained files (aai/train.py), so trusted.
        ck = _torch.load(self.inversion_ckpt, map_location=self.device,
                         weights_only=False)
        state = ck["model"]
        # Head is either a plain Linear (``fc2.weight``, use_tanh=False, EMA
        # checkpoints) or Linear+Tanh (``fc2.0.weight``, use_tanh=True, the TV
        # checkpoints). Detect which so the architecture matches the state dict.
        if "fc2.weight" in state:
            out_ch, use_tanh = state["fc2.weight"].shape[0], False
        elif "fc2.0.weight" in state:
            out_ch, use_tanh = state["fc2.0.weight"].shape[0], True
        else:
            raise KeyError("checkpoint has no fc2 output weight (fc2.weight / fc2.0.weight)")
        inv = BiGRU(in_channels=1024, hidden_size=256, dropout=0.3,
                    out_channels=out_ch, use_tanh=use_tanh)
        inv.load_state_dict(state)
        self._inv_model = inv.eval().to(self.device)
        self._out_ch = out_ch
        self._interp_factor = 2
        # EMA width = first min(out_ch, 12) channels. In every BiGRU we use the
        # EMA is the LEADING block of fc2 outputs, so the leading slice is exact:
        #   pure (out_ch=12)        -> 12  (all EMA)
        #   MTL  (out_ch=12+phones) -> 12  (drops the trailing phoneme logits)
        #   released (out_ch=53)    -> 12  (drops trailing aux channels)
        #   NKI per-speaker (out_ch=10, different articulator set, no head) -> 10
        # WARNING: this heuristic assumes any aux/phoneme head sits ABOVE a
        # >=12-wide EMA block. A model with EMA width < 12 AND a phoneme head
        # (e.g. a hypothetical NKI-style 10-ch corpus trained MTL, out_ch=10+P>12)
        # would mis-slice phoneme logits as EMA channels 10,11. No such checkpoint
        # exists; if one is ever trained, pass the EMA width in explicitly.
        self._n_ema = min(out_ch, EMA_CHANNELS)
        # Global de-norm stats are optional: the NKI checkpoints normalize EMA
        # per-speaker (spk_stats.npz) and ship no global emean/estd. Per-channel
        # z-scoring in the ART-NAD evaluator washes out any affine de-norm, so
        # an identity de-norm (0/1) is exact for DTW.
        emean, estd = ck.get("emean"), ck.get("estd")
        self._ema_mean = (np.asarray(emean, dtype=np.float32)
                          if emean is not None else np.float32(0.0))
        self._ema_std = (np.asarray(estd, dtype=np.float32)
                         if estd is not None else np.float32(1.0))

    def _ssl_feats_custom(self, audio: np.ndarray) -> "torch.Tensor":
        """Custom backend: 16 kHz waveform -> chosen SSL hidden state at 50 Hz,
        ``(T_50, 1024)``. Per-utterance zero-mean/unit-var waveform norm matches
        aai/features.py (the recipe these checkpoints were trained on)."""
        a = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        x = torch.from_numpy(a).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self._ssl_model(x, output_hidden_states=True)
        return out.hidden_states[self._ssl_layer].squeeze(0)  # (T_50, 1024)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def extract_ema(self, audio: np.ndarray) -> np.ndarray:
        """16 kHz mono float waveform -> ``(T, 12)`` float32 EMA at
        ``50 * interp_factor`` Hz (100 Hz for the released checkpoint).

        Raises :class:`ValueError` if the resulting trajectory is too short
        to be useful (< 2 frames); callers must handle that as a skip.
        """
        self._ensure_loaded()
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if self._custom:
            return self._extract_ema_custom(audio)

        # ---- HuBERT-Large hidden state @ 50 Hz ----
        # Wav2Vec2FeatureExtractor handles the layer-norm-style scaling the
        # upstream model was trained with (do_normalize=True for hubert-large).
        inputs = self._hubert_processor(
            audio, sampling_rate=16000, return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)
        with torch.no_grad():
            hubert_out = self._hubert(input_values).last_hidden_state  # (1, T_50, 1024)

        if hubert_out.size(1) < 1:
            raise ValueError(f"HuBERT produced 0 frames for {len(audio)} samples")

        # ---- Upsample 50 Hz -> 100 Hz (or 200 Hz) via linear interpolation ----
        # ``interpolate`` wants (N, C, T); HuBERT gives (N, T, C). Transpose
        # back after to match the BiGRU's ``inference(c)`` contract of (T, C).
        target_length = hubert_out.size(1) * self._interp_factor
        feat = torch.nn.functional.interpolate(
            hubert_out.transpose(1, 2),  # (1, 1024, T_50)
            size=target_length, mode="linear", align_corners=False,
        ).transpose(1, 2).squeeze(0)  # (T_target, 1024)

        # ---- BiGRU inversion: (T, 1024) -> (T, 53) -> slice first 12 ----
        with torch.no_grad():
            pred = self._inv_model.inference(feat, normalize_before=False)

        ema = pred[:, :EMA_CHANNELS].detach().cpu().numpy().astype(np.float32)
        if ema.shape[0] < 2:
            raise ValueError(
                f"Articulatory feature length {ema.shape[0]} < 2; cannot DTW"
            )
        return ema

    def _extract_ema_custom(self, audio: np.ndarray) -> np.ndarray:
        """Custom backend EMA: SSL feats @ 50 Hz -> 2x interp -> BiGRU ->
        slice 12 -> de-normalize with training EMA stats. Per-channel z-score
        in the ART-NAD evaluator makes the affine de-norm a no-op for DTW, but
        it is applied for unit-consistency with the released path."""
        feat50 = self._ssl_feats_custom(audio)  # (T_50, 1024) torch
        if feat50.size(0) < 1:
            raise ValueError(f"SSL produced 0 frames for {len(audio)} samples")
        target_length = feat50.size(0) * self._interp_factor
        feat = torch.nn.functional.interpolate(
            feat50.unsqueeze(0).transpose(1, 2),  # (1, 1024, T_50)
            size=target_length, mode="linear", align_corners=False,
        ).transpose(1, 2).squeeze(0)  # (T_100, 1024)
        with torch.no_grad():
            pred = self._inv_model.inference(feat, normalize_before=False)
        ema = pred[:, :self._n_ema].detach().cpu().numpy().astype(np.float32)
        ema = ema * self._ema_std + self._ema_mean
        if ema.shape[0] < 2:
            raise ValueError(
                f"Articulatory feature length {ema.shape[0]} < 2; cannot DTW"
            )
        return ema

    def extract_tv(self, audio: np.ndarray) -> np.ndarray:
        """16 kHz mono waveform -> ``(T, n_tv)`` tract-variable trajectory at
        100 Hz from a custom TV inversion checkpoint (e.g. tv_bigru_w2v10_sen,
        9 Seneviratne TVs). Returns all output channels (no EMA slicing); the
        per-utterance z-score in the evaluator handles scaling. Requires a
        custom checkpoint."""
        if not self._custom:
            raise RuntimeError("extract_tv requires a custom TV inversion checkpoint")
        self._ensure_loaded()
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        feat50 = self._ssl_feats_custom(audio)  # (T_50, 1024)
        if feat50.size(0) < 1:
            raise ValueError(f"SSL produced 0 frames for {len(audio)} samples")
        target_length = feat50.size(0) * self._interp_factor
        feat = torch.nn.functional.interpolate(
            feat50.unsqueeze(0).transpose(1, 2),  # (1, 1024, T_50)
            size=target_length, mode="linear", align_corners=False,
        ).transpose(1, 2).squeeze(0)  # (T_100, 1024)
        with torch.no_grad():
            pred = self._inv_model.inference(feat, normalize_before=False)
        tv = pred[:, :self._out_ch].detach().cpu().numpy().astype(np.float32)
        if tv.shape[0] < 2:
            raise ValueError(f"TV feature length {tv.shape[0]} < 2; cannot DTW")
        return tv

    def extract_hubert(self, audio: np.ndarray) -> np.ndarray:
        """16 kHz mono waveform -> HuBERT-Large last hidden state at 50 Hz,
        ``(T, 1024)`` float32. This is the same SSL representation the BiGRU
        consumes (before linear-interp upsample to 100 Hz and inversion).
        Exposed so that NAD-style metrics can reuse this exact backbone
        without loading a second HuBERT instance.
        """
        self._ensure_loaded()
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        inputs = self._hubert_processor(
            audio, sampling_rate=16000, return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)
        with torch.no_grad():
            return self._hubert(input_values).last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)

    def extract_ssl(self, audio: np.ndarray) -> np.ndarray:
        """16 kHz mono waveform -> configured-backbone SSL hidden state at 50 Hz,
        ``(T, 1024)`` float32. Returns the w2v10 layer-10 (or HuBERT last-layer)
        representation the BiGRU consumes, for fusion with the inverted EMA."""
        self._ensure_loaded()
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if self._custom:
            return self._ssl_feats_custom(audio).detach().cpu().numpy().astype(np.float32)
        return self.extract_hubert(audio)

    @property
    def framerate_hz(self) -> int:
        """Output EMA framerate. 100 Hz for the released ``hprc_*`` checkpoint."""
        self._ensure_loaded()
        return 50 * self._interp_factor
