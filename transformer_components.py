"""Transformer-backed sequence tagging components for spaCy.

This module registers BiLSTM, BiLSTM-CRF, and CRF NER components that
consume embeddings from ``TransformerListener`` instances so gradients can
flow back into the shared transformer during training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from wasabi import msg
from spacy.language import Language
from spacy.training import biluo_tags_to_offsets, offsets_to_biluo_tags
from spacy import util

try:  # pragma: no cover - import guard for optional torch stack
    import torch
    import torch.nn as nn
    from torchcrf import CRF

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully at runtime
    TORCH_AVAILABLE = False
    msg.warn("PyTorch with torchcrf is required for transformer sequence taggers.")

if TORCH_AVAILABLE:
    import numpy
    from spacy.pipeline import TrainablePipe
    from spacy.tokens import Span
    from thinc.api import PyTorchWrapper
    from spacy_transformers.layers.listener import TransformerListener

    def _collect_transformer_listeners(model) -> List[TransformerListener]:
        listeners: List[TransformerListener] = []
        if model is None:
            return listeners
        if hasattr(model, "walk"):
            for node in model.walk():
                if isinstance(node, TransformerListener):
                    listeners.append(node)
        elif isinstance(model, TransformerListener):
            listeners.append(model)
        return listeners

    class BiLSTM(nn.Module):
        """BiLSTM tagger without CRF."""

        def __init__(self, input_dim: int, hidden_dim: int, num_labels: int, dropout: float = 0.3):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_labels = num_labels

            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)

        def forward(self, embeddings, labels=None, mask=None):  # type: ignore[override]
            lstm_out, _ = self.lstm(embeddings)
            lstm_out = self.dropout(lstm_out)
            logits = self.hidden2tag(lstm_out)

            if labels is not None:
                labels = torch.clamp(labels, 0, self.num_labels - 1)
                logits_flat = logits.view(-1, self.num_labels)
                labels_flat = labels.view(-1)

                if mask is not None:
                    mask_flat = mask.view(-1)
                    logits_flat = logits_flat[mask_flat]
                    labels_flat = labels_flat[mask_flat]

                return nn.functional.cross_entropy(logits_flat, labels_flat)

            predictions = torch.argmax(logits, dim=-1)
            return predictions.tolist()

    class CRFOnly(nn.Module):
        """CRF tagger that consumes embeddings directly."""

        def __init__(self, input_dim: int, num_labels: int, dropout: float = 0.3):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = 0
            self.num_labels = num_labels

            self.dropout = nn.Dropout(dropout)
            self.hidden2tag = nn.Linear(input_dim, num_labels)
            self.crf = CRF(num_labels, batch_first=True)

        def forward(self, embeddings, labels=None, mask=None):  # type: ignore[override]
            emissions = self.hidden2tag(self.dropout(embeddings))

            if labels is not None:
                labels = torch.clamp(labels, 0, self.num_labels - 1)
                if mask is None:
                    mask = torch.ones_like(labels, dtype=torch.bool)
                loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
                return loss

            if mask is None:
                mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
            decoded = self.crf.decode(emissions, mask=mask)
            return decoded

    class BiLSTMCRF(nn.Module):
        """BiLSTM encoder followed by a CRF decoder."""

        def __init__(self, input_dim: int, hidden_dim: int, num_labels: int, dropout: float = 0.3):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_labels = num_labels

            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
            self.crf = CRF(num_labels, batch_first=True)

        def forward(self, embeddings, labels=None, mask=None):  # type: ignore[override]
            lstm_out, _ = self.lstm(embeddings)
            lstm_out = self.dropout(lstm_out)
            emissions = self.hidden2tag(lstm_out)

            if labels is not None:
                labels = torch.clamp(labels, 0, self.num_labels - 1)
                if mask is None:
                    mask = torch.ones_like(labels, dtype=torch.bool)
                loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
                return loss

            if mask is None:
                mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=embeddings.device)
            decoded = self.crf.decode(emissions, mask=mask)
            return decoded

    class TransformerSequenceTagger(TrainablePipe):
        """Trainable spaCy pipe that reads TransformerListener embeddings."""

        def __init__(
            self,
            vocab,
            model,
            name: str = "transformer_sequence_tagger",
            *,
            tok2vec=None,
            labels: Optional[List[str]] = None,
            mode: str = "bilstm-crf",
            transformer_pipe=None,
        ) -> None:
            self.vocab = vocab
            self.model = model
            self.name = name
            self.cfg = {}
            self.tok2vec = tok2vec
            self.mode = mode
            self.use_crf = mode in {"bilstm-crf", "crf"}
            self.transformer_pipe = transformer_pipe
            self._logged_listener_debug = False

            self._ensure_tok2vec_linked()

            self._label_map = {}
            self._idx_to_label = {}
            self._repair_count = 0
            self._max_repair_logs = 5

            if labels:
                self._initialize_labels(labels)
        def _ensure_tok2vec_linked(self) -> None:
            """Expose the tok2vec listener chain via the thinc model tree.

            spaCy's transformer component discovers downstream listeners by
            walking the ``component.model`` tree. Our architecture keeps the
            ``TransformerListener`` inside ``self.tok2vec`` instead of the
            wrapped PyTorch model, so we append the tok2vec chain to the
            wrapper's ``layers`` once to make it discoverable.
            """

            if self.tok2vec is None:
                return
            if not hasattr(self.model, "layers"):
                return
            if self.tok2vec in self.model.layers:
                return
            self.model.layers.append(self.tok2vec)

        # ------------------------------------------------------------------
        # Label helpers
        # ------------------------------------------------------------------
        def _initialize_labels(self, labels: List[str]) -> None:
            biluo_labels = ["O"]
            for label in sorted(labels):
                biluo_labels.extend([f"B-{label}", f"I-{label}", f"L-{label}", f"U-{label}"])

            self._label_map = {label: idx for idx, label in enumerate(biluo_labels)}
            self._idx_to_label = {idx: label for label, idx in self._label_map.items()}
            msg.info(
                f"Initialized {len(biluo_labels)} BILUO labels for {len(labels)} entity types"
            )

        @property
        def labels(self) -> tuple:
            entity_labels = set()
            for label in self._label_map:
                if label != "O" and "-" in label:
                    entity_labels.add(label.split("-", 1)[1])
            return tuple(sorted(entity_labels))

        def _spans_to_biluo(self, doc) -> List[int]:
            entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
            biluo_tags = offsets_to_biluo_tags(doc, entities)
            return [self._label_map.get(tag, 0) for tag in biluo_tags]

        def _repair_biluo(self, tags: List[str]) -> List[str]:
            repaired = list(tags)
            for i, tag in enumerate(repaired):
                if tag == "O" or "-" not in tag:
                    continue
                prefix, label = tag.split("-", 1)
                if prefix == "I":
                    if i == 0 or not repaired[i - 1].endswith(f"-{label}") or not repaired[i - 1].startswith(("B-", "I-")):
                        repaired[i] = f"B-{label}"
                elif prefix == "L":
                    if i == 0 or not repaired[i - 1].endswith(f"-{label}") or not repaired[i - 1].startswith(("B-", "I-")):
                        repaired[i] = f"U-{label}"
            return repaired

        def _biluo_to_spans(self, doc, tag_indices: List[int]) -> List[Span]:
            biluo_tags = [self._idx_to_label.get(idx, "O") for idx in tag_indices]
            repaired_tags = self._repair_biluo(biluo_tags)

            if repaired_tags != biluo_tags and self._repair_count < self._max_repair_logs:
                self._repair_count += 1
                msg.warn("Invalid BILUO sequence detected and repaired.")

            entities = biluo_tags_to_offsets(doc, repaired_tags)
            spans = []
            for start_char, end_char, label in entities:
                span = doc.char_span(start_char, end_char, label=label)
                if span is not None:
                    spans.append(span)
            return spans

        # ------------------------------------------------------------------
        # Inference utilities
        # ------------------------------------------------------------------
        def predict(self, docs):
            if not docs:
                return []

            pytorch_model = self.model.attrs["pytorch_model"]
            pytorch_model.eval()
            device = next(pytorch_model.parameters()).device

            tokvecs_output = self.tok2vec.predict(docs)
            if hasattr(tokvecs_output, "__iter__") and not hasattr(tokvecs_output, "shape"):
                tokvecs = tokvecs_output
            else:
                tokvecs = [tokvecs_output]

            predictions = []
            with torch.no_grad():
                for doc, tokvec_array in zip(docs, tokvecs):
                    if hasattr(tokvec_array, "get"):
                        tokvec_np = tokvec_array.get()
                    else:
                        tokvec_np = numpy.asarray(tokvec_array)

                    if tokvec_np.ndim == 3 and tokvec_np.shape[0] == 1:
                        tokvec_np = numpy.squeeze(tokvec_np, axis=0)

                    seq_len = min(len(doc), tokvec_np.shape[0])
                    if seq_len == 0:
                        predictions.append([])
                        continue

                    embeddings = torch.tensor(tokvec_np[:seq_len], dtype=torch.float32, device=device).unsqueeze(0)
                    decoded = pytorch_model(embeddings)

                    if isinstance(decoded, torch.Tensor):
                        decoded_list = numpy.asarray(decoded.squeeze(0).cpu()).tolist()
                        predictions.append([int(x) for x in decoded_list])
                    elif isinstance(decoded, list):
                        if decoded and isinstance(decoded[0], (list, tuple)):
                            predictions.append([int(x) for x in decoded[0]])
                        elif decoded and isinstance(decoded[0], (int, numpy.integer)):
                            predictions.append([int(x) for x in decoded])
                        else:
                            predictions.append([0] * seq_len)
                    else:
                        predictions.append([0] * seq_len)

            return predictions

        def set_annotations(self, docs, predictions) -> None:
            for doc, tags in zip(docs, predictions):
                spans = self._biluo_to_spans(doc, tags)
                if not spans:
                    doc.ents = ()
                    continue

                ents = [Span(doc, span.start, span.end, label=span.label_) for span in spans]
                try:
                    doc.ents = ents
                except ValueError:
                    doc.ents = util.filter_spans(ents)

        def score(self, examples, **kwargs):
            from spacy.scorer import Scorer

            scorer = Scorer()
            return scorer.score_spans(examples, "ents", **kwargs)

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        def update(self, examples, *, drop=0.0, sgd=None, losses=None):
            if losses is None:
                losses = {}
            losses.setdefault(self.name, 0.0)

            if not examples:
                return losses

            if (
                self.transformer_pipe is not None
                and not self._logged_listener_debug
            ):
                listeners = self.transformer_pipe.listener_map.get(self.name, [])
                msg.info(
                    f"[{self.name}] transformer listeners registered={len(listeners)}; total={len(self.transformer_pipe.listeners)}"
                )
                if not listeners:
                    msg.warn(f"[{self.name}] No listeners registered with transformer pipe.")
                self._logged_listener_debug = True

            pytorch_model = self.model.attrs["pytorch_model"]
            device = next(pytorch_model.parameters()).device
            pytorch_model.train()

            if not hasattr(self, "_optimizer"):
                self._optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)

            if self.transformer_pipe is not None:
                listeners = _collect_transformer_listeners(self.tok2vec)
                for listener in listeners:
                    self.transformer_pipe.add_listener(listener, self.name)

            docs = [eg.predicted for eg in examples]
            gold_docs = [eg.reference for eg in examples]

            if (
                self.transformer_pipe is not None
                and not self.transformer_pipe.listener_map.get(self.name)
            ):
                msg.warn(
                    f"[{self.name}] No transformer listeners detected before begin_update; forcing transformer predictions as fallback"
                )
                trf_batch = self.transformer_pipe.predict(docs)
                self.transformer_pipe.set_annotations(docs, trf_batch)

            tokvecs, bp_tokvecs = self.tok2vec.begin_update(docs)
            if hasattr(tokvecs, "__iter__") and not hasattr(tokvecs, "shape"):
                raw_tokvecs = list(tokvecs)
            else:
                raw_tokvecs = [tokvecs]

            arrays = []
            array_types = []
            for arr in raw_tokvecs:
                if hasattr(arr, "get"):
                    arrays.append(arr.get())
                    array_types.append("cupy")
                else:
                    arrays.append(numpy.asarray(arr))
                    array_types.append("numpy")

            tokvec_gradients = []
            total_loss = 0.0
            valid_docs = 0
            self._optimizer.zero_grad()

            for doc_idx, (doc, tokvec_np, gold_doc) in enumerate(zip(docs, arrays, gold_docs)):
                gold_biluo = self._spans_to_biluo(gold_doc)
                base_array = tokvec_np
                if base_array.ndim == 3 and base_array.shape[0] == 1:
                    base_array = numpy.squeeze(base_array, axis=0)

                seq_len = min(len(doc), base_array.shape[0], len(gold_biluo))
                if seq_len == 0:
                    tokvec_gradients.append(numpy.zeros_like(tokvec_np))
                    continue

                embeddings = torch.tensor(
                    base_array[:seq_len], dtype=torch.float32, device=device, requires_grad=True
                ).unsqueeze(0)
                embeddings.retain_grad()

                labels = torch.tensor(gold_biluo[:seq_len], dtype=torch.long, device=device).unsqueeze(0)
                labels = torch.clamp(labels, 0, len(self._label_map) - 1)
                mask = torch.ones_like(labels, dtype=torch.bool)

                loss = pytorch_model(embeddings, labels, mask=mask)  # mask ignored for BiLSTM
                loss.backward()

                if embeddings.grad is not None:
                    grad_np = embeddings.grad.squeeze(0).detach().cpu().numpy()
                    full_grad = numpy.zeros_like(tokvec_np)
                    if full_grad.ndim == 3 and full_grad.shape[0] == 1:
                        full_grad[0, :seq_len] = grad_np
                    else:
                        full_grad[:seq_len] = grad_np
                    tokvec_gradients.append(full_grad)
                else:
                    tokvec_gradients.append(numpy.zeros_like(tokvec_np))

                total_loss += loss.item()
                valid_docs += 1

            if valid_docs > 0:
                self._optimizer.step()
                losses[self.name] += total_loss / valid_docs

            if tokvec_gradients and valid_docs > 0 and bp_tokvecs is not None:
                grad_arrays = []
                for idx, grad in enumerate(tokvec_gradients):
                    if array_types[idx] == "cupy":
                        try:
                            import cupy

                            grad_arrays.append(cupy.asarray(grad))
                        except ImportError:
                            grad_arrays.append(grad)
                    else:
                        grad_arrays.append(grad)

                # Apply gradients back to transformer listener chain
                bp_tokvecs(grad_arrays)

                # Fallback: if the listener chain didn't propagate to the
                # transformer (no listeners registered), call into the
                # transformer pipe explicitly so that it updates its cached
                # embeddings and avoids E203. This keeps fine-tuning active
                # when listeners work, but bypasses the value error when they
                # don't.
                if (
                    self.transformer_pipe is not None
                    and not self.transformer_pipe.listener_map.get(self.name)
                ):
                    msg.warn(
                        f"[{self.name}] Transformer listeners missing during backprop; calling transformer.update() directly as fallback"
                    )
                    self.transformer_pipe.update(examples, drop=drop, sgd=sgd, losses=losses)

                if sgd is not None:
                    self.tok2vec.finish_update(sgd)

            return losses

        # ------------------------------------------------------------------
        # Lifecycle hooks
        # ------------------------------------------------------------------
        def initialize(self, get_examples, *, nlp=None, labels=None):
            if labels is not None:
                self._initialize_labels(labels)
            else:
                entity_labels = set()
                for example in get_examples():
                    for ent in example.reference.ents:
                        entity_labels.add(ent.label_)
                self._initialize_labels(sorted(entity_labels))

            num_labels = len(self._label_map)
            if num_labels == 0:
                return

            old_model = self.model.attrs.get("pytorch_model")
            if old_model is None:
                raise ValueError("PyTorch model not initialised.")

            input_dim = old_model.input_dim
            hidden_dim = getattr(old_model, "hidden_dim", 0)
            dropout = getattr(old_model, "dropout", nn.Dropout(0.3)).p
            device = next(old_model.parameters()).device

            if self.mode == "bilstm-crf":
                new_model = BiLSTMCRF(input_dim, hidden_dim, num_labels, dropout)
            elif self.mode == "bilstm":
                new_model = BiLSTM(input_dim, hidden_dim, num_labels, dropout)
            else:
                new_model = CRFOnly(input_dim, num_labels, dropout)

            new_model.to(device)
            self.model.attrs["pytorch_model"] = new_model
            if hasattr(self.model, "_model"):
                self.model._model = new_model
            msg.good(f"Model initialised with {num_labels} labels")

            # At initialization time (after the full pipeline is created),
            # ensure any TransformerListener instances inside our tok2vec
            # chain are attached to the runtime transformer pipe. This is the
            # reliable place to wire listeners to the actual Transformer
            # component that will be used during training.
            if nlp is not None and self.transformer_pipe is None and nlp.has_pipe("transformer"):
                try:
                    self.transformer_pipe = nlp.get_pipe("transformer")
                except Exception:
                    self.transformer_pipe = None

            if self.transformer_pipe is not None:
                listeners = _collect_transformer_listeners(self.tok2vec)
                if listeners:
                    msg.info(f"[{self.name}] initialize: attaching {len(listeners)} listener(s) to transformer")
                for listener in listeners:
                    self.transformer_pipe.add_listener(listener, self.name)

            # Re-register transformer listeners after initialization to ensure the
            # transformer component is aware of the downstream listener chain.
            if self.transformer_pipe is not None:
                listeners = _collect_transformer_listeners(self.tok2vec)
                if listeners:
                    for listener in listeners:
                        self.transformer_pipe.add_listener(listener, self.name)
                    msg.info(
                        f"[{self.name}] ensured {len(listeners)} transformer listener(s) registered post-initialize"
                    )
                else:
                    msg.warn(f"[{self.name}] no transformer listeners found during initialize")
                # allow debug logging during update to report current registrations again
                self._logged_listener_debug = False

        def to_disk(self, path, *, exclude=tuple()):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            with (path / "labels.json").open("w", encoding="utf8") as f:
                json.dump(
                    {
                        "label_map": self._label_map,
                        "idx_to_label": self._idx_to_label,
                        "mode": self.mode,
                    },
                    f,
                    indent=2,
                )

            pytorch_model = self.model.attrs["pytorch_model"]
            torch.save(pytorch_model.state_dict(), path / "model.pt")

            if hasattr(self, "_optimizer"):
                torch.save(self._optimizer.state_dict(), path / "optimizer.pt")

        def from_disk(self, path, *, exclude=tuple()):
            path = Path(path)

            with (path / "labels.json").open("r", encoding="utf8") as f:
                label_data = json.load(f)
                self._label_map = {
                    k: int(v) if isinstance(v, str) and v.isdigit() else v
                    for k, v in label_data["label_map"].items()
                }
                self._idx_to_label = {int(k): v for k, v in label_data["idx_to_label"].items()}
                self.mode = label_data.get("mode", self.mode)
                self.use_crf = self.mode in {"bilstm-crf", "crf"}

            num_labels = len(self._label_map)
            old_model = self.model.attrs.get("pytorch_model")
            if old_model is None:
                raise ValueError("PyTorch model not initialised.")

            input_dim = old_model.input_dim
            hidden_dim = getattr(old_model, "hidden_dim", 0)
            dropout = getattr(old_model, "dropout", nn.Dropout(0.3)).p
            device = next(old_model.parameters()).device

            if self.mode == "bilstm-crf":
                pytorch_model = BiLSTMCRF(input_dim, hidden_dim, num_labels, dropout)
            elif self.mode == "bilstm":
                pytorch_model = BiLSTM(input_dim, hidden_dim, num_labels, dropout)
            else:
                pytorch_model = CRFOnly(input_dim, num_labels, dropout)

            pytorch_model.to(device)
            pytorch_model.load_state_dict(torch.load(path / "model.pt", map_location=device))
            self.model.attrs["pytorch_model"] = pytorch_model
            if hasattr(self.model, "_model"):
                self.model._model = pytorch_model

            if (path / "optimizer.pt").exists() and hasattr(self, "_optimizer"):
                self._optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location=device))

            return self

    def _ensure_listener_dim(tok2vec, width: int) -> None:
        if width <= 0:
            return
        if hasattr(tok2vec, "set_dim"):
            try:
                tok2vec.set_dim("nO", width)
            except (KeyError, ValueError):
                pass
        if hasattr(tok2vec, "layers"):
            for layer in tok2vec.layers:
                if hasattr(layer, "set_dim"):
                    try:
                        layer.set_dim("nO", width)
                    except (KeyError, ValueError):
                        pass
                listener = getattr(layer, "listener", None)
                if listener is not None and hasattr(listener, "set_dim"):
                    try:
                        listener.set_dim("nO", width)
                    except (KeyError, ValueError):
                        pass

    def _resolve_input_dim(tok2vec) -> int:
        if hasattr(tok2vec, "get_dim"):
            try:
                return tok2vec.get_dim("nO")
            except (KeyError, ValueError):
                pass
        if hasattr(tok2vec, "layers"):
            for layer in reversed(tok2vec.layers):
                if hasattr(layer, "get_dim"):
                    try:
                        return layer.get_dim("nO")
                    except (KeyError, ValueError):
                        continue
                if hasattr(layer, "listener") and hasattr(layer.listener, "get_dim"):
                    try:
                        return layer.listener.get_dim("nO")
                    except (KeyError, ValueError):
                        continue
        return 768

    def _ensure_labels(labels: Optional[List[str]]) -> List[str]:
        return list(labels) if labels else []


    def _collect_transformer_listeners(model) -> List[TransformerListener]:
        listeners: List[TransformerListener] = []
        if model is None:
            return listeners
        if hasattr(model, "walk"):
            for node in model.walk():
                if isinstance(node, TransformerListener):
                    listeners.append(node)
        elif isinstance(model, TransformerListener):
            listeners.append(model)
        return listeners


    @Language.factory(
        "bilstm",
        default_config={
            "hidden_dim": 256,
            "dropout": 0.3,
            "labels": None,
            "tok2vec": {
                "@architectures": "spacy-transformers.TransformerListener.v1",
                "grad_factor": 1.0,
                "pooling": {"@layers": "reduce_mean.v1"},
                "upstream": "*",
            },
        },
    )
    def make_bilstm(nlp, name, hidden_dim, dropout, labels, tok2vec):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and torchcrf must be installed to use bilstm_ner_trf.")

        transformer_pipe = nlp.get_pipe("transformer") if nlp.has_pipe("transformer") else None
        entity_labels = _ensure_labels(labels)
        num_labels = 1 + (len(entity_labels) * 4) if entity_labels else 1
        input_dim = _resolve_input_dim(tok2vec)
        _ensure_listener_dim(tok2vec, input_dim)

        pytorch_model = BiLSTM(input_dim, hidden_dim, num_labels, dropout)
        wrapper = PyTorchWrapper(pytorch_model)
        wrapper.attrs["pytorch_model"] = pytorch_model

        component = TransformerSequenceTagger(
            nlp.vocab,
            wrapper,
            name=name,
            tok2vec=tok2vec,
            labels=entity_labels,
            mode="bilstm",
            transformer_pipe=transformer_pipe,
        )

        if transformer_pipe is not None:
            listeners = _collect_transformer_listeners(component.tok2vec)
            for listener in listeners:
                transformer_pipe.add_listener(listener, component.name)

        return component


    @Language.factory(
        "bilstm-crf",
        default_config={
            "hidden_dim": 256,
            "dropout": 0.3,
            "labels": None,
            "tok2vec": {
                "@architectures": "spacy-transformers.TransformerListener.v1",
                "grad_factor": 1.0,
                "pooling": {"@layers": "reduce_mean.v1"},
                "upstream": "*",
            },
        },
    )
    def make_bilstm_crf(nlp, name, hidden_dim, dropout, labels, tok2vec):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and torchcrf must be installed to use bilstm_crf_ner_trf.")

        transformer_pipe = nlp.get_pipe("transformer") if nlp.has_pipe("transformer") else None
        entity_labels = _ensure_labels(labels)
        num_labels = 1 + (len(entity_labels) * 4) if entity_labels else 1
        input_dim = _resolve_input_dim(tok2vec)
        _ensure_listener_dim(tok2vec, input_dim)

        pytorch_model = BiLSTMCRF(input_dim, hidden_dim, num_labels, dropout)
        wrapper = PyTorchWrapper(pytorch_model)
        wrapper.attrs["pytorch_model"] = pytorch_model

        component = TransformerSequenceTagger(
            nlp.vocab,
            wrapper,
            name=name,
            tok2vec=tok2vec,
            labels=entity_labels,
            mode="bilstm-crf",
            transformer_pipe=transformer_pipe,
        )

        if transformer_pipe is not None:
            listeners = _collect_transformer_listeners(component.tok2vec)
            msg.info(f"[{name}] Attaching {len(listeners)} transformer listener(s)")
            for listener in listeners:
                transformer_pipe.add_listener(listener, component.name)

        return component


    @Language.factory(
        "crf",
        default_config={
            "dropout": 0.3,
            "labels": None,
            "tok2vec": {
                "@architectures": "spacy-transformers.TransformerListener.v1",
                "grad_factor": 1.0,
                "pooling": {"@layers": "reduce_mean.v1"},
                "upstream": "*",
            },
        },
    )
    def make_crf(nlp, name, dropout, hidden_dim, labels, tok2vec): # only include hidden_dim for compatibility
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and torchcrf must be installed to use crf_ner_trf.")

        transformer_pipe = nlp.get_pipe("transformer") if nlp.has_pipe("transformer") else None
        entity_labels = _ensure_labels(labels)
        num_labels = 1 + (len(entity_labels) * 4) if entity_labels else 1
        input_dim = _resolve_input_dim(tok2vec)
        _ensure_listener_dim(tok2vec, input_dim)

        pytorch_model = CRFOnly(input_dim, num_labels, dropout)
        wrapper = PyTorchWrapper(pytorch_model)
        wrapper.attrs["pytorch_model"] = pytorch_model

        component = TransformerSequenceTagger(
            nlp.vocab,
            wrapper,
            name=name,
            tok2vec=tok2vec,
            labels=entity_labels,
            mode="crf",
            transformer_pipe=transformer_pipe,
        )

        if transformer_pipe is not None:
            listeners = _collect_transformer_listeners(component.tok2vec)
            msg.info(f"[{name}] Attaching {len(listeners)} transformer listener(s)")
            for listener in listeners:
                transformer_pipe.add_listener(listener, component.name)

        return component
else:

    @Language.factory("bilstm")
    def _missing_bilstm_ner_trf(*args, **kwargs):  # pragma: no cover - torch missing path
        raise ImportError("PyTorch and torchcrf must be installed to use bilstm_ner_trf.")

    @Language.factory("bilstm-crf")
    def _missing_bilstm_crf_ner_trf(*args, **kwargs):  # pragma: no cover
        raise ImportError("PyTorch and torchcrf must be installed to use bilstm_crf_ner_trf.")

    @Language.factory("crf")
    def _missing_crf_ner_trf(*args, **kwargs):  # pragma: no cover
        raise ImportError("PyTorch and torchcrf must be installed to use crf_ner_trf.")

__all__ = ["TORCH_AVAILABLE"]
