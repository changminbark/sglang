"""End-to-end MLX model runner for Apple Silicon.

Runs the entire model within MLX, bypassing PyTorch MPS entirely.
"""

import logging
import time
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import load as mlx_lm_load
from mlx_lm.models.cache import (
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
    make_prompt_cache,
)

logger = logging.getLogger(__name__)


@dataclass
class MlxRequestState:
    """Per-request state for MLX inference."""

    token_ids: list[int]
    cache: list  # List of KVCache per layer
    generated_tokens: int = 0


# Named tuple-like container returned by the async start methods.
@dataclass
class MlxPendingDecode:
    """Lazy decode state to be finalized after mx.eval()."""

    lazy_tokens: mx.array
    batch_cache: list
    decode_reqs: list  # list[tuple[str, MlxRequestState]]


@dataclass
class MlxPendingPrefill:
    """Lazy prefill state to be finalized after mx.eval()."""

    lazy_token: mx.array
    cache: list
    req_id: str
    token_ids: list[int]


def _merge_kv_caches(
    caches_list: list[list],
) -> list:
    """Merge multiple per-request caches into batched caches."""
    if not caches_list:
        return []

    num_layers = len(caches_list[0])
    merged = []

    for layer_idx in range(num_layers):
        layer_caches = [caches[layer_idx] for caches in caches_list]
        if isinstance(layer_caches[0], KVCache):
            batch_cache = BatchKVCache.merge(layer_caches)
        elif isinstance(layer_caches[0], RotatingKVCache):
            batch_cache = BatchRotatingKVCache.merge(layer_caches)
        else:
            raise TypeError(f"Unsupported cache type: {type(layer_caches[0]).__name__}")
        merged.append(batch_cache)

    return merged


def _extract_kv_cache(batch_caches: list, idx: int) -> list:
    """Extract a single request's cache from batched caches.

    Works with both BatchKVCache (has .extract) and plain KVCache
    populated with batched data of shape (B, H, L, D).
    """
    extracted = []
    for cache in batch_caches:
        if hasattr(cache, "extract"):
            extracted.append(cache.extract(idx))
        else:
            # Plain KVCache with batched data — slice along batch dim
            new_cache = KVCache()
            new_cache.keys = mx.contiguous(cache.keys[idx : idx + 1])
            new_cache.values = mx.contiguous(cache.values[idx : idx + 1])
            new_cache.offset = cache.offset
            extracted.append(new_cache)
    return extracted


class MlxModelRunner:
    """Model runner that executes the entire model in MLX.

    This avoids the MPS<->MLX tensor bridge overhead by keeping all
    computation within MLX.
    """

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = False,
    ):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.model = None
        self._request_states: dict[str, MlxRequestState] = {}
        # Counter used to trigger periodic mx.clear_cache() calls.
        self._decode_step_ct: int = 0
        self.generation_stream: mx.Stream = mx.new_stream(mx.default_device())

        self._load_model()

    @staticmethod
    def _extract_logits(model_output):
        """Extract logits from model output, handling both tuple and direct returns."""
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    def _load_model(self):
        """Load model using mlx_lm."""
        logger.info(f"Loading MLX model: {self.model_path}")
        start_time = time.time()

        self.model, _ = mlx_lm_load(
            self.model_path,
            tokenizer_config={"trust_remote_code": self.trust_remote_code},
        )

        load_time = time.time() - start_time
        logger.info(f"MLX model loaded in {load_time:.2f}s")

    def prefill(
        self,
        req_id: str,
        token_ids: list[int],
    ) -> int:
        """Run prefill for a single request.

        If a request with the same req_id already has state (e.g. from a
        previous partial prefill), the existing KV cache is reused and only
        the new tokens are fed through the model.

        Args:
            req_id: Request identifier
            token_ids: Input token IDs (full sequence, including any
                previously prefilled tokens)

        Returns:
            Next token ID (greedy sampled)
        """
        existing_state = self._request_states.get(req_id)
        if existing_state is not None:
            # Continuation: reuse existing cache, feed only new tokens
            cached_input_len = (
                len(existing_state.token_ids) - existing_state.generated_tokens
            )
            new_tokens = token_ids[cached_input_len:]
            cache = existing_state.cache
        else:
            new_tokens = token_ids
            cache = make_prompt_cache(self.model)

        input_ids = mx.array([new_tokens], dtype=mx.int32)
        model_output = self.model(input_ids, cache=cache)

        logits = self._extract_logits(model_output)

        last_logits = logits[:, -1, :]
        next_token_mlx = mx.argmax(last_logits, axis=-1)

        # Evaluate everything together
        mx.eval(next_token_mlx, *[c.state for c in cache])
        next_token = int(next_token_mlx.item())

        # Store state for future decoding
        self._request_states[req_id] = MlxRequestState(
            token_ids=list(token_ids) + [next_token],
            cache=cache,
            generated_tokens=1,
        )

        return next_token

    def prefill_batch(
        self,
        req_ids: list[str],
        token_ids_list: list[list[int]],
    ) -> list[int]:
        """Run batched prefill for multiple requests in a single forward pass.

        When all sequences have the same length, they are stacked into a single
        batch tensor for one forward pass.  For variable-length sequences the
        method falls back to serial prefill.

        Args:
            req_ids: List of request identifiers
            token_ids_list: List of token ID sequences, one per request

        Returns:
            List of next token IDs (greedy sampled)
        """
        if len(req_ids) == 1:
            return [self.prefill(req_ids[0], token_ids_list[0])]

        # Check if all sequences have the same length (enables true batching)
        lengths = [len(tids) for tids in token_ids_list]
        if len(set(lengths)) != 1:
            # Variable lengths – fall back to serial prefill
            return [
                self.prefill(rid, tids) for rid, tids in zip(req_ids, token_ids_list)
            ]

        # All same length – use a single set of fresh caches;
        # they'll be populated with shape (batch_size, ...) on the first forward pass
        batch_cache = make_prompt_cache(self.model)

        # Stack into (batch_size, seq_len)
        batched_input = mx.array(
            [list(tids) for tids in token_ids_list], dtype=mx.int32
        )

        # Single forward pass
        model_output = self.model(batched_input, cache=batch_cache)
        logits = self._extract_logits(model_output)

        last_logits = logits[:, -1, :]
        next_tokens_mlx = mx.argmax(last_logits, axis=-1)

        # Evaluate everything together
        mx.eval(next_tokens_mlx, *[c.state for c in batch_cache])
        next_tokens = next_tokens_mlx.tolist()

        # Extract individual caches and store per-request state
        for i, req_id in enumerate(req_ids):
            individual_cache = _extract_kv_cache(batch_cache, i)
            self._request_states[req_id] = MlxRequestState(
                token_ids=list(token_ids_list[i]) + [next_tokens[i]],
                cache=individual_cache,
                generated_tokens=1,
            )

        return next_tokens

    def decode_batch(
        self,
        req_ids: list[str],
    ) -> list[int]:
        """Run batched decode for multiple requests.

        Args:
            req_ids: List of request IDs to decode

        Returns:
            List of next token IDs
        """
        if len(req_ids) == 1:
            return [self._decode_single(req_ids[0])]

        decode_reqs = []
        for req_id in req_ids:
            state = self._request_states[req_id]
            decode_reqs.append((req_id, state))

        return self._batched_decode(decode_reqs)

    def _decode_single(self, req_id: str) -> int:
        """Decode a single token for one request."""
        state = self._request_states[req_id]
        last_token = state.token_ids[-1]

        input_ids = mx.array([[last_token]], dtype=mx.int32)
        model_output = self.model(input_ids, cache=state.cache)

        logits = self._extract_logits(model_output)

        last_logits = logits[:, -1, :]
        next_token_mlx = mx.argmax(last_logits, axis=-1)

        mx.eval(next_token_mlx, *[c.state for c in state.cache])
        next_token = int(next_token_mlx.item())

        state.token_ids.append(next_token)
        state.generated_tokens += 1

        return next_token

    def _batched_decode(
        self, decode_reqs: list[tuple[str, MlxRequestState]]
    ) -> list[int]:
        """Run a single batched forward pass for multiple decode requests."""
        last_tokens = [state.token_ids[-1] for _, state in decode_reqs]

        # Merge individual KV caches into batched cache
        caches_list = [state.cache for _, state in decode_reqs]
        batch_cache = _merge_kv_caches(caches_list)

        # Create batched input: shape (batch_size, 1)
        batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]

        # Single forward pass
        model_output = self.model(batched_input, cache=batch_cache)
        logits = self._extract_logits(model_output)

        next_token_logits = logits[:, -1, :]
        next_tokens_mlx = mx.argmax(next_token_logits, axis=-1)

        mx.eval(next_tokens_mlx, *[c.state for c in batch_cache])
        next_tokens = next_tokens_mlx.tolist()

        # Extract updated caches back to individual requests
        for i, (_, state) in enumerate(decode_reqs):
            state.cache = _extract_kv_cache(batch_cache, i)
            state.token_ids.append(next_tokens[i])
            state.generated_tokens += 1

        return next_tokens

    # ------------------------------------------------------------------
    # Async (lazy-eval) API for overlap scheduling
    # ------------------------------------------------------------------

    def prefill_start(self, req_id: str, token_ids: list[int]) -> MlxPendingPrefill:
        """Queue a prefill forward pass without evaluating.

        Returns an :class:`MlxPendingPrefill` containing the lazy next-token
        ``mx.array``.  Call ``mx.async_eval(pending.lazy_token)`` to kick off
        GPU work, then :meth:`prefill_finalize`.
        """
        with mx.stream(self.generation_stream):
            existing_state = self._request_states.get(req_id)
            if existing_state is not None:
                cached_input_len = (
                    len(existing_state.token_ids) - existing_state.generated_tokens
                )
                new_tokens = token_ids[cached_input_len:]
                cache = existing_state.cache
            else:
                new_tokens = token_ids
                cache = make_prompt_cache(self.model)

            input_ids = mx.array([new_tokens], dtype=mx.int32)
            model_output = self.model(input_ids, cache=cache)
            logits = self._extract_logits(model_output)
            lazy_token = mx.argmax(logits[:, -1, :], axis=-1)

            return MlxPendingPrefill(
                lazy_token=lazy_token,
                cache=cache,
                req_id=req_id,
                token_ids=token_ids,
            )

    def prefill_finalize(self, pending: MlxPendingPrefill) -> int:
        """Materialise a pending prefill and store per-request state.

        Must be called *after* ``mx.async_eval(pending.lazy_token)`` has been called.
        """
        with mx.stream(self.generation_stream):
            # Evaluate lazy tokens/sync MLX computation graph
            next_token = int(pending.lazy_token.item())
            self._request_states[pending.req_id] = MlxRequestState(
                token_ids=list(pending.token_ids) + [next_token],
                cache=pending.cache,
                generated_tokens=1,
            )
            return next_token

    def decode_batch_start(self, req_ids: list[str]) -> MlxPendingDecode:
        """Queue a decode forward pass without evaluating.

        Returns an :class:`MlxPendingDecode` containing the lazy ``mx.array``
        of next-token indices plus the references needed by
        :meth:`decode_batch_finalize`.  The caller should call
        ``mx.async_eval(pending.lazy_tokens)`` to kick off GPU computation,
        then do CPU scheduling work, then call :meth:`decode_batch_finalize`
        (after ``mx.eval``) to materialise the tokens and update state.
        """
        with mx.stream(self.generation_stream):
            if len(req_ids) == 1:
                req_id = req_ids[0]
                state = self._request_states[req_id]
                last_token = state.token_ids[-1]
                input_ids = mx.array([[last_token]], dtype=mx.int32)
                model_output = self.model(input_ids, cache=state.cache)
                logits = self._extract_logits(model_output)
                lazy_token = mx.argmax(logits[:, -1, :], axis=-1)
                return MlxPendingDecode(
                    lazy_tokens=lazy_token,
                    batch_cache=state.cache,
                    decode_reqs=[(req_id, state)],
                )

            decode_reqs = [(rid, self._request_states[rid]) for rid in req_ids]
            last_tokens = [state.token_ids[-1] for _, state in decode_reqs]
            caches_list = [state.cache for _, state in decode_reqs]
            batch_cache = _merge_kv_caches(caches_list)

            batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]
            model_output = self.model(batched_input, cache=batch_cache)
            logits = self._extract_logits(model_output)
            lazy_tokens = mx.argmax(logits[:, -1, :], axis=-1)

            return MlxPendingDecode(
                lazy_tokens=lazy_tokens,
                batch_cache=batch_cache,
                decode_reqs=decode_reqs,
            )

    def decode_batch_start_chained(
        self,
        prev: MlxPendingDecode,
    ) -> MlxPendingDecode:
        """Build the next decode step on top of a still-lazy previous decode.

        We feed ``prev.lazy_tokens`` (an unevaluated ``mx.array`` of shape
        ``(B,)``) directly as the next step's input ids, reusing
        ``prev.batch_cache`` in-place so that the KV cache updates from
        step N and step N+1 live in the same tensors. MLX tracks the full
        dependency graph, so once ``mx.async_eval`` is called the GPU
        executes N+1 immediately after N with no gap.

        Caller contract:
        * ``prev`` MUST refer to the same set of requests as the batch the
          caller intends to run next (no composition change).
        * After calling this, finalize ``prev`` BEFORE finalizing the
          returned pending: state bookkeeping for step N has to happen
          before step N+1's bookkeeping.
        * Only the FINAL pending in the chain may extract per-request
          caches (see ``decode_batch_finalize``'s ``extract_cache`` flag),
          because the ``batch_cache`` is shared across all chained steps
          and reflects the latest state after every chained call.
        """
        with mx.stream(self.generation_stream):
            # prev.lazy_tokens has shape (B,) — add seq dim to get (B, 1)
            batched_input = prev.lazy_tokens[:, None]  # still a graph node
            model_output = self.model(
                batched_input, cache=prev.batch_cache
            )  # update batch_cache
            logits = self._extract_logits(model_output)
            next_lazy = mx.argmax(logits[:, -1, :], axis=-1)

            return MlxPendingDecode(
                lazy_tokens=next_lazy,
                batch_cache=prev.batch_cache,  # shared across the chain
                decode_reqs=prev.decode_reqs,  # identical req set
            )

    def decode_batch_finalize(
        self,
        pending: MlxPendingDecode,
        extract_cache: bool = True,
    ) -> list[int]:
        """Materialise a pending decode and update per-request state.

        The call to ``pending.lazy_tokens.tolist()`` below implicitly blocks
        until that specific lazy array is evaluated, so the caller does NOT
        need to call ``mx.eval`` ahead of time. It only needs to have
        previously handed the pending's outputs to ``mx.async_eval`` (or
        trust that something downstream in the MLX graph will).

        Args:
            pending: The lazy decode returned by :meth:`decode_batch_start`
                or :meth:`decode_batch_start_chained`.
            extract_cache: If True, snapshot the batched KV cache back into
                each request's ``state.cache``. Pass ``False`` when this
                pending is mid-chain — i.e. another step has already been
                built on top of ``pending.batch_cache`` via
                :meth:`decode_batch_start_chained` — because the shared
                ``batch_cache`` now reflects the later step's state, not
                this one's. Only the tail of a chain should extract.
        """
        with mx.stream(self.generation_stream):
            # Evaluate lazy tokens/sync MLX computation graph
            raw = pending.lazy_tokens.tolist()
            if not isinstance(raw, list):
                raw = [raw]
            # Extract next tokens for each batch
            next_tokens = [int(t) for t in raw]

            is_batched = len(pending.decode_reqs) > 1
            # Update each request's state
            for i, (_, state) in enumerate(pending.decode_reqs):
                if is_batched and extract_cache:
                    state.cache = _extract_kv_cache(pending.batch_cache, i)
                state.token_ids.append(next_tokens[i])
                state.generated_tokens += 1

            # Clear KV cache after enough decode passes
            self._decode_step_ct += 1
            if self._decode_step_ct % 256 == 0:
                mx.clear_cache()

            return next_tokens

    def remove_request(self, req_id: str):
        """Clean up state for a completed request."""
        self._request_states.pop(req_id, None)

    def clear(self):
        """Clear all request states."""
        self._request_states.clear()
