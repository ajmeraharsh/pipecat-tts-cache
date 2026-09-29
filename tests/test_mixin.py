#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""End-to-end behavior of ``TTSCacheMixin`` against a real Pipecat 1.5.0 pipeline.

Tests drive frames through ``run_test`` (Pipecat's own pipeline test harness) with a
real ``TTSService`` subclass fake and a real ``MemoryCacheBackend`` — no mocking of the
cache or the framework. The fakes mirror the two real delivery models:

- HTTP-style: ``run_tts`` yields audio synchronously.
- WebSocket-style: ``run_tts`` delivers audio asynchronously from a background task.
"""

import asyncio

import pytest
from collections.abc import AsyncGenerator

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test

from pipecat_tts_cache import CacheBackend, MemoryCacheBackend, TTSCacheMixin

_AUDIO = b"\x00\x01" * 320  # 640 bytes of 16-bit PCM
_SAMPLE_RATE = 16000


class _BaseFakeTTS(TTSService):
    """Common fake TTS setup: complete store-mode settings + call accounting."""

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )
        # Complete store-mode settings (all non-`extra` fields given).
        self._settings = TTSSettings(voice="test-voice", model="test-model", language=None)
        self.run_tts_calls = 0
        self.synthesized_texts: list[str] = []

    def can_generate_metrics(self) -> bool:
        return False


class FakeHttpTTS(_BaseFakeTTS):
    """HTTP-style: yields one audio frame synchronously; no word timestamps."""

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        self.run_tts_calls += 1
        self.synthesized_texts.append(text)
        yield TTSAudioRawFrame(
            audio=_AUDIO, sample_rate=_SAMPLE_RATE, num_channels=1, context_id=context_id
        )


class FakeWordTTS(_BaseFakeTTS):
    """Word-timestamp style: emits per-word timestamps then audio (HTTP delivery)."""

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        self.run_tts_calls += 1
        self.synthesized_texts.append(text)
        word_times = [(word, i * 0.1) for i, word in enumerate(text.split())]
        await self.add_word_timestamps(word_times, context_id=context_id)
        yield TTSAudioRawFrame(
            audio=_AUDIO, sample_rate=_SAMPLE_RATE, num_channels=1, context_id=context_id
        )


class FakeSlowWsTTS(_BaseFakeTTS):
    """WebSocket-style: audio arrives from a delayed background task (not yielded)."""

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        self.run_tts_calls += 1
        self.synthesized_texts.append(text)

        async def _deliver():
            await asyncio.sleep(0.3)
            await self.append_to_audio_context(
                context_id,
                TTSAudioRawFrame(
                    audio=_AUDIO, sample_rate=_SAMPLE_RATE, num_channels=1, context_id=context_id
                ),
            )
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

        self.create_task(_deliver(), name=f"fake_ws_deliver_{context_id}")
        if False:  # make this an async generator that yields nothing
            yield


class CachedHttpTTS(TTSCacheMixin, FakeHttpTTS):
    pass


class CachedWordTTS(TTSCacheMixin, FakeWordTTS):
    pass


class CachedSlowWsTTS(TTSCacheMixin, FakeSlowWsTTS):
    pass


def _speak(text: str) -> TTSSpeakFrame:
    return TTSSpeakFrame(text=text, append_to_context=False)


async def test_cache_miss_synthesizes_and_stores():
    """A first request is a miss: it synthesizes, emits audio, and populates the cache."""
    backend = MemoryCacheBackend()
    tts = CachedHttpTTS(cache_backend=backend)

    down, _ = await run_test(tts, frames_to_send=[_speak("hello world")])

    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert tts.run_tts_calls == 1
    assert len(audio) == 1
    assert (await backend.get_stats())["size"] == 1  # one entry stored


async def test_cache_hit_replays_without_resynthesizing():
    """A repeated request within the session hits cache: audio replays, provider not called again."""
    backend = MemoryCacheBackend()
    tts = CachedHttpTTS(cache_backend=backend)

    down, _ = await run_test(
        tts,
        frames_to_send=[
            _speak("hello world"),
            SleepFrame(sleep=0.3),  # let the first request finalize + store
            _speak("hello world"),  # cache hit
        ],
    )

    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    stats = await tts.get_cache_stats()

    assert tts.run_tts_calls == 1  # second request served from cache
    assert len(audio) == 2  # both the live and the replayed audio reached the transport
    assert stats["hits"] == 1
    assert stats["misses"] == 1


async def test_distinct_texts_do_not_collide():
    """Different texts are cached independently; both are synthesized."""
    backend = MemoryCacheBackend()
    tts = CachedHttpTTS(cache_backend=backend)

    await run_test(
        tts,
        frames_to_send=[
            _speak("first phrase"),
            SleepFrame(sleep=0.2),
            _speak("second phrase"),
            SleepFrame(sleep=0.2),
        ],
    )

    assert tts.run_tts_calls == 2
    assert (await backend.get_stats())["size"] == 2


async def test_word_timestamps_preserved_on_cache_hit():
    """Word timestamps are cached and replayed, so transcripts survive a hit (GitHub #6)."""
    backend = MemoryCacheBackend()
    tts = CachedWordTTS(cache_backend=backend)

    down, _ = await run_test(
        tts,
        frames_to_send=[
            _speak("hello world"),
            SleepFrame(sleep=0.3),
            _speak("hello world"),  # cache hit
        ],
    )

    text_frames = [f.text for f in down if isinstance(f, TTSTextFrame)]

    assert tts.run_tts_calls == 1
    # Both the live pass and the cached pass regenerate the word TTSTextFrames.
    assert text_frames == ["hello", "world", "hello", "world"]


async def test_disabled_cache_is_a_transparent_passthrough():
    """With no backend the mixin is inert: the wrapped service runs normally every time."""
    tts = CachedHttpTTS(cache_backend=None)

    down, _ = await run_test(
        tts,
        frames_to_send=[
            _speak("hello world"),
            SleepFrame(sleep=0.2),
            _speak("hello world"),
        ],
    )

    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert tts.run_tts_calls == 2  # no caching -> synthesized both times
    assert len(audio) == 2


async def test_websocket_audio_is_captured_and_replayed():
    """Async (websocket-style) audio is captured via push_frame and replayed on hit."""
    backend = MemoryCacheBackend()
    tts = CachedSlowWsTTS(cache_backend=backend)

    down, _ = await run_test(
        tts,
        frames_to_send=[
            _speak("hello world"),
            SleepFrame(sleep=0.5),  # allow async delivery + finalize
            _speak("hello world"),  # cache hit
            SleepFrame(sleep=0.2),
        ],
    )

    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert tts.run_tts_calls == 1
    assert len(audio) == 2


async def test_cache_hit_audio_stays_in_its_own_bracket():
    """GitHub #2: replayed audio must flow through the audio context in order, not
    interleave. Each request (live or cached) forms a clean Started -> Audio -> Stopped
    group; the cached group must not bleed into the live one."""
    backend = MemoryCacheBackend()
    tts = CachedHttpTTS(cache_backend=backend)

    down, _ = await run_test(
        tts,
        frames_to_send=[
            _speak("alpha"),
            SleepFrame(sleep=0.3),
            _speak("alpha"),  # cache hit, replayed through the same audio-context path
        ],
    )

    seq = [
        type(f).__name__
        for f in down
        if isinstance(f, (TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame))
    ]
    assert seq == [
        "TTSStartedFrame",
        "TTSAudioRawFrame",
        "TTSStoppedFrame",
        "TTSStartedFrame",
        "TTSAudioRawFrame",
        "TTSStoppedFrame",
    ]


async def test_interruption_discards_pending_capture():
    """An interruption discards the in-flight capture context so interrupted audio is
    never cached. Asserting ``_contexts`` (not just backend size) gives the test teeth:
    it fails if the mixin ever stops clearing capture state on interruption."""
    backend = MemoryCacheBackend()
    tts = CachedSlowWsTTS(cache_backend=backend)

    await run_test(
        tts,
        frames_to_send=[
            _speak("hello world"),
            SleepFrame(sleep=0.05),  # run_tts registered the context; audio not delivered yet
            InterruptionFrame(),
            SleepFrame(sleep=0.1),  # still before the 0.3s delivery -> nothing finalized
        ],
    )

    assert tts._contexts == {}  # the pending capture context was cleared
    assert (await backend.get_stats())["size"] == 0  # nothing interrupted was cached


async def test_clear_cache_removes_stored_entries():
    """The public clear_cache() empties the backend."""
    backend = MemoryCacheBackend()
    tts = CachedHttpTTS(cache_backend=backend)

    await run_test(
        tts,
        frames_to_send=[_speak("one"), SleepFrame(sleep=0.2), _speak("two"), SleepFrame(sleep=0.2)],
    )
    assert (await backend.get_stats())["size"] == 2

    assert await tts.clear_cache() == 2
    assert (await backend.get_stats())["size"] == 0


async def test_get_cache_stats_reports_hits_misses_and_backend():
    backend = MemoryCacheBackend()
    tts = CachedHttpTTS(cache_backend=backend)

    await run_test(
        tts,
        frames_to_send=[_speak("hello"), SleepFrame(sleep=0.3), _speak("hello")],
    )

    stats = await tts.get_cache_stats()
    assert stats["enabled"] is True
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["total_requests"] == 2
    assert stats["hit_rate"] == 0.5
    assert stats["backend"]["type"] == "memory"


class _FailingBackend(CacheBackend):
    """A backend whose reads and writes always raise (to prove fail-safe behavior)."""

    async def get(self, key):
        raise RuntimeError("backend down")

    async def set(self, key, response, ttl=None):
        raise RuntimeError("backend down")

    async def delete(self, key):
        return False

    async def clear(self, namespace=None):
        return 0

    async def exists(self, key):
        return False

    async def get_stats(self):
        return {"type": "failing"}

    async def close(self):
        pass


async def test_backend_failure_never_breaks_synthesis():
    """If the cache raises, the request degrades to a normal (uncached) synthesis."""
    tts = CachedHttpTTS(cache_backend=_FailingBackend())

    down, _ = await run_test(tts, frames_to_send=[_speak("hello world")])

    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert tts.run_tts_calls == 1  # get() raised -> treated as a miss -> synthesized
    assert len(audio) == 1  # audio still reached the transport despite set() raising


async def test_object_valued_setting_does_not_break_synthesis():
    """A non-JSON-serializable settings value (like Cartesia's ``GenerationConfig``
    pydantic model) must not break cache-key generation or synthesis (review C1)."""
    from pydantic import BaseModel

    class _GenCfg(BaseModel):
        speed: float = 1.2

    backend = MemoryCacheBackend()
    tts = CachedHttpTTS(cache_backend=backend)
    tts._settings = TTSSettings(
        voice="v", model="m", language=None, extra={"generation_config": _GenCfg()}
    )

    down, _ = await run_test(tts, frames_to_send=[_speak("hello")])

    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert len(audio) == 1  # synthesized despite the object-valued setting
    assert (await backend.get_stats())["size"] == 1  # and cached (the model was coerced)


async def test_different_wrapped_services_do_not_share_a_key():
    """Two different providers with identical voice/model must not collide (review H4)."""
    backend = MemoryCacheBackend()
    a = CachedHttpTTS(cache_backend=backend)
    a._settings = TTSSettings(voice="v", model="m", language=None)
    a._sample_rate = _SAMPLE_RATE
    b = CachedWordTTS(cache_backend=backend)
    b._settings = TTSSettings(voice="v", model="m", language=None)
    b._sample_rate = _SAMPLE_RATE

    assert a._generate_cache_key("hi") != b._generate_cache_key("hi")


async def test_clear_cache_by_namespace_is_scoped():
    """clear_cache(namespace) removes only that namespace's entries (review H1)."""
    backend = MemoryCacheBackend()
    tenant_a = CachedHttpTTS(cache_backend=backend, cache_namespace="tenant_a")
    tenant_b = CachedHttpTTS(cache_backend=backend, cache_namespace="tenant_b")

    await run_test(tenant_a, frames_to_send=[_speak("shared phrase"), SleepFrame(sleep=0.2)])
    await run_test(tenant_b, frames_to_send=[_speak("shared phrase"), SleepFrame(sleep=0.2)])
    assert (await backend.get_stats())["size"] == 2  # namespaces isolate identical text

    removed = await tenant_a.clear_cache(namespace="tenant_a")
    assert removed == 1
    assert (await backend.get_stats())["size"] == 1  # tenant_b's entry survives


async def test_run_tts_failure_clears_context_and_caches_nothing():
    """If the wrapped service raises mid-synthesis, the pending capture context is
    cleared (no stale state leaks) and nothing is cached."""

    class _FailingProvider(_BaseFakeTTS):
        async def run_tts(self, text, context_id):
            raise RuntimeError("synthesis failed")
            yield  # pragma: no cover - makes run_tts an async generator

    class _CachedFailing(TTSCacheMixin, _FailingProvider):
        pass

    backend = MemoryCacheBackend()
    tts = _CachedFailing(cache_backend=backend)

    await run_test(tts, frames_to_send=[_speak("hello world")])

    assert tts._contexts == {}  # context discarded on failure
    assert (await backend.get_stats())["size"] == 0


async def test_keygen_bypass_does_not_pollute_a_sibling_context():
    """review NG1: when cache-key generation fails mid-turn, the bypassed request must not
    leave its audio in an earlier sibling's capture (siblings in a turn reuse one context_id).
    The bypass drops the capture context so no foreign audio is appended to it."""
    from unittest import mock

    from pipecat_tts_cache.mixin import _ContextCapture, _PendingTask
    from pipecat_tts_cache.models import CachedAudioChunk

    good_audio = b"\xaa\xaa" * 160
    bad_audio = b"\xbb\xbb" * 160

    class _TextAudioTTS(_BaseFakeTTS):
        async def run_tts(self, text, context_id):
            self.run_tts_calls += 1
            yield TTSAudioRawFrame(
                audio=good_audio if "good" in text else bad_audio,
                sample_rate=_SAMPLE_RATE,
                num_channels=1,
                context_id=context_id,
            )

    class _Cached(TTSCacheMixin, _TextAudioTTS):
        pass

    backend = MemoryCacheBackend()
    tts = _Cached(cache_backend=backend)
    good_key = tts._generate_cache_key("good sentence")

    # Sentence 1 is a live miss, mid-capture in a reused turn context.
    tts._contexts["turn"] = _ContextCapture(
        tasks=[_PendingTask("good sentence", good_key, 2)],
        audio=[CachedAudioChunk(good_audio, _SAMPLE_RATE, 1)],
    )

    # A sibling in the same turn whose key generation fails -> the bypass path runs.
    with mock.patch.object(_Cached, "_generate_cache_key", side_effect=RuntimeError("boom")):
        yielded = [f async for f in tts.run_tts("bad sibling", "turn")]
        for frame in yielded:
            if isinstance(frame, TTSAudioRawFrame):
                await tts.push_frame(frame)  # the framework would push the bypassed audio

    # (a) the bypassed request still synthesized (fail-safe: audio keeps flowing)
    assert any(isinstance(f, TTSAudioRawFrame) for f in yielded)
    # (b) the bypass dropped the capture context, so the bad audio was never captured
    assert "turn" not in tts._contexts
    # (c) the sibling's cache entry is never polluted with the bypassed audio
    await tts._finalize_context("turn")
    entry = await backend.get(good_key)
    assert entry is None or bad_audio not in entry.audio_chunks[0].audio


async def test_unserializable_setting_bypasses_cache_without_breaking_synthesis():
    """review NG2: an opaque settings value makes key generation raise; the mixin must bypass
    the cache and still synthesize (rather than crash or mint a non-deterministic key)."""

    class _Opaque:
        pass

    backend = MemoryCacheBackend()
    tts = CachedHttpTTS(cache_backend=backend)
    tts._settings = TTSSettings(voice="v", model="m", language=None, extra={"weird": _Opaque()})

    down, _ = await run_test(tts, frames_to_send=[_speak("hello")])

    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert len(audio) == 1  # synthesized despite un-keyable settings
    assert (await backend.get_stats())["size"] == 0  # bypassed: nothing cached


async def test_provider_id_skips_an_intermediate_mixin():
    """review NG4: with an extra mixin between TTSCacheMixin and the service, the provider
    discriminator must resolve to the real service, not the intermediate mixin — otherwise two
    different services stacked behind the same mixin would collide on a shared backend."""

    class _ExtraMixin:
        pass

    class _ServiceOne(_BaseFakeTTS):
        async def run_tts(self, text, context_id):  # pragma: no cover - not exercised
            yield None

    class _ServiceTwo(_BaseFakeTTS):
        async def run_tts(self, text, context_id):  # pragma: no cover - not exercised
            yield None

    class _CachedOne(TTSCacheMixin, _ExtraMixin, _ServiceOne):
        pass

    class _CachedTwo(TTSCacheMixin, _ExtraMixin, _ServiceTwo):
        pass

    one = _CachedOne(cache_backend=MemoryCacheBackend())
    one._sample_rate = _SAMPLE_RATE
    two = _CachedTwo(cache_backend=MemoryCacheBackend())
    two._sample_rate = _SAMPLE_RATE

    one_id = one._wrapped_service_id()
    two_id = two._wrapped_service_id()
    assert one_id is not None and one_id.endswith("_ServiceOne")
    assert two_id is not None and two_id.endswith("_ServiceTwo")
    assert one._generate_cache_key("hi") != two._generate_cache_key("hi")


async def test_add_word_timestamps_adapts_to_an_older_base_signature():
    """Backward-compat: the mixin forwards only the kwargs the wrapped service declares, so
    word-timestamp replay works on older Pipecat bases that predate
    includes_inter_frame_spaces / pre_merge_tokens (no TypeError)."""
    received = []

    class _NarrowBase(_BaseFakeTTS):
        # Mimics pre-1.3.0: add_word_timestamps without the newer optional kwargs.
        async def add_word_timestamps(self, word_times, context_id=None):
            received.append((list(word_times), context_id))

    class _Cached(TTSCacheMixin, _NarrowBase):
        pass

    tts = _Cached(cache_backend=MemoryCacheBackend())

    # The mixin's own signature accepts these newer kwargs; forwarding must drop the ones the
    # narrow base cannot accept instead of raising TypeError.
    await tts.add_word_timestamps(
        [("hi", 0.0)], context_id="c", includes_inter_frame_spaces=True, pre_merge_tokens=True
    )
    assert received == [([("hi", 0.0)], "c")]



class FakeTurnTTS(_BaseFakeTTS):
    """Turn-style provider: one shared context per LLM turn, turn-cumulative word timestamps.

    Mirrors what this caching runs against in production. Each sentence gets its own marker
    byte and a length proportional to its word count, so audio sliced at the wrong origin
    shows up as a neighbour's bytes instead of as coincidentally-correct audio.
    """

    PER_WORD_BYTES = int(0.25 * _SAMPLE_RATE) * 2  # 0.25s of 16-bit mono per word

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._turn_clock = 0.0
        self._marks: dict[str, bytes] = {}

    def mark_for(self, text: str) -> bytes:
        if text not in self._marks:
            self._marks[text] = bytes([0xA0 + len(self._marks)])
        return self._marks[text]

    async def on_turn_context_created(self, context_id: str):
        self._turn_clock = 0.0
        await super().on_turn_context_created(context_id)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        self.run_tts_calls += 1
        self.synthesized_texts.append(text)
        words = text.split()
        word_times = [(w, self._turn_clock + i * 0.25) for i, w in enumerate(words)]
        self._turn_clock += 0.25 * len(words)
        await self.add_word_timestamps(word_times, context_id=context_id)
        yield TTSAudioRawFrame(
            audio=self.mark_for(text) * (self.PER_WORD_BYTES * len(words)),
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            context_id=context_id,
        )


class CachedTurnTTS(TTSCacheMixin, FakeTurnTTS):
    pass


def _turn(sentences: list[str]) -> list[Frame]:
    """One LLM turn: streamed text aggregated into sentences that share one audio context.

    Text must arrive as ``TextFrame`` (the LLM streaming path). ``TTSSpeakFrame`` is
    explicitly excluded from the turn context by ``TTSService``, so it would give each
    sentence its own context and never exercise the multi-sentence split.
    """
    frames: list[Frame] = [LLMFullResponseStartFrame()]
    for text in sentences:
        frames.append(TextFrame(text + ". "))  # trailing period -> aggregator emits a sentence
    frames.append(LLMFullResponseEndFrame())
    frames.append(SleepFrame(sleep=0.4))
    return frames


async def _assert_each_sentence_kept_its_own_audio(backend, tts, texts: list[str]) -> None:
    """Every cached sentence holds only its own audio, at its own full length."""
    for text in texts:
        entry = await backend.get(tts._generate_cache_key(text))
        if entry is None:
            continue  # a skipped write is safe; contaminated audio is not
        audio = entry.audio_chunks[0].audio
        own = tts.mark_for(text)
        foreign = sorted({bytes([b]).hex() for b in set(audio)} - {own.hex()})
        assert not foreign, (
            f"'{text}' was cached carrying another sentence's audio "
            f"(found {foreign}, expected only {own.hex()})"
        )
        expected = tts.PER_WORD_BYTES * len(text.split())
        assert len(audio) == expected, (
            f"'{text}' cached {len(audio)} bytes, expected {expected} — "
            "its slice boundary was taken at the wrong origin"
        )


async def test_turn_leading_cached_sentence_does_not_shift_the_misses():
    """hit, miss, miss — the leading hit contributes no audio but does consume turn-clock
    time, which is exactly the offset that used to shift every following slice."""
    backend = MemoryCacheBackend()
    tts = CachedTurnTTS(cache_backend=backend)

    await run_test(tts, frames_to_send=_turn(["alpha one"]))  # prime sentence 1
    await run_test(
        tts, frames_to_send=_turn(["alpha one", "bravo two three", "charlie four"])
    )

    await _assert_each_sentence_kept_its_own_audio(
        backend, tts, ["alpha one", "bravo two three", "charlie four"]
    )


async def test_turn_trailing_cached_sentence():
    """miss, hit — the trailing hit contributes no bytes; the miss keeps its whole slice."""
    backend = MemoryCacheBackend()
    tts = CachedTurnTTS(cache_backend=backend)

    await run_test(tts, frames_to_send=_turn(["bravo two three"]))
    await run_test(tts, frames_to_send=_turn(["alpha one", "bravo two three"]))

    await _assert_each_sentence_kept_its_own_audio(
        backend, tts, ["alpha one", "bravo two three"]
    )


async def test_turn_cached_sentence_in_the_middle():
    """miss, hit, miss — a gap mid-turn, the case no single rebase offset can fix."""
    backend = MemoryCacheBackend()
    tts = CachedTurnTTS(cache_backend=backend)

    await run_test(tts, frames_to_send=_turn(["bravo two three"]))
    await run_test(
        tts, frames_to_send=_turn(["alpha one", "bravo two three", "charlie four"])
    )

    await _assert_each_sentence_kept_its_own_audio(
        backend, tts, ["alpha one", "bravo two three", "charlie four"]
    )


async def test_turn_alternating_hits_and_misses():
    """hit, miss, hit, miss across four sentences of differing length."""
    backend = MemoryCacheBackend()
    tts = CachedTurnTTS(cache_backend=backend)

    await run_test(tts, frames_to_send=_turn(["alpha one"]))
    await run_test(tts, frames_to_send=_turn(["charlie four five six"]))
    await run_test(
        tts,
        frames_to_send=_turn(
            ["alpha one", "bravo two three", "charlie four five six", "delta seven"]
        ),
    )

    await _assert_each_sentence_kept_its_own_audio(
        backend,
        tts,
        ["alpha one", "bravo two three", "charlie four five six", "delta seven"],
    )


async def test_multi_sentence_turn_stays_uncached_across_repeats():
    """A multi-sentence turn is never cached, so repeating it re-synthesizes every sentence.

    This is the accepted cost of refusing to guess sentence boundaries. Turns that speak a
    single text — the canned lines this cache targets — still hit; see
    ``test_cache_hit_replays_without_resynthesizing``.
    """
    backend = MemoryCacheBackend()
    tts = CachedTurnTTS(cache_backend=backend)

    await run_test(tts, frames_to_send=_turn(["alpha one", "bravo two three"]))
    assert (await backend.get_stats())["size"] == 0

    calls = tts.run_tts_calls
    await run_test(tts, frames_to_send=_turn(["alpha one", "bravo two three"]))

    assert tts.run_tts_calls > calls  # nothing was cached, so both sentences run again
    assert (await backend.get_stats())["size"] == 0


async def test_multi_sentence_context_is_not_cached():
    """A context holding several texts is skipped rather than split.

    Nothing in the frame stream delimits one sentence's audio from the next inside a reused
    turn context, so any boundary would be an estimate. Skipping costs one re-synthesis; a
    wrong estimate would replay mis-sliced audio for the whole TTL.
    """
    from pipecat_tts_cache.mixin import _ContextCapture, _PendingTask
    from pipecat_tts_cache.models import CachedAudioChunk

    backend = MemoryCacheBackend()
    tts = CachedHttpTTS(cache_backend=backend)

    tts._contexts["turn-ctx"] = _ContextCapture(
        tasks=[
            _PendingTask("hello world", tts._generate_cache_key("hello world"), 2),
            _PendingTask("foo bar", tts._generate_cache_key("foo bar"), 2),
        ],
        audio=[CachedAudioChunk(b"\x11" * 8000 + b"\x22" * 8000, _SAMPLE_RATE, 1)],
        word_timestamps=[("hello", 0.0), ("world", 0.25), ("foo", 0.5), ("bar", 0.75)],
    )

    await tts._finalize_context("turn-ctx")

    assert (await backend.get_stats())["size"] == 0


async def test_single_sentence_context_caches_its_exact_audio():
    """The single-text path stores the captured audio verbatim — no slicing, no estimate.

    This is the path fixed canned lines take, which is what the cache exists for.
    """
    from pipecat_tts_cache.mixin import _ContextCapture, _PendingTask
    from pipecat_tts_cache.models import CachedAudioChunk

    backend = MemoryCacheBackend()
    tts = CachedHttpTTS(cache_backend=backend)

    key = tts._generate_cache_key("connecting you now")
    audio = b"\x11" * 12000

    tts._contexts["turn-ctx"] = _ContextCapture(
        tasks=[_PendingTask("connecting you now", key, 3)],
        audio=[CachedAudioChunk(audio, _SAMPLE_RATE, 1)],
        word_timestamps=[("connecting", 0.0), ("you", 0.25), ("now", 0.5)],
    )

    await tts._finalize_context("turn-ctx")

    entry = await backend.get(key)
    assert entry is not None
    assert entry.audio_chunks[0].audio == audio
    assert [wt.word for wt in (entry.word_timestamps or [])] == ["connecting", "you", "now"]


@pytest.mark.parametrize("miss_count", [2, 3, 4, 5], ids=lambda n: f"{n}-texts")
async def test_any_multi_text_context_is_skipped_regardless_of_layout(miss_count):
    """No arrangement of texts in one context is cached — the count alone decides.

    Covers the interleavings that motivated this: hits contribute no audio but do advance
    the provider's turn clock, so a turn like miss,hit,hit,miss leaves a buffer whose gaps
    no timestamp math can recover. The rule is uniform, so none of those shapes can slip in.
    """
    from pipecat_tts_cache.mixin import _ContextCapture, _PendingTask
    from pipecat_tts_cache.models import CachedAudioChunk

    backend = MemoryCacheBackend()
    tts = CachedHttpTTS(cache_backend=backend)

    tasks = [
        _PendingTask(f"sentence {i}", tts._generate_cache_key(f"sentence {i}"), 2)
        for i in range(miss_count)
    ]
    tts._contexts["turn-ctx"] = _ContextCapture(
        tasks=tasks,
        audio=[CachedAudioChunk(b"\x33" * (8000 * miss_count), _SAMPLE_RATE, 1)],
        word_timestamps=[(f"w{i}", i * 0.25) for i in range(2 * miss_count)],
    )

    await tts._finalize_context("turn-ctx")

    assert (await backend.get_stats())["size"] == 0


class SelfBracketingTTS(TTSService):
    """A service that opens its own audio context and TTFB clock inside ``run_tts``.

    Faithful to ElevenLabs, Deepgram, Inworld, NVIDIA, ResembleAI and Rime: they pass
    ``push_start_frame=False`` because they emit their own ``TTSStartedFrame``, which makes
    ``TTSService._push_tts_frames`` skip the bracket it normally does before ``run_tts``.
    """

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=False,
            push_stop_frames=False,
            push_text_frames=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )
        self._settings = TTSSettings(voice="test-voice", model="test-model", language=None)
        self.run_tts_calls = 0
        self.ttfb_events: list[str] = []

    def can_generate_metrics(self) -> bool:
        return True

    async def start_ttfb_metrics(self):
        self.ttfb_events.append("start")
        await super().start_ttfb_metrics()

    async def stop_ttfb_metrics(self):
        self.ttfb_events.append("stop")
        await super().stop_ttfb_metrics()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        self.run_tts_calls += 1
        if not self.audio_context_available(context_id):
            await self.create_audio_context(context_id)
            await self.start_ttfb_metrics()
            yield TTSStartedFrame(context_id=context_id)

        async def _deliver():
            await asyncio.sleep(0.2)
            await self.append_to_audio_context(
                context_id,
                TTSAudioRawFrame(
                    audio=_AUDIO,
                    sample_rate=_SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                ),
            )
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

        self.create_task(_deliver(), name=f"deliver_{context_id}")


class CachedSelfBracketingTTS(TTSCacheMixin, SelfBracketingTTS):
    pass


async def test_cache_hit_reports_ttfb_for_a_self_bracketing_service():
    """A replayed response must still open a context and start the TTFB clock.

    Services that pass ``push_start_frame=False`` do that work inside their own ``run_tts``,
    which a cache hit never reaches. Without the mixin standing in, the clock is stopped
    having never started (so no TTFB is reported at all) and the replayed audio is appended
    to a context nobody opened, which the framework papers over by recreating it.
    """
    backend = MemoryCacheBackend()
    tts = CachedSelfBracketingTTS(cache_backend=backend)

    await run_test(
        tts,
        frames_to_send=[
            TTSSpeakFrame(text="hello world", append_to_context=False),
            SleepFrame(sleep=0.6),
        ],
    )
    assert tts.ttfb_events == ["start", "stop"], "baseline miss should bracket normally"

    tts.ttfb_events.clear()
    calls_before = tts.run_tts_calls
    down, _ = await run_test(
        tts,
        frames_to_send=[
            TTSSpeakFrame(text="hello world", append_to_context=False),
            SleepFrame(sleep=0.6),
        ],
    )

    assert tts.run_tts_calls == calls_before, "the hit should not reach the provider"
    assert [f for f in down if isinstance(f, TTSAudioRawFrame)], "replayed audio never arrived"
    assert tts.ttfb_events == ["start", "stop"], (
        f"TTFB was not bracketed on the cache hit: {tts.ttfb_events} — a lone 'stop' means "
        "the clock was never started, so no TTFB is reported for replayed speech"
    )
