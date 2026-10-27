#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The labels Pipecat Tone's tagger chooses, and the expression tags they're written as.

The tagger labels with the standard deliveries and sounds. A call's label space holds
those its TTS service performs and its policy offers, each with the name the service
performs it by, and a sentence's labels are written as tags in those names.
"""

from dataclasses import dataclass, field, replace

from pipecat.expressive.expression import (
    BREATH_KINDS,
    NEUTRAL_DELIVERY,
    STANDARD_DELIVERY,
    STANDARD_SOUNDS,
    ExpressionCapabilities,
    standard_name,
)
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy

# The services the model was trained with, by the names it reads for them. It reads
# "other" for any other.
TRAINED_SERVICES = ("cartesia", "elevenlabs", "fish", "minimax")
OTHER_SERVICE = "other"


@dataclass(frozen=True)
class LabelSpace:
    """The labels a call can use, each with the name its TTS service performs it by.

    Parameters:
        deliveries: Standard delivery to the name to write, in standard order.
        sounds: Standard sound to the name to write, in standard order.
        emphasis: Whether the service stresses ``^^words^^``.
        tts: The service, by the name the model was trained to read for it.
        breaths: Kind of breath to the name to write, where a breath is offered.
            The model chooses only a breath, and the tagger draws its kind.
    """

    deliveries: dict[str, str]
    sounds: dict[str, str]
    emphasis: bool
    tts: str = OTHER_SERVICE
    breaths: dict[str, str] = field(default_factory=dict)

    @classmethod
    def resolve(
        cls,
        capabilities: ExpressionCapabilities,
        policy: ExpressionPolicy,
        service_name: str = "",
        vocabulary: dict[str, list[str]] | None = None,
    ):
        """The labels the service performs and the policy offers.

        A breath, and its kinds, are offered only with ``ExpressionGuidance.BREATHE``
        in the policy.

        Args:
            capabilities: What the service performs.
            policy: The pipeline's expression policy.
            service_name: The service's name, such as ``"ElevenLabsTTSService#0"``,
                which says which of the services the model was trained with it is.
            vocabulary: Names to try first for a label, before the service's own.
        """
        if vocabulary:
            names = capabilities.standard_names
            first = {label: [*own, *names.get(label, [label])] for label, own in vocabulary.items()}
            capabilities = replace(capabilities, standard_names={**names, **first})
        sounds = [s for s in STANDARD_SOUNDS if s != "breath" or _breathes(policy)]
        resolved = _resolve(capabilities, sounds, policy.sounds)
        return cls(
            deliveries=_resolve(capabilities, STANDARD_DELIVERY, policy.delivery),
            sounds=resolved,
            emphasis=capabilities.emphasis,
            tts=_trained_service(service_name),
            breaths=_resolve(capabilities, BREATH_KINDS, policy.sounds)
            if "breath" in resolved
            else {},
        )

    def sound_name(self, label: str) -> str | None:
        """The name to write for a sound or a kind of breath, if the space holds it."""
        return self.sounds.get(label) or self.breaths.get(label)


@dataclass(frozen=True)
class Span:
    """A delivery for some words, ``<<delivery: words>>``, by offsets in the sentence.

    Parameters:
        start: The offset of the first word's first character.
        end: The offset just past the last word.
        delivery: The words' delivery, a standard label.
    """

    start: int
    end: int
    delivery: str


@dataclass
class SentenceLabels:
    """What the tagger chose for one sentence, in standard labels; usually nothing.

    Parameters:
        delivery: The sentence's delivery, or None for the voice's baseline.
        spans: Words in a delivery of their own, mainly slow details.
        emphasis: Words to stress, as ``(start, end)`` offsets.
        sound_before: A sound before the sentence, or None.
        sound_after: A sound after the sentence, or None.
    """

    delivery: str | None = None
    spans: list[Span] = field(default_factory=list)
    emphasis: list[tuple[int, int]] = field(default_factory=list)
    sound_before: str | None = None
    sound_after: str | None = None


class ReplyRenderer:
    """Writes one reply's sentences as tags, a delivery only where it changes."""

    def __init__(self, space: LabelSpace):
        """Initialize the renderer.

        Args:
            space: The labels the call can use.
        """
        self._space = space
        self._delivery: str | None = None

    def render(self, sentence: str, labels: SentenceLabels) -> str:
        """Write the next sentence of the reply with its labels as tags.

        Labels the space doesn't hold, or that would overlap or cut a word, are
        dropped, so the tags always speak exactly the sentence's words.
        """
        text = sentence.strip()
        if not text:
            return text
        kept = _keep(text, labels, self._space)
        names = self._space.deliveries

        # (offset, order, tag): at one offset a close comes before an open.
        inserts: list[tuple[int, int, str]] = []
        for span in kept.spans:
            inserts += [(span.start, 1, f"<<{names[span.delivery]}: "), (span.end, 0, ">>")]
        for start, end in kept.emphasis:
            inserts += [(start, 1, "^^"), (end, 0, "^^")]
        for at, _, tag in sorted(inserts, key=lambda i: (i[0], i[1]), reverse=True):
            text = text[:at] + tag + text[at:]

        if kept.sound_before:
            text = f"[[{self._space.sound_name(kept.sound_before)}]] {text}"
        if kept.sound_after:
            text = f"{text} [[{self._space.sound_name(kept.sound_after)}]]"
        delivery = names[kept.delivery] if kept.delivery else None
        if delivery != self._delivery:
            text = f"<<{delivery or NEUTRAL_DELIVERY}>> {text}"
            self._delivery = delivery
        return text


def _trained_service(service_name: str) -> str:
    name = service_name.lower()
    return next((service for service in TRAINED_SERVICES if service in name), OTHER_SERVICE)


def _breathes(policy: ExpressionPolicy) -> bool:
    return ExpressionGuidance.BREATHE in policy.guidance


def _resolve(
    capabilities: ExpressionCapabilities, labels, offered: list[str] | None
) -> dict[str, str]:
    resolved = {}
    for label in labels:
        if name := standard_name(capabilities, label, offered):
            resolved[label] = name
    return resolved


def _keep(sentence: str, labels: SentenceLabels, space: LabelSpace) -> SentenceLabels:
    """The labels the space holds, on whole words and not overlapping."""
    taken: list[tuple[int, int]] = []
    spans = []
    for span in sorted(labels.spans, key=lambda s: s.start):
        if (
            span.delivery in space.deliveries
            and _words(sentence, span.start, span.end)
            and not _overlaps(taken, span.start, span.end)
        ):
            taken.append((span.start, span.end))
            spans.append(span)
    emphasis = []
    if space.emphasis:
        for start, end in sorted(labels.emphasis):
            if _words(sentence, start, end) and not _overlaps(taken, start, end):
                taken.append((start, end))
                emphasis.append((start, end))
    return SentenceLabels(
        delivery=labels.delivery if labels.delivery in space.deliveries else None,
        spans=spans,
        emphasis=emphasis,
        sound_before=labels.sound_before if space.sound_name(labels.sound_before or "") else None,
        sound_after=labels.sound_after if space.sound_name(labels.sound_after or "") else None,
    )


def _words(text: str, start: int, end: int) -> bool:
    return (
        0 <= start < end <= len(text)
        and not text[start].isspace()
        and not text[end - 1].isspace()
        and _boundary(text, start)
        and _boundary(text, end)
    )


def _boundary(text: str, at: int) -> bool:
    return at in (0, len(text)) or not (text[at - 1].isalnum() and text[at].isalnum())


def _overlaps(taken: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(start < e and s < end for s, e in taken)
