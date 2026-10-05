"""Catch a model repeating itself while it streams (task 317).

Small models fall into loops: the same sentence over and over (gpt-oss:20b),
or the same paragraph re-derived with near-identical wording (nemotron-nano).
The watch counts how many 8-word phrases in the last `window` words were
already said earlier in the call. Healthy reasoning stays under ~0.45 (it
re-quotes numbers and names); a loop goes to 1.0."""
import re
from collections import Counter
from dataclasses import dataclass

_WORD = re.compile(r"\w+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Loop:
    start: int    # char offset in the fed text where the looping part begins
    sample: str   # one of the repeated sentences
    words: int    # words seen when the loop was caught


class RepetitionWatch:
    def __init__(self, window: int = 600, threshold: float = 0.7,
                 every: int = 100, n: int = 8):
        self.window, self.threshold, self.every, self.n = window, threshold, every, n
        self._text = ""
        self._scan = 0
        self._words: list[tuple[str, int]] = []
        self._seen: dict[tuple, int] = {}
        self._flags: list[bool] = []

    def feed(self, chunk: str) -> Loop | None:
        """Add streamed text; return a Loop once the window is mostly repeats."""
        self._text += chunk
        for m in _WORD.finditer(self._text, self._scan):
            if m.end() == len(self._text):
                break  # the word may continue in the next chunk
            self._scan = m.end()
            self._words.append((m.group().lower(), m.start()))
            if len(self._words) < self.n:
                continue
            shingle = tuple(w for w, _ in self._words[-self.n:])
            k = len(self._flags)
            self._flags.append(shingle in self._seen)
            self._seen.setdefault(shingle, k)
            if (k + 1 >= self.window and (k + 1) % self.every == 0
                    and sum(self._flags[-self.window:]) >= self.threshold * self.window):
                return self._loop()
        return None

    def _loop(self) -> Loop:
        flags = self._flags
        # onset: walk back over blocks that are mostly repeats ...
        k = len(flags)
        while k - self.every >= 0 and sum(flags[k - self.every:k]) * 2 >= self.every:
            k -= self.every
        # ... then to the first repeat of that run, then to where the repeated
        # text was first said, if that is near (else it is a common phrase)
        while k > 0 and flags[k - 1]:
            k -= 1
        while k < len(flags) and not flags[k]:
            k += 1
        shingle = tuple(w for w, _ in self._words[k:k + self.n])
        first = self._seen.get(shingle, k)
        if k - first <= self.window:
            k = first
        start = self._words[k][1]
        return Loop(start, self._sample(self._text[:start], self._text[start:]),
                    len(self._words))

    @staticmethod
    def _sample(before: str, text: str) -> str:
        """The most repeated sentence of the loop, preferring one not said
        before it and then a longer one (short ones are often stock phrases)."""
        sentences = [s.strip() for s in _SENTENCE_END.split(text) if len(s.strip()) >= 20]
        if not sentences:
            return text.strip()[:300]
        counts = Counter(sentences)
        best = max(counts, key=lambda s: (counts[s] >= 2, s not in before, counts[s], len(s)))
        return (best if counts[best] >= 2 else sentences[0])[:300]


def loop_nudge(e, act: str, keep: int = 40_000) -> str:
    """Re-ask after a RepetitionLoopDetected `e`: one looped sentence and the
    text from before the loop (its last `keep` chars), never the loop itself
    (carried back, it is continued). `act` says what to do now."""
    text = (f"You started repeating yourself and were stopped. You kept saying: "
            f"\"{e.sample}\" That point is settled; do not go back to it. Take the "
            f"most likely reading and {act}")
    before = e.before
    if len(before) > keep:
        before = "(earlier part omitted)\n" + before[-keep:]
    if before.strip():
        text += "\n\nWhat you wrote before you started repeating:\n" + before
    return text
