"""Quality signals for machine-transcribed audio.

Speech-to-text models invent text. Whisper in particular was trained on audio
paired with scraped subtitles, so silence and ambience in its training data
were frequently paired with caption boilerplate -- "Thanks for watching",
"Subtitles by ...". It learned that mapping, and reproduces it whenever it is
handed audio without speech in it.

This is not a decoding bug and no confidence threshold fixes it: measured on
one archive, twenty seconds of generated silence produced "Thank you." at
no_speech=0.782 and avg_logprob=-0.24. The model is CONFIDENTLY wrong, so
confidence cannot separate invention from speech.

What works is judging the text.
"""

from corpus.transcripts.quality import (
    CAPTION_SIGNOFF_TAILS,
    SUBTITLE_BOILERPLATE,
    SUBTITLE_CREDIT_PREFIXES,
    TranscriptVerdict,
    impossible_speech_rate,
    judge_transcript,
    looping_share,
    only_unspoken_languages,
    repeat_share,
    strip_caption_tail,
    subtitle_boilerplate,
)
from corpus.transcripts.store import (
    SCHEMA,
    Transcript,
    Window,
    already_done,
    connect,
    open_store,
    policy_fingerprint,
)

__all__ = [
    "CAPTION_SIGNOFF_TAILS",
    "SCHEMA",
    "SUBTITLE_BOILERPLATE",
    "SUBTITLE_CREDIT_PREFIXES",
    "Transcript",
    "TranscriptVerdict",
    "Window",
    "already_done",
    "connect",
    "impossible_speech_rate",
    "judge_transcript",
    "looping_share",
    "only_unspoken_languages",
    "open_store",
    "policy_fingerprint",
    "repeat_share",
    "strip_caption_tail",
    "subtitle_boilerplate",
]
