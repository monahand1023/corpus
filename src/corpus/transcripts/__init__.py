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
    TranscriptVerdict,
    impossible_speech_rate,
    judge_transcript,
    only_unspoken_languages,
    repeat_share,
    subtitle_boilerplate,
)

__all__ = [
    "TranscriptVerdict",
    "impossible_speech_rate",
    "judge_transcript",
    "only_unspoken_languages",
    "repeat_share",
    "subtitle_boilerplate",
]
