"""Reference voices supplied by Sesame's upstream run_csm.py demo."""
from functools import lru_cache

VOICE_OPTIONS = {
    'random': 'Random voice',
    'conversational_a': 'Conversational A',
    'conversational_b': 'Conversational B',
}
PROMPT_TEXT = {
    'conversational_a': (
        "like revising for an exam I'd have to try and like keep up the momentum because I'd "
        "start really early I'd be like okay I'm gonna start revising now and then like "
        "you're revising for ages and then I just like start losing steam I didn't do that "
        "for the exam we had recently to be fair that was a more of a last minute scenario "
        "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
        "sort of start the day with this not like a panic but like a"
    ),
    'conversational_b': (
        "like a super Mario level. Like it's very like high detail. And like, once you get "
        "into the park, it just like, everything looks like a computer game and they have all "
        "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
        "will have like a question block. And if you like, you know, punch it, a coin will "
        "come out. So like everyone, when they come into the park, they get like this little "
        "bracelet and then you can go punching question blocks around."
    ),
}


@lru_cache(maxsize=2)
def prompt_path(voice):
    if voice not in PROMPT_TEXT:
        raise ValueError('Choose a reference voice.')
    from huggingface_hub import hf_hub_download
    return hf_hub_download('sesame/csm-1b', f'prompts/{voice}.wav')


@lru_cache(maxsize=4)
def _prompt_segment(voice, sample_rate):
    import torchaudio
    from generator import Segment
    audio, rate = torchaudio.load(prompt_path(voice))
    audio = torchaudio.functional.resample(audio.mean(dim=0), rate, sample_rate)
    return Segment(speaker=0, text=PROMPT_TEXT[voice], audio=audio)


def voice_context(voice, sample_rate):
    if voice not in VOICE_OPTIONS:
        raise ValueError('Choose one of the listed voices.')
    return [] if voice == 'random' else [_prompt_segment(voice, sample_rate)]
