"""Reference voices supplied by Sesame's upstream run_csm.py demo."""
from functools import lru_cache

VOICE_OPTIONS = {
    'random': 'Random (no reference)',
    'conversational_a': 'Exam conversation sample',
    'conversational_b': 'Mario park conversation sample',
    'read_speech_a': "Reading 1: Lake's gaze",
    'read_speech_b': 'Reading 2: Boy and cap',
    'read_speech_c': 'Reading 3: Tree story',
    'read_speech_d': 'Reading 4: Old days',
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
    'read_speech_a': (
        'And Lake turned round upon me, a little abruptly, his odd yellowish eyes, a little '
        'like those of the sea eagle, and the ghost of his smile that flickered on his '
        'singularly pale face, with a stern and insidious look, confronted me.'
    ),
    'read_speech_b': (
        'He was such a big boy that he wore high boots and carried a jack knife. He gazed and '
        'gazed at the cap, and could not keep from fingering the blue tassel.'
    ),
    'read_speech_c': (
        'All passed so quickly, there was so much going on around him, the Tree quite forgot '
        'to look to himself.'
    ),
    'read_speech_d': (
        'Suddenly I was back in the old days Before you felt we ought to drift apart. It was '
        'some trick-the way your eyebrows raise.'
    ),
}


@lru_cache(maxsize=6)
def prompt_path(voice):
    if voice not in PROMPT_TEXT:
        raise ValueError('Choose a reference voice.')
    from huggingface_hub import hf_hub_download
    return hf_hub_download('sesame/csm-1b', f'prompts/{voice}.wav')


@lru_cache(maxsize=6)
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
