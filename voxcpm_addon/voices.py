"""The voices the add-on offers, and the languages VoxCPM2 speaks.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Two kinds of voice:

- `voxcpm-clone` clones the speaker from the video: the host hands over a
  clip of their speech, and its transcript when it has one.
- The others are made from a description. VoxCPM2 makes a voice up from a few
  words, but a different one each time, and a dub needs the same voice on every
  line. So a described voice is made once per speaker, as a fixed sentence in
  the line's language (an "anchor"), cached, and every line clones the anchor.
"""

from __future__ import annotations

from dataclasses import dataclass

CLONE = 'voxcpm-clone'
DESCRIBED = 'voxcpm-described'


@dataclass(frozen=True)
class Voice:
    id: str
    display_name: str
    description: str = ''   # empty for the clone, and for DESCRIBED (from the settings)


VOICES = (
    Voice(CLONE, "Speaker's own voice (cloned from the video)"),
    Voice('voxcpm-warm-woman', 'Warm woman',
          'An adult woman, warm and friendly voice, clear and natural'),
    Voice('voxcpm-young-woman', 'Bright young woman',
          'A young woman, bright and lively voice'),
    Voice('voxcpm-older-woman', 'Gentle older woman',
          'An elderly woman, soft and gentle voice, slightly husky'),
    Voice('voxcpm-calm-man', 'Calm man',
          'A middle-aged man, deep and calm voice, steady pace'),
    Voice('voxcpm-young-man', 'Energetic young man',
          'A young man, clear and energetic voice'),
    Voice('voxcpm-older-man', 'Gravelly older man',
          'An elderly man, low and gravelly voice, unhurried'),
    Voice('voxcpm-narrator', 'Narrator',
          'A professional narrator, rich and articulate male voice, measured pace'),
    Voice(DESCRIBED, 'Voice you describe (in Configure)'),
)
BY_ID = {voice.id: voice for voice in VOICES}

# The 30 languages of the VoxCPM2 model card, as BCP-47 primary subtags. The
# model reads the language from the text; it takes no language parameter.
LANGUAGES = (
    'ar', 'my', 'zh', 'da', 'nl', 'en', 'fi', 'fr', 'de', 'el', 'he', 'hi', 'id', 'it', 'ja',
    'km', 'ko', 'lo', 'ms', 'no', 'pl', 'pt', 'ru', 'es', 'sw', 'sv', 'tl', 'th', 'tr', 'vi',
)

# Anchor sentences: about ten seconds of plain speech, worded so that no form
# depends on the speaker's gender. In a language with none here, the anchor is
# English and lines clone only its timbre, so English cadence does not carry
# over into them.
ANCHORS = {
    'en': 'Good morning. Today we have a long road ahead of us, so it would be wise to set off early, '
          'before the rain begins.',
    'pt': 'Bom dia. Hoje temos um longo caminho pela frente, então é melhor sair cedo, '
          'antes que a chuva comece.',
    'es': 'Buenos días. Hoy tenemos un largo camino por delante, así que conviene salir temprano, '
          'antes de que empiece a llover.',
    'fr': "Bonjour. Aujourd'hui, nous avons une longue route devant nous, alors mieux vaut partir tôt, "
          'avant que la pluie ne commence.',
    'de': 'Guten Morgen. Heute liegt ein langer Weg vor uns, also brechen wir besser früh auf, '
          'bevor der Regen beginnt.',
    'it': 'Buongiorno. Oggi abbiamo una lunga strada davanti a noi, quindi conviene partire presto, '
          'prima che cominci a piovere.',
    'nl': 'Goedemorgen. Vandaag hebben we een lange weg voor de boeg, dus we kunnen beter vroeg vertrekken, '
          'voordat het gaat regenen.',
    'pl': 'Dzień dobry. Dziś czeka nas długa droga, więc lepiej wyruszyć wcześnie, '
          'zanim zacznie padać deszcz.',
    'ru': 'Доброе утро. Сегодня у нас впереди долгая дорога, поэтому лучше выйти пораньше, '
          'пока не начался дождь.',
    'zh': '早上好。今天我们还有很长的路要走，所以最好早点出发，趁着还没下雨。',
    'ja': 'おはようございます。今日は長い道のりになりそうなので、雨が降り出す前に早めに出発しましょう。',
    'ko': '좋은 아침이에요. 오늘은 갈 길이 멀어서, 비가 오기 전에 일찍 출발하는 게 좋겠어요.',
}


def primary_language(tag) -> str:
    """`pt-BR` -> `pt`; '' when there is none."""
    return str(tag or '').strip().lower().replace('_', '-').split('-', 1)[0]


def anchor_for(tag) -> tuple[str, str, bool]:
    """(language, sentence, native) for the line's language tag.

    `native` is False when the sentence is the English fallback.
    """
    language = primary_language(tag)
    if language in ANCHORS:
        return language, ANCHORS[language], True
    return 'en', ANCHORS['en'], False


def manifest_voices() -> list[dict]:
    """The voice list as manifest.json and the hello frame carry it."""
    out = []
    for voice in VOICES:
        entry = {'id': voice.id, 'language': '*', 'display_name': voice.display_name}
        if voice.id == CLONE:
            entry['requires'] = ['voice_ref_audio']
        out.append(entry)
    return out
