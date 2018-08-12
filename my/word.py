from typing import Optional, Dict, Union
from my import Phonetic


class Word(object):
    STRESSED_CHAR = chr(769)

    text: str
    stressed_index: int
    dictionary_name: str
    normal_form: Optional['Word'] = None
    is_normal_form: Optional[bool] = None
    frequency: float = 0

    def __init__(self, text: str, stressed_index: int, dictionary_name: str):
        self.text = text
        self.stressed_index = stressed_index
        self.dictionary_name = dictionary_name
        self._cached_phonetic_fuzzy = None

    def stressed(self) -> str:
        stressed = self.text[self.stressed_index]
        vowels = 0
        for char in self.text.lower():
            if char in Phonetic.VOWELS:
                vowels += 1
        if vowels < 2 or stressed == 'ё' or stressed == 'Ё':
            return self.text
        else:
            prev = self.text[:self.stressed_index]
            post = self.text[self.stressed_index + 1:]
            return prev + stressed + Word.STRESSED_CHAR + post

    def phonetic(self, fuzzy: int = 0) -> str:
        if self._cached_phonetic_fuzzy is None:
            self._cached_phonetic_fuzzy = dict()
        cached = self._cached_phonetic_fuzzy.get(fuzzy)
        if cached is None:
            val = Phonetic.phonetic(self.text.lower(), self.stressed_index, fuzzy)
            self._cached_phonetic_fuzzy[fuzzy] = val
            return val
        else:
            return cached

    def phonetic_after_stress(self, fuzzy: int = 0):
        """Inclusive stressed character"""
        phonetic_value = self.phonetic(fuzzy=fuzzy)
        if len(phonetic_value) == self.stressed_index + 1 and self.stressed_index > 0:
            return phonetic_value[self.stressed_index - 1:]
        else:
            return phonetic_value[self.stressed_index:]

    def syllables(self):
        syllables, stressed_syllable_index = Phonetic.syllables(self.text, self.stressed_index)
        return syllables

    def stressed_syllable_index(self):
        syllables, stressed_syllable_index = Phonetic.syllables(self.text, self.stressed_index)
        return stressed_syllable_index

    def __str__(self):
        return "<Word %s>" % (self.stressed())

    def __repr__(self):
        return "<Word %s>" % (self.stressed())

    def __hash__(self):
        return (self.text, self.stressed_index).__hash__()

    def __eq__(self, other):
        return self.text == other.text and self.stressed_index == other.stressed_index

    @classmethod
    def from_stressed(cls,
                      stressed_text: str,
                      dictionary_name: str,
                      stressed_char: str,
                      stress_char_after) -> Optional['Word']:
        text = stressed_text.replace(stressed_char, '')
        if stress_char_after:
            stressed_index = stressed_text.find(stressed_char) - 1
        else:
            stressed_index = stressed_text.find(stressed_char)
        if stressed_index < 0:
            first_vowel_index = None
            many_vowels = False
            yo_vowel_index = None
            for index, char in enumerate(stressed_text.lower()):
                if char in Phonetic.VOWELS:
                    if first_vowel_index is None:
                        first_vowel_index = index
                    else:
                        many_vowels = True

                    if char == 'ё':
                        yo_vowel_index = index

            if first_vowel_index is not None and not many_vowels:
                stressed_index = first_vowel_index
            elif yo_vowel_index is not None:
                stressed_index = yo_vowel_index
            else:
                #raise RuntimeError("Stress char not found and word contains more then one vowels: %s" % stressed_text)
                return None
        if text[stressed_index].lower() not in Phonetic.VOWELS:
            return None
        return cls(text, stressed_index, dictionary_name)

    @classmethod
    def allow_word(cls, text: str) -> bool:
        for c in text:
            if not c in Phonetic.ALLOWED_CHARS:
                return False
        if len(text) < 2 or text[0] == '-' or text[-1] == '-':
            return False
        return True