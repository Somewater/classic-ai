# from https://how-to-all.com/фонетика
from typing import List, Optional, Tuple, Union


class Phonetic(object):
    VOWELS = {'а', 'о', 'у', 'э', 'ы', 'и', 'е', 'ю', 'я', 'ё'}
    VOWEL_STRESS_PAIRS = {'ю': 'у',
                          'я': 'а',
                          'э': 'е',
                          'ё': 'о',
                          'ы': 'и'}
    CONS = {'б', 'в', 'г', 'д', 'ж', 'з', 'й', 'к', 'л', 'м', 'н',
            'п', 'р', 'с', 'т', 'ф', 'х', 'ц', 'ш', 'щ', 'ч'}
    VOISED_TO_VOISLESS = {'б': 'п', 'в': 'ф', 'г': 'к', 'д': 'т', 'ж': 'ш', 'з': 'с'}
    VOISLESS_TO_VOISED = dict(map(lambda x: (x[1], x[0]), VOISED_TO_VOISLESS.items()))
    VOISED_CONS = set(VOISED_TO_VOISLESS.keys())
    VOISELESS_CONS = set(VOISED_TO_VOISLESS.values())
    HARD_CONS = {'ш', 'ж', 'ц'}
    HISSING_CONS = {'ш', 'ж'}
    WORD_DELIMITERS = {' ', '-'}
    ALLOWED_CHARS = {chr(code) for code in range(ord('А'), ord('я') + 1)} | set(('-', 'ё', 'Ё'))

    FUZZY_CONS_STUNING =      0b0001
    FUZZY_CUTOFF_FINAL_CONS = 0b0010

    @staticmethod
    def phonetic(word: str, stressed: int, fuzzy: int = 0) -> str:
        result = []
        prev_char = None
        next_char = None
        full_len = len(word)
        for index, char in enumerate(word):
            if index + 1 < full_len:
                next_char = word[index + 1]
            else:
                next_char = None
            # https://is.muni.cz/do/ped/kat/KRus/fonetika/ch10.html
            if char in Phonetic.VOWELS:
                is_stressed = index == stressed
                vowel = Phonetic._vowel_reductions(char, is_stressed, next_char)
                if is_stressed:
                    vowel = vowel.upper()
                result.append(vowel)
            elif char in Phonetic.CONS:
                if fuzzy & Phonetic.FUZZY_CONS_STUNING > 0:
                    result.append(Phonetic.VOISED_TO_VOISLESS.get(char) or char)
                else:
                    result.append(Phonetic._cons_transform(char, next_char))
            else:
                result.append(char)
            prev_char = char
        if fuzzy & Phonetic.FUZZY_CUTOFF_FINAL_CONS > 0 and (result[-1] in Phonetic.CONS or result[-1] == 'ь'):
            result.pop()
        return "".join(result)

    @staticmethod
    def syllables(word: str, stressed: int = 0) -> Optional[Tuple[List[str], int]]:
        """
        :param word:
        :param stressed:
        :return: (syllables, index_of_stressed_syllable)
        """
        syllable = []
        syllables = []
        stressed_syllable_index = None
        for index, char in enumerate(word.lower()):
            if char != '-':
                syllable.append(char)
            if index == stressed:
                stressed_syllable_index = len(syllables)
            if syllable and (char in Phonetic.VOWELS or char == '-'):
                if char == '-':
                    if not syllables:
                        return None
                    last_syllable = syllables[-1]
                    syllables[-1] = last_syllable + "".join(syllable)
                else:
                    syllables.append("".join(syllable))
                syllable.clear()
        if syllable:
            if not syllables:
                return None
            last_syllable = syllables[-1]
            syllables[-1] = last_syllable + "".join(syllable)
        if stressed_syllable_index == len(syllables):
            raise RuntimeError('stressed_syllable_index = %d on word %s using syllables %s' %
                               (stressed_syllable_index, word, repr(syllables)))
            # stressed_syllable_index -= 1
        return syllables, stressed_syllable_index

    @staticmethod
    def _vowel_reductions(char: str, is_stressed: bool, next_char: Optional[str]) -> str:
        if is_stressed:
            # can reduct only 'Э' on stress
            if char == 'э':
                return 'е'
            else:
                return char
        else:
            if char == 'э':
                return 'е'
            elif char == 'ё':
                return 'о'
            elif char == 'о':
                return 'а'
            elif char == 'е':
                if next_char is None or next_char in Phonetic.WORD_DELIMITERS:
                    return char
                else:
                    return 'и'
            elif char == 'я':
                if next_char is None or next_char in Phonetic.WORD_DELIMITERS:
                    return char
                else:
                    return 'и'
            else:
                return char

    @staticmethod
    def _cons_transform(char: str, next_char: Optional[str]):
        if char in Phonetic.VOISED_CONS:
            if next_char is None or next_char in Phonetic.VOISELESS_CONS:
                return Phonetic.VOISED_TO_VOISLESS[char]
            else:
                return char
        elif char in Phonetic.VOISELESS_CONS:
            if next_char in Phonetic.VOISED_CONS and next_char != 'в':
                return Phonetic.VOISLESS_TO_VOISED[char]
            else:
                return char
        else:
            return char


