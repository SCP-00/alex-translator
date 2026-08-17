"""Unit tests for alex-translator config module."""

import pytest


class TestLangMap:
    def test_contains_core_languages(self):
        from config import LANG_MAP
        for code in ('en', 'es', 'ja'):
            assert code in LANG_MAP

    def test_values_are_proper_names(self):
        from config import LANG_MAP
        for code, name in LANG_MAP.items():
            assert isinstance(code, str) and code
            assert isinstance(name, str) and name
            assert name[0].isupper()


class TestPairModelMap:
    def test_known_pairs_have_models(self):
        from config import PAIR_MODEL_MAP
        for pair in [('en', 'es'), ('es', 'en'), ('en', 'ja'), ('ja', 'en'), ('ja', 'es')]:
            assert pair in PAIR_MODEL_MAP, f"Missing model for {pair}"

    def test_unknown_pair_has_no_model(self):
        from config import PAIR_MODEL_MAP
        assert ('fr', 'de') not in PAIR_MODEL_MAP


class TestPivotPairs:
    def test_pivot_through_english(self):
        from config import PIVOT_PAIRS
        assert PIVOT_PAIRS[('es', 'ja')] == ('es', 'en', 'ja')
        assert PIVOT_PAIRS[('ja', 'es')] == ('ja', 'en', 'es')


class TestIdiomMap:
    def test_idioms_are_tuples(self):
        from config import IDIOM_MAP
        for lang, idioms in IDIOM_MAP.items():
            assert lang in ('en', 'es')
            assert isinstance(idioms, list)
            for idiom, replacement in idioms:
                assert isinstance(idiom, str) and idiom
                assert isinstance(replacement, str) and replacement

    def test_en_has_break_a_leg(self):
        from config import IDIOM_MAP
        en_idioms = {i.lower() for i, _ in IDIOM_MAP['en']}
        assert 'break a leg' in en_idioms

    def test_es_has_lloviendo_a_cantaros(self):
        from config import IDIOM_MAP
        es_idioms = {i.lower() for i, _ in IDIOM_MAP['es']}
        assert 'lloviendo a cántaros' in es_idioms


class TestKokoroConfig:
    def test_es_en_ja_configured(self):
        from config import KOKORO_CONFIG
        for lang in ('es', 'en', 'ja'):
            assert lang in KOKORO_CONFIG
            cfg = KOKORO_CONFIG[lang]
            assert 'voice' in cfg
            assert 'lang' in cfg
            assert cfg['voice']
