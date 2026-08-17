"""Unit tests for alex-translator translate module.

Tests the pure logic (idiom replacement, language detection) and the
translation flow without transformers installed (HAVE_TRANSFORMERS=False).
"""

import pytest
from unittest.mock import patch


class TestReplaceIdioms:
    def test_english_idiom_replaced(self):
        from translate import _replace_idioms
        result = _replace_idioms("Good luck, break a leg!", "en")
        assert "break a leg" not in result
        assert "good luck" in result.lower()

    def test_spanish_idiom_replaced(self):
        from translate import _replace_idioms
        result = _replace_idioms("Está lloviendo a cántaros hoy.", "es")
        assert "lloviendo a cántaros" not in result
        assert "lloviendo mucho" in result

    def test_no_idiom_unchanged(self):
        from translate import _replace_idioms
        text = "The weather is beautiful today."
        assert _replace_idioms(text, "en") == text

    def test_word_boundary_protection(self):
        """'once in a blue moon' inside 'once in a blue moonlight' must NOT match."""
        from translate import _replace_idioms
        text = "I saw it once in a blue moonlight."
        result = _replace_idioms(text, "en")
        # 'blue moonlight' is a partial word match → not replaced
        assert "blue moonlight" in result

    def test_case_preserved(self):
        from translate import _replace_idioms
        result = _replace_idioms("Break a leg!", "en")
        assert result.startswith("Good luck") or result.startswith("good luck")
        assert "break a leg" not in result.lower()

    def test_unknown_lang_unchanged(self):
        from translate import _replace_idioms
        text = "Hello world"
        assert _replace_idioms(text, "fr") == text


class TestDetectLanguage:
    def test_detect_english(self):
        from translate import detect_language
        assert detect_language("Hello, how are you?") == "en"

    def test_detect_spanish(self):
        from translate import detect_language
        assert detect_language("Hola, ¿cómo estás?") == "es"

    def test_detect_japanese(self):
        from translate import detect_language
        assert detect_language("こんにちは、元気ですか？") == "ja"

    def test_detect_empty(self):
        from translate import detect_language
        assert detect_language("") == "en"

    def test_detect_chinese_characters(self):
        from translate import detect_language
        assert detect_language("你好世界") == "ja"  # kanji-only heuristic treats as ja


class TestTranslateNoTransformers:
    def test_returns_error_without_transformers(self):
        with patch("translate.HAVE_TRANSFORMERS", False):
            from translate import translate
            result, error = translate("Hello", "en", "es")
            assert result is None
            assert error is not None

    def test_same_language_returns_text(self):
        # Con transformers disponible, mismo idioma devuelve el texto tal cual
        with patch("translate.HAVE_TRANSFORMERS", True):
            from translate import translate
            result, error = translate("Hello", "en", "en")
            assert result == "Hello"
            assert error is None


class TestGetTranslationModel:
    def test_returns_none_without_transformers(self):
        with patch("translate.HAVE_TRANSFORMERS", False):
            from translate import _get_translation_model
            assert _get_translation_model("en", "es") is None
