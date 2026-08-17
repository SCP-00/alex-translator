"""Integration tests for alex-translator HTTP handler.

Uses a real HTTP server with external services (ASR/translate/TTS)
mocked — handler routing, validation and JSON serialization stay real.
"""

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import HTTPServer
from unittest.mock import patch, Mock

import pytest


@pytest.fixture
def server_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


@pytest.fixture
def mock_services():
    """Mock external services (GPU/models), keep handler logic real."""
    import asr, translate, tts
    fake_path = Mock()
    fake_path.exists.return_value = False
    with patch.object(asr, 'HAVE_WHISPER', False), \
         patch.object(translate, 'HAVE_TRANSFORMERS', False), \
         patch.object(tts, 'HAVE_KOKORO', False), \
         patch.object(tts, 'HAVE_PIPER_PYTHON', False), \
         patch.object(tts, 'KOKORO_MODEL_PATH', fake_path), \
         patch.object(tts, 'KOKORO_VOICES_PATH', fake_path):
        yield


@pytest.fixture
def running_server(server_port, mock_services):
    from handler import ThreadingHTTPServer, TranslatorHandler
    server = ThreadingHTTPServer(('127.0.0.1', server_port), TranslatorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    yield f'http://127.0.0.1:{server_port}'
    server.shutdown()
    server.server_close()


class TestStatusEndpoint:
    def test_status_returns_json(self, running_server):
        with urllib.request.urlopen(f"{running_server}/api/status") as resp:
            data = json.loads(resp.read().decode())
        assert resp.status == 200
        assert 'whisper_loaded' in data
        assert 'transformers_loaded' in data
        assert 'kokoro_loaded' in data
        assert isinstance(data['languages'], list)


class TestTranslateEndpoint:
    def test_empty_text_returns_error(self, running_server):
        req = urllib.request.Request(
            f"{running_server}/api/translate",
            data=json.dumps({"text": ""}).encode(),
            headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        assert 'error' in data

    def test_translate_without_models_returns_error(self, running_server):
        req = urllib.request.Request(
            f"{running_server}/api/translate",
            data=json.dumps({"text": "Hello", "to_lang": "es"}).encode(),
            headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        assert 'error' in data


class TestTTSEndpoint:
    def test_empty_text_returns_error(self, running_server):
        req = urllib.request.Request(
            f"{running_server}/api/tts",
            data=json.dumps({"text": ""}).encode(),
            headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        assert 'error' in data

    def test_tts_without_models_returns_error(self, running_server):
        req = urllib.request.Request(
            f"{running_server}/api/tts",
            data=json.dumps({"text": "Hola", "language": "Spanish"}).encode(),
            headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        assert 'error' in data


class TestASREndpoint:
    def test_empty_audio_returns_error(self, running_server):
        req = urllib.request.Request(
            f"{running_server}/api/asr",
            data=json.dumps({"audio": ""}).encode(),
            headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        assert 'error' in data


class TestPipelineEndpoint:
    def test_empty_audio_returns_error(self, running_server):
        req = urllib.request.Request(
            f"{running_server}/api/pipeline",
            data=json.dumps({"audio": ""}).encode(),
            headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        assert 'error' in data


class TestCORSAndRouting:
    def test_options_returns_cors_headers(self, running_server):
        req = urllib.request.Request(f"{running_server}/api/translate", method='OPTIONS')
        with urllib.request.urlopen(req) as resp:
            headers = dict(resp.headers)
        assert resp.status == 200
        assert headers.get('Access-Control-Allow-Origin') == '*'

    def test_unknown_path_returns_404(self, running_server):
        req = urllib.request.Request(f"{running_server}/api/unknown", method='GET')
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 404
