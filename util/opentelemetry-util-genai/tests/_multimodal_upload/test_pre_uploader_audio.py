"""
Test automatic audio format detection for MultimodalPreUploader
Focuses on testing the _detect_audio_format method's ability to recognize various audio formats
and audio format conversion (e.g., PCM16 to WAV)
"""

from pathlib import Path

import pytest

from opentelemetry.util.genai._multimodal_upload import MultimodalPreUploader
from opentelemetry.util.genai._multimodal_upload.pre_uploader import (
    _audio_libs_available,
)
from opentelemetry.util.genai.types import Blob, InputMessage

# Test audio file directory
TEST_AUDIO_DIR = Path(__file__).parent / "test_audio_samples"


class TestAudioFormatDetection:
    """Test automatic audio format detection functionality"""

    @pytest.fixture
    def pre_uploader(self):  # pylint: disable=R6301
        """Create PreUploader instance"""
        base_path = "/tmp/test_upload"
        return MultimodalPreUploader(base_path=base_path)

    @staticmethod
    def _read_audio_file(filename: str) -> bytes:
        """Read audio file"""
        filepath = TEST_AUDIO_DIR / filename
        if not filepath.exists():
            pytest.skip(f"Test audio file does not exist: {filepath}")

        with open(filepath, "rb") as file_obj:
            return file_obj.read()

    # ========== Edge Case Tests ==========

    @staticmethod
    def test_detect_unknown_format(pre_uploader):
        """Test unrecognizable format"""
        unknown_data = b"\x00" * 100
        detected = pre_uploader._detect_audio_format(unknown_data)
        assert detected is None

    @staticmethod
    def test_detect_insufficient_data(pre_uploader):
        """Test insufficient data case"""
        short_data = b"\x00" * 5
        detected = pre_uploader._detect_audio_format(short_data)
        assert detected is None

    # ========== Real Audio File Format Detection Tests ==========

    @staticmethod
    @pytest.mark.parametrize(
        "filename,expected_mime,file_header_check",
        [
            (
                "test.wav",
                "audio/wav",
                lambda d: d[:4] == b"RIFF" and d[8:12] == b"WAVE",
            ),
            (
                "test.mp3",
                "audio/mp3",
                lambda d: d[:3] == b"ID3"
                or (d[0] == 0xFF and (d[1] & 0xE0) == 0xE0),
            ),
            (
                "test.aac",
                "audio/aac",
                lambda d: d[0] == 0xFF and (d[1] & 0xF6) == 0xF0,
            ),
            ("test.m4a", "audio/m4a", lambda d: d[4:8] == b"ftyp"),
            ("test.ogg", "audio/ogg", lambda d: d[:4] == b"OggS"),
            ("test.flac", "audio/flac", lambda d: d[:4] == b"fLaC"),
            ("test.amr", "audio/amr", lambda d: d[:6] == b"#!AMR\n"),
            ("test.3gp", "audio/3gpp", lambda d: d[4:8] == b"ftyp"),
        ],
    )
    def test_detect_real_audio_format(
        pre_uploader, filename, expected_mime, file_header_check
    ):
        """Test real audio file format detection"""
        data = TestAudioFormatDetection._read_audio_file(filename)

        # Verify file header
        assert file_header_check(data), f"{filename} file header is incorrect"

        # Test format detection
        detected = pre_uploader._detect_audio_format(data)
        # AMR and 3GP may have multiple MIME types
        if expected_mime == "audio/amr":
            assert detected in ("audio/amr", "audio/amr-wb"), (
                f"{filename} format detection failed, detected: {detected}"
            )
        elif expected_mime == "audio/3gpp":
            assert detected in ("audio/3gpp", "audio/3gpp2"), (
                f"{filename} format detection failed, detected: {detected}"
            )
        else:
            assert detected == expected_mime, (
                f"{filename} format detection failed, expected {expected_mime}, got {detected}"
            )

    # ========== PCM16 to WAV Conversion Tests ==========

    @staticmethod
    @pytest.mark.parametrize(
        "pcm_mime_type",
        [
            "audio/pcm16",
            "audio/l16",
            "audio/pcm",
        ],
    )
    def test_pcm16_to_wav_conversion(pre_uploader, pcm_mime_type):
        """Test PCM16 to WAV format conversion"""
        # Create simulated PCM16 data
        pcm_data = b"\x00\x01" * 1000

        part = Blob(
            content=pcm_data, mime_type=pcm_mime_type, modality="audio"
        )

        input_message = InputMessage(role="user", parts=[part])
        input_messages = [input_message]

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        assert len(uploads) == 1
        # Verify result based on library availability
        if _audio_libs_available:
            assert uploads[0].content_type == "audio/wav"
            assert uploads[0].url.endswith(".wav")
        else:
            # If library unavailable, should keep original format
            assert uploads[0].content_type == pcm_mime_type
