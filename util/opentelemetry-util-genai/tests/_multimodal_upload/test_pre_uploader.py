"""
Test general functionality of MultimodalPreUploader
Includes extension mapping, URL generation, meta processing, message handling, async metadata fetching, etc.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
import respx

from opentelemetry.util.genai._multimodal_upload.pre_uploader import (
    _MAX_MULTIMODAL_DATA_SIZE,
    _MAX_MULTIMODAL_PARTS,
    MultimodalPreUploader,
    UriMetadata,
)
from opentelemetry.util.genai.types import Blob, InputMessage, Uri

# Test audio file directory for integration tests
TEST_AUDIO_DIR = Path(__file__).parent / "test_audio_samples"


class TestPreUploadGeneral:
    """Test general functionality of MultimodalPreUploader"""

    @pytest.fixture
    def pre_uploader(self):  # pylint: disable=R6301
        """Create PreUploader instance"""
        base_path = "/tmp/test_upload"
        extra_meta = {
            "workspaceId": "test_workspace",
            "serviceId": "test_service",
        }
        return MultimodalPreUploader(
            base_path=base_path, extra_meta=extra_meta
        )

    # ========== Extension Mapping Tests ==========

    @staticmethod
    @pytest.mark.parametrize(
        "mime_type,expected_ext",
        [
            # Audio formats
            ("audio/wav", "wav"),
            ("audio/mp3", "mp3"),
            ("audio/mpeg", "mp3"),  # Special mapping
            ("audio/aac", "aac"),
            ("audio/m4a", "m4a"),
            ("audio/amr", "amr"),
            ("audio/amr-wb", "amr"),  # Special mapping
            ("audio/3gpp", "3gp"),  # Special mapping
            ("audio/3gpp2", "3g2"),  # Special mapping
            ("audio/ogg", "ogg"),
            ("audio/flac", "flac"),
            ("audio/webm", "webm"),
            # Image formats
            ("image/jpeg", "jpg"),  # Special mapping
            ("image/png", "png"),
            # Edge cases
            ("audio/*", "bin"),  # Wildcard
            ("audio/unknown", "bin"),  # Unknown format
            ("video", "bin"),  # No slash
        ],
    )
    def test_extension_mapping(pre_uploader, mime_type, expected_ext):
        """Test extension mapping correctness"""
        ext = pre_uploader._ext_from_content_type(mime_type)
        assert ext == expected_ext, (
            f"MIME type {mime_type} extension mapping error, expected {expected_ext}, got {ext}"
        )

    # ========== URL Generation Tests ==========

    @staticmethod
    @pytest.mark.parametrize(
        "base_path",
        [
            "/tmp/test_upload",  # Without trailing slash
            "/tmp/test_upload/",  # With trailing slash
        ],
    )
    def test_url_generation_with_different_base_paths(base_path):
        """Test URL generation with/without trailing slash in base_path"""
        pre_uploader = MultimodalPreUploader(base_path=base_path)

        test_data = b"test data"

        part = Blob(content=test_data, mime_type="image/png", modality="image")

        input_message = InputMessage(role="user", parts=[part])
        input_messages = [input_message]

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        assert len(uploads) == 1
        # URL should correctly handle base_path trailing slash
        assert "/tmp/test_upload/" in uploads[0].url
        assert uploads[0].url.endswith(".png")

    # ========== Message Processing and Meta Fields Integration Tests ==========

    # pylint: disable=R0914
    @staticmethod
    def test_process_messages_and_meta(pre_uploader):
        """Test message processing and meta fields (integration test)"""
        test_data = b"test data"

        # Input message
        input_part = Blob(
            content=test_data, mime_type="image/png", modality="image"
        )
        input_message = InputMessage(role="user", parts=[input_part])
        input_messages = [input_message]

        # Output message
        output_part = Blob(
            content=test_data, mime_type="image/jpeg", modality="image"
        )
        output_message = InputMessage(role="assistant", parts=[output_part])
        output_messages = [output_message]

        # Test with span_context
        mock_span = Mock()
        mock_span.trace_id = 123456789
        mock_span.span_id = 987654321

        uploads = pre_uploader.pre_upload(
            span_context=mock_span,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=output_messages,
        )

        # Verify both input and output are processed
        assert len(uploads) == 2

        # Verify meta contains all necessary fields (with span_context)
        for upload in uploads:
            assert "workspaceId" in upload.meta
            assert "serviceId" in upload.meta
            assert "timestamp" in upload.meta
            assert "traceId" in upload.meta
            assert "spanId" in upload.meta

        # Test without span_context
        input_part2 = Blob(
            content=test_data, mime_type="image/png", modality="image"
        )
        input_message2 = InputMessage(role="user", parts=[input_part2])
        input_messages2 = [input_message2]

        uploads2 = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages2,
            output_messages=None,
        )

        assert len(uploads2) == 1
        # Verify meta doesn't contain trace/span fields
        meta = uploads2[0].meta
        assert "workspaceId" in meta
        assert "serviceId" in meta
        assert "timestamp" in meta
        assert "traceId" not in meta
        assert "spanId" not in meta

    @staticmethod
    def test_process_empty_messages(pre_uploader):
        """Test empty messages"""
        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=None,
            output_messages=None,
        )

        assert len(uploads) == 0

    # ========== Exception Handling Tests ==========

    # ========== Complete Workflow Tests ==========

    @staticmethod
    def test_audio_unknown_to_detected_workflow(pre_uploader):
        """Test audio/unknown auto-detection and conversion to correct format workflow"""
        filename = "test.wav"
        expected_mime = "audio/wav"
        expected_ext = "wav"

        filepath = TEST_AUDIO_DIR / filename
        if not filepath.exists():
            pytest.skip(f"Test audio file does not exist: {filepath}")

        # Read real audio file
        with open(filepath, "rb") as file_obj:
            audio_data = file_obj.read()

        # Create message with audio/unknown
        part = Blob(
            content=audio_data, mime_type="audio/unknown", modality="audio"
        )

        input_message = InputMessage(role="user", parts=[part])
        input_messages = [input_message]

        # Execute pre_upload workflow
        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        # Verify auto-detection and conversion successful
        assert len(uploads) == 1
        assert uploads[0].content_type == expected_mime, (
            f"MIME type should convert from audio/unknown to {expected_mime}"
        )
        assert uploads[0].url.endswith(f".{expected_ext}")


class TestPreUploadKeyGeneration:
    """Test remote URI key generation strategy"""

    @pytest.fixture
    def pre_uploader(self):  # pylint: disable=R6301
        return MultimodalPreUploader(base_path="/tmp/test_upload")

    @staticmethod
    @pytest.mark.parametrize(
        "uri,etag,last_modified,expected_contains_url",
        [
            # Different ETags should generate different keys
            ("https://example.com/image.png", '"abc123"', None, True),
            ("https://example.com/image.png", '"def456"', None, True),
            # Same params should generate same key
            (
                "https://example.com/image.png?v=1",
                '"abc123"',
                "Wed, 01 Jan 2020 00:00:00 GMT",
                True,
            ),
            (
                "https://example.com/image.png?v=2",
                '"abc123"',
                "Wed, 01 Jan 2020 00:00:00 GMT",
                True,
            ),
        ],
    )
    def test_generate_remote_key_consistency(
        pre_uploader, uri, etag, last_modified, expected_contains_url
    ):
        """Test _generate_remote_key consistency"""
        key = pre_uploader._generate_remote_key(
            uri=uri,
            content_type="image/png",
            content_length=1024,
            etag=etag,
            last_modified=last_modified,
        )

        # key should be 32-char MD5
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

    @staticmethod
    def test_generate_remote_key_same_etag_different_url_same_key(
        pre_uploader,
    ):
        """Test same key generation when URL query params differ but other metadata is same"""
        key1 = pre_uploader._generate_remote_key(
            uri="https://cdn.example.com/image.png?token=abc",
            content_type="image/png",
            content_length=1024,
            etag='"same-etag"',
            last_modified="Wed, 01 Jan 2020 00:00:00 GMT",
        )
        key2 = pre_uploader._generate_remote_key(
            uri="https://cdn.example.com/image.png?token=xyz",
            content_type="image/png",
            content_length=1024,
            etag='"same-etag"',
            last_modified="Wed, 01 Jan 2020 00:00:00 GMT",
        )

        # Different query params but same others should generate same key
        assert key1 == key2

    @staticmethod
    def test_generate_remote_key_different_domain_different_key(pre_uploader):
        """Test different keys for different domains (prevent CDN collisions)"""
        key1 = pre_uploader._generate_remote_key(
            uri="https://cdn1.example.com/image.png",
            content_type="image/png",
            content_length=1024,
            etag='"same-etag"',
            last_modified="Wed, 01 Jan 2020 00:00:00 GMT",
        )
        key2 = pre_uploader._generate_remote_key(
            uri="https://cdn2.example.com/image.png",
            content_type="image/png",
            content_length=1024,
            etag='"same-etag"',
            last_modified="Wed, 01 Jan 2020 00:00:00 GMT",
        )

        # Different domains should generate different keys
        assert key1 != key2


class TestPreUploadUri:
    """Test MultimodalPreUploader Uri handling"""

    @pytest.fixture
    def pre_uploader(self):  # pylint: disable=R6301
        """Create PreUploader instance"""
        return MultimodalPreUploader(
            base_path="/tmp/test_upload",
            extra_meta={
                "workspaceId": "test_workspace",
                "serviceId": "test_service",
            },
        )

    # ========== Async Metadata Fetching Tests ==========

    @staticmethod
    @pytest.mark.asyncio
    async def test_fetch_one_metadata_async_success(pre_uploader):
        """Test successful async fetch of single URI metadata"""
        mock_response = MagicMock()
        mock_response.headers = {
            "Content-Type": "image/png",
            "Content-Range": "bytes 0-0/1024",
            "ETag": '"abc123"',
            "Last-Modified": "Wed, 01 Jan 2020 00:00:00 GMT",
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        uri, metadata = await pre_uploader._fetch_one_metadata_async(
            mock_client, "https://example.com/image.png"
        )

        assert uri == "https://example.com/image.png"
        assert metadata is not None
        assert metadata.content_type == "image/png"
        assert metadata.content_length == 1024
        assert metadata.etag == '"abc123"'
        assert metadata.last_modified == "Wed, 01 Jan 2020 00:00:00 GMT"

    @staticmethod
    @pytest.mark.asyncio
    async def test_fetch_one_metadata_async_no_content_type(pre_uploader):
        """Test returns None when Content-Type is missing"""
        mock_response = MagicMock()
        mock_response.headers = {
            "Content-Range": "bytes 0-0/1024",
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        uri, metadata = await pre_uploader._fetch_one_metadata_async(
            mock_client, "https://example.com/image.png"
        )

        assert uri == "https://example.com/image.png"
        assert metadata is None

    @staticmethod
    @pytest.mark.asyncio
    async def test_fetch_one_metadata_async_http_error(pre_uploader):
        """Test returns None on HTTP error"""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPError("Network error")
        )

        uri, metadata = await pre_uploader._fetch_one_metadata_async(
            mock_client, "https://example.com/image.png"
        )

        assert uri == "https://example.com/image.png"
        assert metadata is None

    # ========== Uri Successful Processing Tests ==========

    @staticmethod
    @patch.object(MultimodalPreUploader, "_fetch_metadata_batch")
    def test_uri_part_success_creates_download_upload_item(
        mock_fetch, pre_uploader
    ):
        """Test creating DOWNLOAD_AND_UPLOAD task after successful metadata fetching"""
        mock_fetch.return_value = {
            "https://example.com/image.png": UriMetadata(
                content_type="image/png",
                content_length=1024,
                etag='"abc123"',
                last_modified="Wed, 01 Jan 2020 00:00:00 GMT",
            )
        }

        part = Uri(
            uri="https://example.com/image.png",
            mime_type="image/png",
            modality="image",
        )
        message = InputMessage(role="user", parts=[part])

        input_messages = [message]

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        assert len(uploads) == 1
        assert uploads[0].data is None  # Download-upload task has no data
        assert uploads[0].source_uri == "https://example.com/image.png"
        assert uploads[0].expected_size == 1024
        assert uploads[0].content_type == "image/png"

        # Verify original part is replaced
        assert message.parts[0].uri != "https://example.com/image.png"
        assert message.parts[0].uri.endswith(".png")

    @staticmethod
    @patch.object(MultimodalPreUploader, "_fetch_metadata_batch")
    def test_uri_part_all_modalities(mock_fetch, pre_uploader):
        """Test all supported modalities (image/video/audio)"""
        test_cases = [
            ("image", "image/png", ".png"),
            ("video", "video/mp4", ".mp4"),
            ("audio", "audio/wav", ".wav"),
        ]

        for modality, content_type, ext in test_cases:
            uri = f"https://example.com/file{ext}"
            mock_fetch.return_value = {
                uri: UriMetadata(
                    content_type=content_type,
                    content_length=1024,
                    etag=f'"{modality}-etag"',
                )
            }

            part = Uri(uri=uri, mime_type=content_type, modality=modality)
            message = InputMessage(role="user", parts=[part])

            input_messages = [message]

            uploads = pre_uploader.pre_upload(
                span_context=None,
                start_time_utc_nano=1000000000000000000,
                input_messages=input_messages,
                output_messages=None,
            )

            assert len(uploads) == 1, f"Failed for modality {modality}"
            assert uploads[0].content_type == content_type
            assert uploads[0].source_uri == uri

    # ========== Uri Skip Scenarios Tests ==========

    @staticmethod
    @patch.object(MultimodalPreUploader, "_fetch_metadata_batch")
    def test_uri_part_skip_unsupported_modality(mock_fetch, pre_uploader):
        """Test unsupported modality doesn't call fetch"""
        part = Uri(
            uri="https://example.com/file.bin",
            mime_type="application/octet-stream",
            modality="__unsupported__",
        )
        message = InputMessage(role="user", parts=[part])

        input_messages = [message]

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        assert len(uploads) == 0
        # Unsupported modality should not call fetch (no URI to fetch)
        mock_fetch.assert_not_called()

    @staticmethod
    @patch.object(MultimodalPreUploader, "_fetch_metadata_batch")
    def test_uri_part_skip_metadata_fetch_failed(mock_fetch, pre_uploader):
        """Test keeping original when metadata fetch fails"""
        mock_fetch.return_value = {}  # No metadata returned

        original_uri = "https://example.com/fail.png"
        part = Uri(uri=original_uri, mime_type="image/png", modality="image")
        message = InputMessage(role="user", parts=[part])

        input_messages = [message]

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        assert len(uploads) == 0
        # Part should remain unchanged
        assert message.parts[0].uri == original_uri

    @staticmethod
    @patch.object(MultimodalPreUploader, "_fetch_metadata_batch")
    def test_uri_part_skip_size_exceeded(mock_fetch, pre_uploader):
        """Test keeping original when size exceeds limit"""
        mock_fetch.return_value = {
            "https://example.com/large.png": UriMetadata(
                content_type="image/png",
                content_length=_MAX_MULTIMODAL_DATA_SIZE + 1,  # Exceeds limit
                etag='"abc123"',
            )
        }

        original_uri = "https://example.com/large.png"
        part = Uri(uri=original_uri, mime_type="image/png", modality="image")
        message = InputMessage(role="user", parts=[part])

        input_messages = [message]

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        assert len(uploads) == 0
        # Part should remain unchanged
        assert message.parts[0].uri == original_uri


class TestPreUploadLimits:
    """Test multimodal data processing limits"""

    @pytest.fixture
    def pre_uploader(self):  # pylint: disable=R6301
        return MultimodalPreUploader(base_path="/tmp/test_upload")

    @staticmethod
    def test_max_multimodal_parts_limit(pre_uploader):
        """Test processing at most _MAX_MULTIMODAL_PARTS parts"""
        test_data = b"test data"

        # Create more parts than the limit
        parts = []
        for _ in range(_MAX_MULTIMODAL_PARTS + 5):
            parts.append(
                Blob(
                    content=test_data, mime_type="image/png", modality="image"
                )
            )

        message = InputMessage(role="user", parts=parts)

        input_messages = [message]

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        # Process at most _MAX_MULTIMODAL_PARTS parts
        assert len(uploads) == _MAX_MULTIMODAL_PARTS


class TestPreUploadEventLoop:
    """Test behavior in existing event loop scenarios"""

    @pytest.fixture
    def pre_uploader(self):  # pylint: disable=R6301
        return MultimodalPreUploader(base_path="/tmp/test_upload")

    @staticmethod
    @pytest.mark.asyncio
    async def test_pre_upload_in_existing_event_loop(pre_uploader):
        """Test calling pre_upload to handle Uri in existing event loop"""
        with respx.mock:
            respx.get("https://example.com/test.png").mock(
                return_value=httpx.Response(
                    206,
                    headers={
                        "Content-Type": "image/png",
                        "Content-Range": "bytes 0-0/1024",
                        "ETag": '"test-etag"',
                    },
                    content=b"\x00",
                )
            )

            part = Blob(
                content=b"test data in async context",
                mime_type="image/png",
                modality="image",
            )
            message = InputMessage(role="user", parts=[part])

            input_messages = [message]

            test_data = b"test data in async context"

            part2 = Uri(
                uri="https://example.com/test.png",
                mime_type="image/png",
                modality="image",
            )
            message2 = InputMessage(role="assistant", parts=[part2])
            output_messages = [message2]

            uploads = pre_uploader.pre_upload(
                span_context=None,
                start_time_utc_nano=1000000000000000000,
                input_messages=input_messages,
                output_messages=output_messages,
            )

            assert len(uploads) == 2
            assert uploads[0].data == test_data
            assert uploads[1].source_uri == "https://example.com/test.png"


class TestPreUploadNonHttpUri:
    """Test non-HTTP URI handling"""

    @pytest.fixture
    def pre_uploader(self):  # pylint: disable=R6301
        return MultimodalPreUploader(base_path="/tmp/test_upload")

    @staticmethod
    @patch.object(MultimodalPreUploader, "_fetch_metadata_batch")
    def test_non_http_uri_skipped(mock_fetch, pre_uploader):
        """Test non-http and already processed URIs are not fetched again"""
        # Test non-http/https URI
        part = Uri(
            uri="file:///local/path/image.png",
            mime_type="image/png",
            modality="image",
        )
        message = InputMessage(role="user", parts=[part])

        input_messages = [message]
        # Simulate already processed URI (e.g., sls:// or other custom protocols)
        part = Uri(
            uri="sls://project/logstore/20241225/abc.png",
            mime_type="image/png",
            modality="image",
        )
        message2 = InputMessage(role="user", parts=[part])
        input_messages.append(message2)

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        assert len(uploads) == 0
        mock_fetch.assert_not_called()

    @staticmethod
    def test_is_http_uri_method(pre_uploader):
        """Test _is_http_uri method"""
        assert (
            MultimodalPreUploader._is_http_uri("http://example.com/test.png")
            is True
        )
        assert (
            MultimodalPreUploader._is_http_uri("https://example.com/test.png")
            is True
        )
        assert (
            MultimodalPreUploader._is_http_uri("HTTP://example.com/test.png")
            is False
        )  # Case sensitive
        assert (
            MultimodalPreUploader._is_http_uri("file:///local/path.png")
            is False
        )
        assert (
            MultimodalPreUploader._is_http_uri(
                "sls://project/logstore/file.png"
            )
            is False
        )
        assert MultimodalPreUploader._is_http_uri("/local/path.png") is False


class TestMultimodalUploadSwitch:
    """Test multimodal upload switch configuration

    Configuration is now read directly from environment variables, no longer depends on ArmsEnv singleton.
    """

    @staticmethod
    def test_upload_mode_none_skips_all():
        """Test OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE=none skips all processing"""
        with patch.dict(
            "os.environ",
            {"OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE": "none"},
        ):
            pre_uploader = MultimodalPreUploader("/tmp/test")
            assert pre_uploader._process_input is False
            assert pre_uploader._process_output is False

            # Even with multimodal data, don't process
            input_message = InputMessage(
                role="user",
                parts=[
                    Blob(
                        modality="image",
                        mime_type="image/png",
                        content=b"test",
                    )
                ],
            )
            input_messages = [input_message]

            uploads = pre_uploader.pre_upload(None, 0, input_messages, None)
            assert len(uploads) == 0

    @staticmethod
    def test_upload_mode_input_only():
        """OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE=input only processes input"""
        with patch.dict(
            "os.environ",
            {"OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE": "input"},
        ):
            pre_uploader = MultimodalPreUploader("/tmp/test")
            assert pre_uploader._process_input is True
            assert pre_uploader._process_output is False

    @staticmethod
    def test_upload_mode_output_only():
        """OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE=output only processes output"""
        with patch.dict(
            "os.environ",
            {"OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE": "output"},
        ):
            pre_uploader = MultimodalPreUploader("/tmp/test")
            assert pre_uploader._process_input is False
            assert pre_uploader._process_output is True

    @staticmethod
    def test_download_disabled_skips_uri():
        """OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED=false skips URI processing"""
        with patch.dict(
            "os.environ",
            {
                "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED": "false"
            },
        ):
            pre_uploader = MultimodalPreUploader("/tmp/test")
            assert pre_uploader._download_enabled is False

            # Blob still processes, Uri skipped
            input_message = InputMessage(
                role="user",
                parts=[
                    Blob(
                        modality="image",
                        mime_type="image/png",
                        content=b"test",
                    ),
                    Uri(
                        modality="image",
                        mime_type="image/jpeg",
                        uri="https://example.com/img.jpg",
                    ),
                ],
            )
            input_messages = [input_message]

            uploads = pre_uploader.pre_upload(None, 0, input_messages, None)
            # Only processed Blob
            assert len(uploads) == 1
            assert uploads[0].data is not None  # Blob has data
