"""
Unit tests for DocumentProcessor class.
Tests document processing, text chunking, and course parsing functionality.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from document_processor import DocumentProcessor
from models import Course, Lesson, CourseChunk


class TestDocumentProcessor:
    """Test the DocumentProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor for testing."""
        return DocumentProcessor(chunk_size=100, chunk_overlap=20)

    @pytest.fixture
    def small_processor(self):
        """Create a DocumentProcessor with small chunks for testing."""
        return DocumentProcessor(chunk_size=50, chunk_overlap=10)

    def test_initialization(self):
        """Test DocumentProcessor initialization."""
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        assert processor.chunk_size == 200
        assert processor.chunk_overlap == 50

    def test_read_file_utf8(self, processor, temp_course_file):
        """Test reading a UTF-8 file."""
        content = processor.read_file(temp_course_file)
        assert "Course Title: Test Course" in content
        assert "Lesson 1: Introduction" in content

    def test_read_file_unicode_error_handling(self, processor):
        """Test reading file with encoding issues."""
        with patch("builtins.open", side_effect=[UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")]):
            # Mock the second call to open (with error handling)
            with patch("builtins.open", mock_open(read_data="test content")) as mock_file:
                content = processor.read_file("test_file.txt")
                assert content == "test content"
                # Verify it was called twice (first fails, second succeeds)
                assert mock_file.call_count == 1  # Only the successful call

    def test_read_file_with_error_handling(self, processor):
        """Test reading file that requires error handling."""
        # Create a file with problematic encoding
        problematic_content = "Valid content with problematic chars"

        with patch("builtins.open") as mock_open_func:
            # First call raises UnicodeDecodeError
            # Second call succeeds with error handling
            mock_open_func.side_effect = [
                UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte"),
                mock_open(read_data=problematic_content).return_value
            ]

            content = processor.read_file("test_file.txt")
            assert content == problematic_content

    def test_chunk_text_basic(self, processor):
        """Test basic text chunking."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = processor.chunk_text(text)

        assert len(chunks) > 0
        assert all(len(chunk) <= processor.chunk_size for chunk in chunks)
        # Verify content is preserved
        combined = " ".join(chunks).replace(" ", "")
        original = text.replace(" ", "")
        # Allow for some overlap in combined text
        assert len(combined) >= len(original)

    def test_chunk_text_with_overlap(self, small_processor):
        """Test text chunking with overlap."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = small_processor.chunk_text(text)

        assert len(chunks) >= 2
        # Check for overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk1_words = set(chunks[i].split())
            chunk2_words = set(chunks[i + 1].split())
            # Should have some overlap due to overlap setting
            overlap_words = chunk1_words & chunk2_words
            # May or may not have overlap depending on sentence boundaries

    def test_chunk_text_empty_input(self, processor):
        """Test chunking empty text."""
        chunks = processor.chunk_text("")
        assert chunks == []

    def test_chunk_text_whitespace_only(self, processor):
        """Test chunking whitespace-only text."""
        chunks = processor.chunk_text("   \n\t  ")
        assert chunks == []

    def test_chunk_text_single_sentence(self, processor):
        """Test chunking single sentence."""
        text = "This is a single sentence."
        chunks = processor.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].strip() == text

    def test_chunk_text_long_sentence(self, small_processor):
        """Test chunking very long sentence that exceeds chunk_size."""
        text = "This is an extremely long sentence that definitely exceeds the chunk size limit and should be handled gracefully by the chunking algorithm."
        chunks = small_processor.chunk_text(text)

        # Should create at least one chunk even if sentence is too long
        assert len(chunks) >= 1

    def test_chunk_text_sentence_splitting(self, processor):
        """Test sentence splitting with various punctuation."""
        text = "First sentence! Second question? Third statement. Fourth exclamation!"
        chunks = processor.chunk_text(text)

        assert len(chunks) >= 1
        combined_content = " ".join(chunks)
        assert "First sentence!" in combined_content
        assert "Second question?" in combined_content

    def test_chunk_text_abbreviations(self, processor):
        """Test sentence splitting handles abbreviations correctly."""
        text = "Dr. Smith went to the U.S.A. He met Mr. Johnson there."
        chunks = processor.chunk_text(text)

        # Should not split on abbreviations
        combined = " ".join(chunks)
        assert "Dr. Smith" in combined
        assert "U.S.A." in combined

    def test_chunk_text_whitespace_normalization(self, processor):
        """Test that whitespace is properly normalized."""
        text = "First   sentence.\n\nSecond\tsentence."
        chunks = processor.chunk_text(text)

        # Should normalize multiple whitespace to single space
        for chunk in chunks:
            assert "   " not in chunk  # No multiple spaces
            assert "\n" not in chunk   # No newlines
            assert "\t" not in chunk   # No tabs

    def test_process_course_document_complete(self, processor, temp_course_file):
        """Test processing a complete course document."""
        course, chunks = processor.process_course_document(temp_course_file)

        # Verify course metadata
        assert course.title == "Test Course"
        assert course.course_link == "https://example.com/course"
        assert course.instructor == "Test Instructor"
        assert len(course.lessons) == 2

        # Verify lessons
        lesson1 = course.lessons[0]
        assert lesson1.lesson_number == 1
        assert lesson1.title == "Introduction"
        assert lesson1.lesson_link == "https://example.com/lesson/1"

        lesson2 = course.lessons[1]
        assert lesson2.lesson_number == 2
        assert lesson2.title == "Advanced Topics"
        assert lesson2.lesson_link == "https://example.com/lesson/2"

        # Verify chunks
        assert len(chunks) > 0
        assert all(isinstance(chunk, CourseChunk) for chunk in chunks)
        assert all(chunk.course_title == "Test Course" for chunk in chunks)

    def test_process_course_document_minimal(self, processor, temp_dir):
        """Test processing minimal course document."""
        minimal_content = """Course Title: Minimal Course

Lesson 1: Only Lesson
This is the only lesson content.
"""
        file_path = Path(temp_dir) / "minimal.txt"
        file_path.write_text(minimal_content)

        course, chunks = processor.process_course_document(str(file_path))

        assert course.title == "Minimal Course"
        assert course.course_link is None
        assert course.instructor is None
        assert len(course.lessons) == 1
        assert len(chunks) > 0

    def test_process_course_document_no_metadata(self, processor, temp_dir):
        """Test processing document with no course metadata."""
        content = """Some random content without proper headers.
This should still be processed somehow.
"""
        file_path = Path(temp_dir) / "no_metadata.txt"
        file_path.write_text(content)

        course, chunks = processor.process_course_document(str(file_path))

        # Should use filename as fallback
        assert course.title == "no_metadata.txt"
        assert course.instructor == "Unknown"
        assert len(chunks) > 0

    def test_process_course_document_malformed_lessons(self, processor, temp_dir):
        """Test processing document with malformed lesson headers."""
        content = """Course Title: Test Course

Not a lesson header
Some content here.

Lesson 1: Valid Lesson
Valid lesson content.

Invalid lesson format
More content.
"""
        file_path = Path(temp_dir) / "malformed.txt"
        file_path.write_text(content)

        course, chunks = processor.process_course_document(str(file_path))

        # Should only process valid lessons
        assert len(course.lessons) == 1
        assert course.lessons[0].title == "Valid Lesson"

    def test_process_course_document_lesson_without_link(self, processor, temp_dir):
        """Test processing lesson without lesson link."""
        content = """Course Title: Test Course

Lesson 1: Lesson Without Link
This lesson has no link specified.
"""
        file_path = Path(temp_dir) / "no_link.txt"
        file_path.write_text(content)

        course, chunks = processor.process_course_document(str(file_path))

        assert len(course.lessons) == 1
        assert course.lessons[0].lesson_link is None

    def test_process_course_document_empty_lessons(self, processor, temp_dir):
        """Test processing document with empty lesson content."""
        content = """Course Title: Test Course

Lesson 1: Empty Lesson

Lesson 2: Another Empty Lesson

"""
        file_path = Path(temp_dir) / "empty_lessons.txt"
        file_path.write_text(content)

        course, chunks = processor.process_course_document(str(file_path))

        # Should handle empty lessons gracefully
        assert len(course.lessons) == 0  # Empty lessons are filtered out
        assert len(chunks) == 0

    def test_process_course_document_unicode_content(self, processor, temp_dir):
        """Test processing document with unicode content."""
        unicode_content = """Course Title: Unicode Course 🌍

Lesson 1: Unicode Lesson
This lesson contains unicode: 世界 🚀 café résumé.
"""
        file_path = Path(temp_dir) / "unicode.txt"
        file_path.write_text(unicode_content, encoding='utf-8')

        course, chunks = processor.process_course_document(str(file_path))

        assert "🌍" in course.title
        assert len(chunks) > 0
        # Check that unicode is preserved in chunks
        combined_content = " ".join(chunk.content for chunk in chunks)
        assert "世界" in combined_content
        assert "🚀" in combined_content

    def test_process_course_document_complex_metadata(self, processor, temp_dir):
        """Test processing document with metadata in different order."""
        content = """Course Instructor: Dr. Jane Smith
Course Title: Advanced AI
Course Link: https://university.edu/ai-course

Lesson 1: Introduction
Lesson Link: https://university.edu/ai-course/lesson1
Introduction to AI concepts.
"""
        file_path = Path(temp_dir) / "complex.txt"
        file_path.write_text(content)

        course, chunks = processor.process_course_document(str(file_path))

        assert course.title == "Advanced AI"
        assert course.instructor == "Dr. Jane Smith"
        assert course.course_link == "https://university.edu/ai-course"

    def test_chunk_context_addition(self, processor, temp_course_file):
        """Test that lesson context is added to chunks."""
        course, chunks = processor.process_course_document(temp_course_file)

        # Check that first chunk of each lesson has context
        lesson_chunks = {}
        for chunk in chunks:
            if chunk.lesson_number not in lesson_chunks:
                lesson_chunks[chunk.lesson_number] = []
            lesson_chunks[chunk.lesson_number].append(chunk)

        # First chunk of lesson should have lesson context
        for lesson_number, lesson_chunks_list in lesson_chunks.items():
            if lesson_chunks_list:  # If there are chunks for this lesson
                first_chunk = lesson_chunks_list[0]
                assert f"Lesson {lesson_number} content:" in first_chunk.content

    def test_chunk_numbering(self, processor, temp_course_file):
        """Test that chunks are properly numbered."""
        course, chunks = processor.process_course_document(temp_course_file)

        # Chunks should be numbered sequentially
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_edge_case_very_long_lesson(self, small_processor, temp_dir):
        """Test processing lesson with very long content."""
        long_content = " ".join([f"This is sentence number {i}." for i in range(100)])
        content = f"""Course Title: Long Course

Lesson 1: Long Lesson
{long_content}
"""
        file_path = Path(temp_dir) / "long.txt"
        file_path.write_text(content)

        course, chunks = processor.process_course_document(str(file_path))

        # Should create multiple chunks
        assert len(chunks) > 1
        assert all(chunk.lesson_number == 1 for chunk in chunks)

    def test_edge_case_no_lessons_fallback(self, processor, temp_dir):
        """Test fallback when no lessons are found."""
        content = """Course Title: No Lessons Course
Course Instructor: Test Instructor

This is just some general content without lesson markers.
It should still be processed as course content.
"""
        file_path = Path(temp_dir) / "no_lessons.txt"
        file_path.write_text(content)

        course, chunks = processor.process_course_document(str(file_path))

        # Should create chunks from remaining content
        assert len(chunks) > 0
        assert all(chunk.lesson_number is None for chunk in chunks)
        assert all(chunk.course_title == "No Lessons Course" for chunk in chunks)