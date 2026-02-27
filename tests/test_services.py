"""
Tests for service layer
"""

import pytest

from app.services.file_service import FileService
from app.services.storage_service import FileStorage
from app.models.file_model import FileMetadata
from fastapi import HTTPException


@pytest.fixture
def storage():
    """Create a fresh storage instance for each test"""
    return FileStorage()


@pytest.fixture
def service(storage):
    """Create a file service with fresh storage"""
    return FileService(storage=storage)


class TestFileService:
    """Tests for FileService"""
    
    def test_import_single_file(self, service):
        """Test importing a single file"""
        uri = "https://example.com/test.pdf"
        result = service.import_single_file(uri)
        
        assert result.uuid is not None
        assert result.uri == uri
        assert result.imported_at is not None
    
    def test_import_batch_files(self, service):
        """Test importing multiple files"""
        uris = [
            "https://example.com/doc1.pdf",
            "https://example.com/doc2.pdf",
            "https://example.com/doc3.pdf"
        ]
        
        result = service.import_batch_files(uris)
        
        assert result.total == 3
        assert len(result.files) == 3
        
        for i, file_response in enumerate(result.files):
            assert file_response.uri == uris[i]
    
    def test_import_batch_empty_list(self, service):
        """Test batch import with empty list raises error"""
        with pytest.raises(HTTPException) as exc:
            service.import_batch_files([])
        
        assert exc.value.status_code == 400
    
    def test_retrieve_file(self, service):
        """Test retrieving an existing file"""
        # Import a file first
        uri = "https://example.com/test.pdf"
        import_result = service.import_single_file(uri)
        
        # Retrieve it
        retrieve_result = service.retrieve_file(import_result.uuid)
        
        assert retrieve_result.uuid == import_result.uuid
        assert retrieve_result.uri == uri
        assert retrieve_result.access_link is not None
    
    def test_retrieve_nonexistent_file(self, service):
        """Test retrieving a file that doesn't exist"""
        with pytest.raises(HTTPException) as exc:
            service.retrieve_file("nonexistent-uuid")
        
        assert exc.value.status_code == 404
    
    def test_get_file_metadata(self, service):
        """Test getting raw file metadata"""
        uri = "https://example.com/test.pdf"
        import_result = service.import_single_file(uri)
        
        metadata = service.get_file_metadata(import_result.uuid)
        
        assert isinstance(metadata, FileMetadata)
        assert metadata.uuid == import_result.uuid
        assert metadata.uri == uri
    
    def test_get_total_files(self, service):
        """Test getting total file count"""
        assert service.get_total_files() == 0
        
        service.import_single_file("https://example.com/file1.pdf")
        assert service.get_total_files() == 1
        
        service.import_single_file("https://example.com/file2.pdf")
        assert service.get_total_files() == 2


class TestFileStorage:
    """Tests for FileStorage"""
    
    def test_create_file(self, storage):
        """Test creating a file record"""
        uri = "https://example.com/test.pdf"
        metadata = storage.create(uri)
        
        assert metadata.uuid is not None
        assert metadata.uri == uri
        assert metadata.imported_at is not None
    
    def test_get_file(self, storage):
        """Test getting a file record"""
        uri = "https://example.com/test.pdf"
        created = storage.create(uri)
        
        retrieved = storage.get(created.uuid)
        
        assert retrieved is not None
        assert retrieved.uuid == created.uuid
        assert retrieved.uri == uri
    
    def test_get_nonexistent_file(self, storage):
        """Test getting a file that doesn't exist"""
        result = storage.get("nonexistent-uuid")
        assert result is None
    
    def test_exists(self, storage):
        """Test checking if file exists"""
        uri = "https://example.com/test.pdf"
        metadata = storage.create(uri)
        
        assert storage.exists(metadata.uuid) is True
        assert storage.exists("nonexistent-uuid") is False
    
    def test_list_all(self, storage):
        """Test listing all files"""
        assert len(storage.list_all()) == 0
        
        storage.create("https://example.com/file1.pdf")
        storage.create("https://example.com/file2.pdf")
        
        all_files = storage.list_all()
        assert len(all_files) == 2
    
    def test_count(self, storage):
        """Test counting files"""
        assert storage.count() == 0
        
        storage.create("https://example.com/file1.pdf")
        assert storage.count() == 1
        
        storage.create("https://example.com/file2.pdf")
        assert storage.count() == 2
    
    def test_delete(self, storage):
        """Test deleting a file"""
        uri = "https://example.com/test.pdf"
        metadata = storage.create(uri)
        
        assert storage.exists(metadata.uuid) is True
        
        result = storage.delete(metadata.uuid)
        assert result is True
        assert storage.exists(metadata.uuid) is False
    
    def test_delete_nonexistent(self, storage):
        """Test deleting a file that doesn't exist"""
        result = storage.delete("nonexistent-uuid")
        assert result is False
    
    def test_clear(self, storage):
        """Test clearing all storage"""
        storage.create("https://example.com/file1.pdf")
        storage.create("https://example.com/file2.pdf")
        
        assert storage.count() == 2
        
        storage.clear()
        assert storage.count() == 0
