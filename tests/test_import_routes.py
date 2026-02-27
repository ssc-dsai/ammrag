"""
Tests for import routes
"""

import pytest
from fastapi.testclient import TestClient

from main import app
from app.services.storage_service import file_storage

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_storage():
    """Clear file storage before each test"""
    file_storage.clear()
    yield
    file_storage.clear()


class TestImportSingle:
    """Tests for single file import endpoint"""
    
    def test_import_single_success(self):
        """Test successful single file import"""
        uri = "https://example.com/document.pdf"
        response = client.get(f"/import/single?uri={uri}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "uuid" in data
        assert data["uri"] == uri
        assert "imported_at" in data
    
    def test_import_single_invalid_uri(self):
        """Test import with invalid URI"""
        response = client.get("/import/single?uri=not-a-valid-uri")
        assert response.status_code == 422
    
    def test_import_single_missing_uri(self):
        """Test import without URI parameter"""
        response = client.get("/import/single")
        assert response.status_code == 422
    
    def test_import_single_multiple_files(self):
        """Test importing multiple files sequentially"""
        uris = [
            "https://example.com/doc1.pdf",
            "https://example.com/doc2.pdf",
            "https://example.com/doc3.pdf"
        ]
        
        uuids = []
        for uri in uris:
            response = client.get(f"/import/single?uri={uri}")
            assert response.status_code == 200
            uuids.append(response.json()["uuid"])
        
        # Verify all UUIDs are unique
        assert len(uuids) == len(set(uuids))


class TestImportBatch:
    """Tests for batch file import endpoint"""
    
    def test_import_batch_success(self):
        """Test successful batch import"""
        uris = [
            "https://example.com/doc1.pdf",
            "https://example.com/doc2.pdf",
            "https://example.com/doc3.pdf"
        ]
        
        query_string = "&".join([f"uri={uri}" for uri in uris])
        response = client.get(f"/import/batch?{query_string}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total"] == 3
        assert len(data["files"]) == 3
        
        for i, file_data in enumerate(data["files"]):
            assert "uuid" in file_data
            assert file_data["uri"] == uris[i]
            assert "imported_at" in file_data
    
    def test_import_batch_single_file(self):
        """Test batch import with single file"""
        uri = "https://example.com/single.pdf"
        response = client.get(f"/import/batch?uri={uri}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["files"]) == 1
    
    def test_import_batch_no_uris(self):
        """Test batch import with no URIs"""
        response = client.get("/import/batch")
        assert response.status_code == 422
    
    def test_import_batch_invalid_uri(self):
        """Test batch import with invalid URI"""
        response = client.get("/import/batch?uri=not-valid&uri=https://example.com/valid.pdf")
        assert response.status_code == 422
    
    def test_import_batch_unique_uuids(self):
        """Test that batch import generates unique UUIDs"""
        uris = [f"https://example.com/doc{i}.pdf" for i in range(10)]
        query_string = "&".join([f"uri={uri}" for uri in uris])
        response = client.get(f"/import/batch?{query_string}")
        
        assert response.status_code == 200
        data = response.json()
        
        uuids = [f["uuid"] for f in data["files"]]
        assert len(uuids) == len(set(uuids))
