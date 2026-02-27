"""
Tests for retrieve routes
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


class TestRetrieveFile:
    """Tests for file retrieval endpoint"""
    
    def test_retrieve_success(self):
        """Test successful file retrieval"""
        # First import a file
        uri = "https://example.com/test.pdf"
        import_response = client.get(f"/import/single?uri={uri}")
        file_uuid = import_response.json()["uuid"]
        
        # Now retrieve it
        retrieve_response = client.get(f"/retrieve/{file_uuid}")
        
        assert retrieve_response.status_code == 200
        data = retrieve_response.json()
        
        assert data["uuid"] == file_uuid
        assert data["uri"] == uri
        assert "imported_at" in data
        assert "access_link" in data
        assert file_uuid in data["access_link"]
    
    def test_retrieve_nonexistent_file(self):
        """Test retrieving a file that doesn't exist"""
        fake_uuid = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/retrieve/{fake_uuid}")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_retrieve_invalid_uuid_format(self):
        """Test retrieving with invalid UUID format"""
        # This should still work as we don't validate UUID format strictly
        response = client.get("/retrieve/not-a-uuid")
        assert response.status_code == 404
    
    def test_retrieve_multiple_files(self):
        """Test retrieving multiple different files"""
        uris = [
            "https://example.com/doc1.pdf",
            "https://example.com/doc2.pdf",
            "https://example.com/doc3.pdf"
        ]
        
        # Import files
        uuids = []
        for uri in uris:
            response = client.get(f"/import/single?uri={uri}")
            uuids.append(response.json()["uuid"])
        
        # Retrieve each file
        for i, uuid in enumerate(uuids):
            response = client.get(f"/retrieve/{uuid}")
            assert response.status_code == 200
            data = response.json()
            assert data["uuid"] == uuid
            assert data["uri"] == uris[i]


class TestDownloadFile:
    """Tests for file download endpoint"""
    
    def test_download_success(self):
        """Test successful file download"""
        # First import a file
        uri = "https://example.com/test.pdf"
        import_response = client.get(f"/import/single?uri={uri}")
        file_uuid = import_response.json()["uuid"]
        
        # Try to download
        download_response = client.get(f"/files/{file_uuid}/download")
        
        assert download_response.status_code == 200
        data = download_response.json()
        assert data["uuid"] == file_uuid
        assert data["original_uri"] == uri
    
    def test_download_nonexistent_file(self):
        """Test downloading a file that doesn't exist"""
        fake_uuid = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/files/{fake_uuid}/download")
        
        assert response.status_code == 404
    
    def test_download_via_access_link(self):
        """Test downloading file using access link from retrieve endpoint"""
        # Import file
        uri = "https://example.com/test.pdf"
        import_response = client.get(f"/import/single?uri={uri}")
        file_uuid = import_response.json()["uuid"]
        
        # Retrieve file to get access link
        retrieve_response = client.get(f"/retrieve/{file_uuid}")
        access_link = retrieve_response.json()["access_link"]
        
        # Download using access link
        download_response = client.get(access_link)
        assert download_response.status_code == 200


class TestCompleteWorkflow:
    """Tests for complete import -> retrieve -> download workflow"""
    
    def test_single_file_workflow(self):
        """Test complete workflow for single file"""
        # 1. Import
        uri = "https://example.com/workflow-test.pdf"
        import_response = client.get(f"/import/single?uri={uri}")
        assert import_response.status_code == 200
        file_uuid = import_response.json()["uuid"]
        
        # 2. Retrieve
        retrieve_response = client.get(f"/retrieve/{file_uuid}")
        assert retrieve_response.status_code == 200
        access_link = retrieve_response.json()["access_link"]
        
        # 3. Download
        download_response = client.get(access_link)
        assert download_response.status_code == 200
    
    def test_batch_workflow(self):
        """Test complete workflow for batch import"""
        # 1. Batch import
        uris = [
            "https://example.com/batch1.pdf",
            "https://example.com/batch2.pdf",
            "https://example.com/batch3.pdf"
        ]
        query_string = "&".join([f"uri={uri}" for uri in uris])
        import_response = client.get(f"/import/batch?{query_string}")
        assert import_response.status_code == 200
        
        files = import_response.json()["files"]
        
        # 2. Retrieve each file
        for file_data in files:
            retrieve_response = client.get(f"/retrieve/{file_data['uuid']}")
            assert retrieve_response.status_code == 200
            
            # 3. Download each file
            access_link = retrieve_response.json()["access_link"]
            download_response = client.get(access_link)
            assert download_response.status_code == 200
