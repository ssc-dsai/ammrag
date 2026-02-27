"""
Example tests for the FastAPI File Import Service

To run these tests:
1. Install pytest: pip install pytest httpx
2. Run: pytest test_main.py
"""

import pytest
from fastapi.testclient import TestClient
from main import app, file_storage

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_storage():
    """Clear file storage before each test"""
    file_storage.clear()
    yield
    file_storage.clear()


def test_root():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "version" in response.json()


def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_import_single():
    """Test importing a single file"""
    uri = "https://example.com/document.pdf"
    response = client.get(f"/import/single?uri={uri}")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "uuid" in data
    assert data["uri"] == uri
    assert "imported_at" in data


def test_import_single_invalid_uri():
    """Test importing with an invalid URI"""
    response = client.get("/import/single?uri=not-a-valid-uri")
    assert response.status_code == 422  # Validation error


def test_import_batch():
    """Test importing multiple files"""
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


def test_import_batch_no_uris():
    """Test batch import with no URIs provided"""
    response = client.get("/import/batch")
    assert response.status_code == 422  # Validation error


def test_retrieve_file():
    """Test retrieving a file by UUID"""
    # First, import a file
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


def test_retrieve_nonexistent_file():
    """Test retrieving a file that doesn't exist"""
    fake_uuid = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"/retrieve/{fake_uuid}")
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_download_endpoint():
    """Test the download endpoint"""
    # First, import a file
    uri = "https://example.com/test.pdf"
    import_response = client.get(f"/import/single?uri={uri}")
    file_uuid = import_response.json()["uuid"]
    
    # Try to download
    download_response = client.get(f"/files/{file_uuid}/download")
    
    assert download_response.status_code == 200
    data = download_response.json()
    assert data["uuid"] == file_uuid
    assert data["original_uri"] == uri


def test_download_nonexistent_file():
    """Test downloading a file that doesn't exist"""
    fake_uuid = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"/files/{fake_uuid}/download")
    
    assert response.status_code == 404


def test_complete_workflow():
    """Test the complete workflow: import -> retrieve -> download"""
    # Import a file
    uri = "https://example.com/workflow-test.pdf"
    import_response = client.get(f"/import/single?uri={uri}")
    assert import_response.status_code == 200
    file_uuid = import_response.json()["uuid"]
    
    # Retrieve file info
    retrieve_response = client.get(f"/retrieve/{file_uuid}")
    assert retrieve_response.status_code == 200
    access_link = retrieve_response.json()["access_link"]
    
    # Download the file using the access link
    download_response = client.get(access_link)
    assert download_response.status_code == 200
    
    # Verify storage
    health_response = client.get("/health")
    assert health_response.json()["total_files"] == 1
