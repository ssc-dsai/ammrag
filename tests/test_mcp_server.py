"""
Tests for MCP Server

These tests verify the MCP server tools work correctly.
Note: Requires the FastAPI service to be running on localhost:8000
"""

import pytest
import json
from mcp_server import (
    import_single_file,
    import_batch_files,
    retrieve_file,
    check_health,
    ImportSingleInput,
    ImportBatchInput,
    RetrieveFileInput,
    QueryFilesInput,
    ResponseFormat
)
from pydantic import HttpUrl


@pytest.mark.asyncio
class TestMCPTools:
    """Test MCP tool functionality"""
    
    async def test_import_single_markdown(self):
        """Test single file import with markdown response"""
        params = ImportSingleInput(
            uri=HttpUrl("https://example.com/test.pdf"),
            response_format=ResponseFormat.MARKDOWN
        )
        
        result = await import_single_file(params)
        
        assert "# File Import Complete" in result
        assert "UUID:" in result
        assert "https://example.com/test.pdf" in result
    
    async def test_import_single_json(self):
        """Test single file import with JSON response"""
        params = ImportSingleInput(
            uri=HttpUrl("https://example.com/test.pdf"),
            response_format=ResponseFormat.JSON
        )
        
        result = await import_single_file(params)
        data = json.loads(result)
        
        assert "uuid" in data
        assert data["uri"] == "https://example.com/test.pdf"
        assert "imported_at" in data
    
    async def test_import_batch_markdown(self):
        """Test batch import with markdown response"""
        params = ImportBatchInput(
            uris=[
                HttpUrl("https://example.com/doc1.pdf"),
                HttpUrl("https://example.com/doc2.pdf"),
                HttpUrl("https://example.com/doc3.pdf")
            ],
            response_format=ResponseFormat.MARKDOWN
        )
        
        result = await import_batch_files(params)
        
        assert "# Batch Import Complete" in result
        assert "Total Files Imported: 3" in result
        assert "## File 1" in result
        assert "## File 2" in result
        assert "## File 3" in result
    
    async def test_import_batch_json(self):
        """Test batch import with JSON response"""
        params = ImportBatchInput(
            uris=[
                HttpUrl("https://example.com/doc1.pdf"),
                HttpUrl("https://example.com/doc2.pdf")
            ],
            response_format=ResponseFormat.JSON
        )
        
        result = await import_batch_files(params)
        data = json.loads(result)
        
        assert data["total"] == 2
        assert len(data["files"]) == 2
        assert all("uuid" in f for f in data["files"])
    
    async def test_retrieve_file_markdown(self):
        """Test file retrieval with markdown response"""
        # First import a file
        import_params = ImportSingleInput(
            uri=HttpUrl("https://example.com/retrieve-test.pdf"),
            response_format=ResponseFormat.JSON
        )
        import_result = await import_single_file(import_params)
        import_data = json.loads(import_result)
        file_uuid = import_data["uuid"]
        
        # Now retrieve it
        retrieve_params = RetrieveFileInput(
            file_uuid=file_uuid,
            response_format=ResponseFormat.MARKDOWN
        )
        
        result = await retrieve_file(retrieve_params)
        
        assert "# File Information" in result
        assert file_uuid in result
        assert "https://example.com/retrieve-test.pdf" in result
        assert "Access Link:" in result
    
    async def test_retrieve_file_json(self):
        """Test file retrieval with JSON response"""
        # First import a file
        import_params = ImportSingleInput(
            uri=HttpUrl("https://example.com/retrieve-test-2.pdf"),
            response_format=ResponseFormat.JSON
        )
        import_result = await import_single_file(import_params)
        import_data = json.loads(import_result)
        file_uuid = import_data["uuid"]
        
        # Now retrieve it
        retrieve_params = RetrieveFileInput(
            file_uuid=file_uuid,
            response_format=ResponseFormat.JSON
        )
        
        result = await retrieve_file(retrieve_params)
        data = json.loads(result)
        
        assert data["uuid"] == file_uuid
        assert data["uri"] == "https://example.com/retrieve-test-2.pdf"
        assert "access_link" in data
    
    async def test_retrieve_nonexistent_file(self):
        """Test retrieving a file that doesn't exist"""
        params = RetrieveFileInput(
            file_uuid="00000000-0000-0000-0000-000000000000",
            response_format=ResponseFormat.MARKDOWN
        )
        
        result = await retrieve_file(params)
        
        assert "Error" in result
        assert "not found" in result.lower()
    
    async def test_health_check_markdown(self):
        """Test health check with markdown response"""
        params = QueryFilesInput(
            response_format=ResponseFormat.MARKDOWN
        )
        
        result = await check_health(params)
        
        assert "# Service Health Status" in result
        assert "Status:" in result
        assert "Total Files:" in result
    
    async def test_health_check_json(self):
        """Test health check with JSON response"""
        params = QueryFilesInput(
            response_format=ResponseFormat.JSON
        )
        
        result = await check_health(params)
        data = json.loads(result)
        
        assert "status" in data
        assert "total_files" in data
        assert data["status"] == "healthy"


class TestInputValidation:
    """Test Pydantic input validation"""
    
    def test_import_single_invalid_uri(self):
        """Test that invalid URI is rejected"""
        with pytest.raises(Exception):
            ImportSingleInput(
                uri="not-a-valid-uri",
                response_format=ResponseFormat.MARKDOWN
            )
    
    def test_import_batch_empty_list(self):
        """Test that empty URI list is rejected"""
        with pytest.raises(Exception):
            ImportBatchInput(
                uris=[],
                response_format=ResponseFormat.MARKDOWN
            )
    
    def test_import_batch_too_many(self):
        """Test that too many URIs are rejected"""
        with pytest.raises(Exception):
            uris = [HttpUrl(f"https://example.com/file{i}.pdf") for i in range(51)]
            ImportBatchInput(
                uris=uris,
                response_format=ResponseFormat.MARKDOWN
            )
    
    def test_retrieve_invalid_uuid_length(self):
        """Test that invalid UUID length is rejected"""
        with pytest.raises(Exception):
            RetrieveFileInput(
                file_uuid="too-short",
                response_format=ResponseFormat.MARKDOWN
            )


@pytest.mark.asyncio
class TestCompleteWorkflow:
    """Test complete MCP workflows"""
    
    async def test_import_and_retrieve_workflow(self):
        """Test importing a file and then retrieving it"""
        # Step 1: Import
        import_params = ImportSingleInput(
            uri=HttpUrl("https://example.com/workflow-test.pdf"),
            response_format=ResponseFormat.JSON
        )
        import_result = await import_single_file(import_params)
        import_data = json.loads(import_result)
        file_uuid = import_data["uuid"]
        
        # Step 2: Retrieve
        retrieve_params = RetrieveFileInput(
            file_uuid=file_uuid,
            response_format=ResponseFormat.JSON
        )
        retrieve_result = await retrieve_file(retrieve_params)
        retrieve_data = json.loads(retrieve_result)
        
        # Verify
        assert retrieve_data["uuid"] == file_uuid
        assert retrieve_data["uri"] == "https://example.com/workflow-test.pdf"
        assert "access_link" in retrieve_data
    
    async def test_batch_import_and_retrieve_all(self):
        """Test batch importing files and retrieving each one"""
        # Step 1: Batch import
        uris = [
            HttpUrl("https://example.com/batch1.pdf"),
            HttpUrl("https://example.com/batch2.pdf"),
            HttpUrl("https://example.com/batch3.pdf")
        ]
        import_params = ImportBatchInput(
            uris=uris,
            response_format=ResponseFormat.JSON
        )
        import_result = await import_batch_files(import_params)
        import_data = json.loads(import_result)
        
        # Step 2: Retrieve each file
        for file_info in import_data["files"]:
            retrieve_params = RetrieveFileInput(
                file_uuid=file_info["uuid"],
                response_format=ResponseFormat.JSON
            )
            retrieve_result = await retrieve_file(retrieve_params)
            retrieve_data = json.loads(retrieve_result)
            
            assert retrieve_data["uuid"] == file_info["uuid"]
            assert retrieve_data["uri"] == file_info["uri"]
