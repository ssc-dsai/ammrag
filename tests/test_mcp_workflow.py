"""
Quick test of the MCP server workflow
"""
import asyncio
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


async def main():
    print("=== MCP Server Workflow Test ===\n")

    # Test 1: Health Check
    print("1. Health Check")
    print("-" * 50)
    health_result = await check_health(
        QueryFilesInput(response_format=ResponseFormat.MARKDOWN)
    )
    print(health_result)
    print("\n")

    # Test 2: Import Single File (JSON format to get UUID)
    print("2. Import Single File")
    print("-" * 50)
    import_result_json = await import_single_file(
        ImportSingleInput(
            uri=HttpUrl("https://example.com/test-document.pdf"),
            response_format=ResponseFormat.JSON
        )
    )
    import_data = json.loads(import_result_json)
    file_uuid = import_data["uuid"]
    print(f"Imported file with UUID: {file_uuid}")
    print(f"Full response: {import_result_json}")
    print("\n")

    # Test 3: Import Single File (Markdown format)
    print("3. Import Another File (Markdown Format)")
    print("-" * 50)
    import_result_md = await import_single_file(
        ImportSingleInput(
            uri=HttpUrl("https://example.com/another-document.pdf"),
            response_format=ResponseFormat.MARKDOWN
        )
    )
    print(import_result_md)
    print("\n")

    # Test 4: Batch Import
    print("4. Batch Import (3 files)")
    print("-" * 50)
    batch_result = await import_batch_files(
        ImportBatchInput(
            uris=[
                HttpUrl("https://example.com/batch1.pdf"),
                HttpUrl("https://example.com/batch2.pdf"),
                HttpUrl("https://example.com/batch3.pdf")
            ],
            response_format=ResponseFormat.MARKDOWN
        )
    )
    print(batch_result)
    print("\n")

    # Test 5: Retrieve File
    print("5. Retrieve File by UUID")
    print("-" * 50)
    retrieve_result = await retrieve_file(
        RetrieveFileInput(
            file_uuid=file_uuid,
            response_format=ResponseFormat.MARKDOWN
        )
    )
    print(retrieve_result)
    print("\n")

    # Test 6: Final Health Check
    print("6. Final Health Check (should show 5 files)")
    print("-" * 50)
    final_health = await check_health(
        QueryFilesInput(response_format=ResponseFormat.MARKDOWN)
    )
    print(final_health)
    print("\n")

    print("=== All Tests Completed Successfully! ===")


if __name__ == "__main__":
    asyncio.run(main())
