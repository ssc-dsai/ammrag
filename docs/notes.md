# Notes

### Process
1. Import files
2. Determine type of files
3. Determine embedding strategy
- embedder
- chunking
    - size (based on MAX_TOKEN size for embedder)
    - method (semantic, sentence)
- distance
- filters
- responses returned count

### Import
- Get path of file (uri)
- add path to database
- Process file with MMORE to flat files
- Additional processing for certain types
    - jpg
        - Send to ollama, get description
    - xlsx
        - extract subtables, import into relational DB
    - csv
        - import into relational DB

Import object:
    - uuid
    - uri
    - DB file ID
    - chunk array
    - filters

### Retrieve
Parse question to have specific info:
    - Is there a location?
    - Are there any other keywords/filters?

Query Qdrant
    Return:
        - Anything 

