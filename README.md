# RESTful RAG Utilities

In this repo, I've implemented a set of bespoke tools needed when working with advanced RAG techniques, such as chunk re-ranking, structured doc conversion to Markdown, and text classification.  


You can build the image yourself (not on-premise, yet scan it before internal use), or use my pre-built image directly.

```bash
docker run --rm -p 8181:8181 falriachi/rag-utils:amd64
```

Swagger API docs will be then served `http://localhost:8181/docs` 


The image exposes a RESTful api with 3 main endpoints (in addition to 2 GET endpoints /healthz & /models)

- POST /rerank
- POST /convert
- POST /classify


## POST /rerank for Pairwise & Listwise Reranking

It provides the ability to rerank a list of chunks (passages) using either Cross-encoder (Pairwise) or LLM (Listwise) based modesl (latter being much heavier when run on commodity machine)

**Endpoint**: `/rerank`  

Takes in a 'query', 'model' and a list of 'passages' that need to be ranked vs the 'query'. optionally 'max_length' to customize max token length allowed.

**model** options:

* `pairwise_nano` → ms-marco-TinyBERT-L-2-v2 (fast model & competitive performance (ranking precision))
* `pairwise_small` → ms-marco-MiniLM-L-12-v2 (slower & best performance (ranking precision))
* `pairwise_medium` → ms-marco-MultiBERT-L-12 (for 100+ languages. don't use for english)
* `pairwise_large` → rank-T5-flan (best zero-shot performance (ranking precision))
* `listwise_medium` → [rank_zephyr_7b_v1_full](https://huggingface.co/castorini/rank_zephyr_7b_v1_full) (4-bit-quantized GGUF based on chat models / LLM) 


**max_length** used in requests:
- For pairwise: max is 512 tokens (includes passage + Query)
- For Listwise: max 8192 tokens (includes all passages + Query)

### Usage Examples

#### Health check:

```bash
curl http://localhost:8181/healthz
```

#### List model aliases:

```bash
curl http://localhost:8181/models | python -m json.tool
```

#### Re-rank example:

```bash
curl -sS http://localhost:8181/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What is the capital of France?",
    "max_length": 128,
    "model": "nano",
    "passages": [
      {
        "id": 1,
        "text": "France is a country in Western Europe known for its rich history, cuisine, and cultural influence. The country has several major cities including Marseille, Lyon, and Paris."
      },
      {
        "id": 2,
        "text": "Paris is the capital and most populous city of France. It is the political, economic, and cultural center of the country."
      },
      {
        "id": 3,
        "text": "Lyon is a major city in France and is sometimes considered the country's gastronomic capital due to its culinary traditions and central location."
      }
    ]
  }' | python -m json.tool
  ```
Response:
```json
{
  "model": "ms-marco-TinyBERT-L-2-v2",
  "query": "What is the capital of France?",
  "max_length": 128,
  "results": [
    {
      "id": 2,
      "text": "Paris is the capital and most populous city of France. It is the political, economic, and cultural center of the country.",
      "meta": null,
      "score": 0.9989894032478333
    },
    {
      "id": 3,
      "text": "Lyon is the gastronomic capital and a major city in France known for its culinary traditions and central location.",
      "meta": null,
      "score": 0.9981333613395691
    },
    {
      "id": 1,
      "text": "France is a country in Western Europe known for its rich history, cuisine, and cultural influence. The country has several major cities including Marseille, Lyon, and Paris.",
      "meta": null,
      "score": 0.33484169840812683
    }
  ]
}
```

Notes:

- `meta`: can be optionally passed to include additional info during the re-ranking. Simply pass in it a dictionary of key:value pairs
- `max_length` is defaulted to 128. This value should be large enough to accommodate your longest passage. For example, if your longest passage (100 tokens) + query (16 tokens) = 116 then setting max_length = 128 is good enough including room for reserved tokens like [CLS] and [SEP]. Note that setting max_length to the max value 512 for smaller passage sizes will negatively affect response time.  


## POST /convert for Conversion to Markdown

It provides the ability to convert to markdown, files with structured format jsonschema, _json instance, yaml instance, xsd schema, xml instance, pydantic and DataClass_. The output is canonical form using markdown sections to re-layout the original document along with each property/element/attribute/class properties.  

The aim is to have a single format which can be relatively easier to chunk before embedding & vector ingestion, without losing a lot on the context , i.e the structure of the file.

**Endpoint**: `/convert`

### Example
Here we pass the file in the json body with its content encoded in Base64, alternatively you can use multi-part request to pass the file as binary (in such case the params max_example_length & max_enum_values can be passed as query parameters)

```bash
curl -X POST "http://localhost:8000/convert" \
  -H "content-type: application/json" \
  -d '{
  "filename":"example.json",
  "max_example_length": 100,
  "max_enum_values": 5,
  "content_base64":"ewogICIkc2NoZW1hIjogImh0dHA6Ly9qc29uLXNjaGVtYS5vcmcvZHJhZnQtMDQvc2NoZW1hIyIsCiAgInRpdGxlIjogIlJlcXVlc3QuRm9yZWlnbl9FeGNoYW5nZS5Gb3J3YXJkLkZvcndhcmQuSW5zdFJlZkRhdGFSZXBvcnRpbmciLAogICJMYXN0TW9kaWZ5RGF0ZVRpbWUiOiAiMjAyMi0wNS0xN1QxNDo0OTo0NCIsCiAgImRlc2NyaXB0aW9uIjogIlJlcXVlc3QgdGVtcGxhdGUgZm9yIEZvcmVpZ25fRXhjaGFuZ2UgRm9yd2FyZCBGb3J3YXJkIiwKICAiY29weXJpZ2h0IjogIkNvcHlyaWdodCDCqSAyMDIwIFRoZSBEZXJpdmF0aXZlcyBTZXJ2aWNlIEJ1cmVhdSAoRFNCKSBMaW1pdGVkLiBBbGwgUmlnaHRzIFJlc2VydmVkLiIsCiAgInR5cGUiOiAib2JqZWN0IiwKICAicHJvcGVydGllcyI6IHsKICAgICJIZWFkZXIiOiB7CiAgICAgICJ0aXRsZSI6ICJIZWFkZXIiLAogICAgICAidHlwZSI6ICJvYmplY3QiLAogICAgICAicHJvcGVydGllcyI6IHsKICAgICAgICAiQXNzZXRDbGFzcyI6IHsKICAgICAgICAgICJ0aXRsZSI6ICJBc3NldCBDbGFzcyIsCiAgICAgICAgICAiZGVzY3JpcHRpb24iOiAiQXMgZGVmaW5lZCBieSBDRkkgY29kZTogSVNPIDEwOTYyICgyMDE1KTsgQ2hhcmFjdGVyICMyIiwKICAgICAgICAgICJ0eXBlIjogInN0cmluZyIsCiAgICAgICAgICAiZW51bSI6IFsKICAgICAgICAgICAgIkZvcmVpZ25fRXhjaGFuZ2UiCiAgICAgICAgICBdCiAgICAgICAgfSwKICAgICAgICAiSW5zdHJ1bWVudFR5cGUiOiB7CiAgICAgICAgICAidGl0bGUiOiAiSW5zdHJ1bWVudCBUeXBlIiwKICAgICAgICAgICJkZXNjcmlwdGlvbiI6ICJBcyBkZWZpbmVkIGJ5IENGSSBjb2RlOiBJU08gMTA5NjIgKDIwMTUpOyBDaGFyYWN0ZXIgIzEiLAogICAgICAgICAgInR5cGUiOiAic3RyaW5nIiwKICAgICAgICAgICJlbnVtIjogWwogICAgICAgICAgICAiRm9yd2FyZCIKICAgICAgICAgIF0KICAgICAgICB9LAogICAgICAgICJVc2VDYXNlIjogewogICAgICAgICAgInRpdGxlIjogIlByb2R1Y3QiLAogICAgICAgICAgImRlc2NyaXB0aW9uIjogIlVuaXF1ZSBsYWJlbCB0aGF0IGRlZmluZXMgdGhlIHByb2R1Y3QiLAogICAgICAgICAgInR5cGUiOiAic3RyaW5nIiwKICAgICAgICAgICJlbnVtIjogWwogICAgICAgICAgICAiRm9yd2FyZCIKICAgICAgICAgIF0KICAgICAgICB9LAogICAgICAgICJMZXZlbCI6IHsKICAgICAgICAgICJ0aXRsZSI6ICJMZXZlbCIsCiAgICAgICAgICAiZGVzY3JpcHRpb24iOiAiTGFiZWwgYXNzaWduZWQgdG8gdGhlIElTSU4gdG8gZGVzY3JpYmUgaXRzIGxldmVsIGluIHRoZSBoaWVyYXJjaHkiLAogICAgICAgICAgInR5cGUiOiAic3RyaW5nIiwKICAgICAgICAgICJlbnVtIjogWwogICAgICAgICAgICAiSW5zdFJlZkRhdGFSZXBvcnRpbmciCiAgICAgICAgICBdCiAgICAgICAgfQogICAgICB9LAogICAgICAicmVxdWlyZWQiOiBbCiAgICAgICAgIkFzc2V0Q2xhc3MiLAogICAgICAgICJJbnN0cnVtZW50VHlwZSIsCiAgICAgICAgIlVzZUNhc2UiLAogICAgICAgICJMZXZlbCIKICAgICAgXSwKICAgICAgImFkZGl0aW9uYWxQcm9wZXJ0aWVzIjogZmFsc2UKICAgIH0sCiAgICAiQXR0cmlidXRlcyI6IHsKICAgICAgInR5cGUiOiAib2JqZWN0IiwKICAgICAgInByb3BlcnRpZXMiOiB7CiAgICAgICAgIk5vdGlvbmFsQ3VycmVuY3kiOiB7CiAgICAgICAgICAiJHJlZiI6ICIuLi8uLi9jb2Rlc2V0cy9JU09DdXJyZW5jeUNvZGUuanNvbiIsCiAgICAgICAgICAidGl0bGUiOiAiTm90aW9uYWwgQ3VycmVuY3kiLAogICAgICAgICAgImRlc2NyaXB0aW9uIjogIkN1cnJlbmN5IGluIHdoaWNoIHRoZSBub3Rpb25hbCBpcyBkZW5vbWluYXRlZC4gVXNhZ2U6IFdpdGhpbiBNaUZJUiwgaW4gdGhlIGNhc2Ugb2YgYW4gaW50ZXJlc3QgcmF0ZSBvciBjdXJyZW5jeSBkZXJpdmF0aXZlIGNvbnRyYWN0LCB0aGlzIHdpbGwgYmUgdGhlIG5vdGlvbmFsIGN1cnJlbmN5IG9mIGxlZyAxIG9yIHRoZSBjdXJyZW5jeSAxIG9mIHRoZSBwYWlyLiBXaXRoaW4gTWlGSVIsIGluIHRoZSBjYXNlIG9mIHN3YXB0aW9ucyB3aGVyZSB0aGUgdW5kZXJseWluZyBzd2FwIGlzIHNpbmdsZS1jdXJyZW5jeSwgdGhpcyB3aWxsIGJlIHRoZSBub3Rpb25hbCBjdXJyZW5jeSBvZiB0aGUgdW5kZXJseWluZyBzd2FwLiBGb3Igc3dhcHRpb25zIHdoZXJlIHRoZSB1bmRlcmx5aW5nIGlzIG11bHRpLWN1cnJlbmN5LCB0aGlzIHdpbGwgYmUgdGhlIG5vdGlvbmFsIGN1cnJlbmN5IG9mIGxlZyAxIG9mIHRoZSBzd2FwIgogICAgICAgIH0sCiAgICAgICAgIkV4cGlyeURhdGUiOiB7CiAgICAgICAgICAidGl0bGUiOiAiRXhwaXJ5IERhdGUiLAogICAgICAgICAgImRlc2NyaXB0aW9uIjogIkV4cGlyeSBkYXRlIG9mIHRoZSBpbnN0cnVtZW50IChZWVlZLU1NLUREKSIsCiAgICAgICAgICAidHlwZSI6ICJzdHJpbmciLAogICAgICAgICAgInBhdHRlcm4iOiAiXlswLTldezR9LVswLTldezJ9LVswLTldezJ9JCIsCiAgICAgICAgICAiZm9ybWF0IjogImRhdGUiCiAgICAgICAgfSwKICAgICAgICAiT3RoZXJOb3Rpb25hbEN1cnJlbmN5IjogewogICAgICAgICAgIiRyZWYiOiAiLi4vLi4vY29kZXNldHMvSVNPQ3VycmVuY3lDb2RlLmpzb24iLAogICAgICAgICAgInRpdGxlIjogIk90aGVyIE5vdGlvbmFsIEN1cnJlbmN5IiwKICAgICAgICAgICJkZXNjcmlwdGlvbiI6ICJJbiB0aGUgY2FzZSBvZiBtdWx0aS1jdXJyZW5jeSBvciBjcm9zcy1jdXJyZW5jeSBzd2FwcyB0aGUgY3VycmVuY3kgaW4gd2hpY2ggbGVnIDIgb2YgdGhlIGNvbnRyYWN0IGlzIGRlbm9taW5hdGVkOyBGb3Igc3dhcHRpb25zIHdoZXJlIHRoZSB1bmRlcmx5aW5nIHN3YXAgaXMgbXVsdGktY3VycmVuY3ksIHRoZSBjdXJyZW5jeSBpbiB3aGljaCBsZWcgMiBvZiB0aGUgc3dhcCBpcyBkZW5vbWluYXRlZCIKICAgICAgICB9LAogICAgICAgICJEZWxpdmVyeVR5cGUiOiB7CiAgICAgICAgICAiZGVmYXVsdCI6ICJQSFlTIiwKICAgICAgICAgICJ0aXRsZSI6ICJEZWxpdmVyeSBUeXBlIiwKICAgICAgICAgICJkZXNjcmlwdGlvbiI6ICJUaGUgRGVsaXZlcnkgVHlwZSBhcyBkZWZpbmVkIGJ5IENGSSBjb2RlOiBJU08gMTA5NjIgKDIwMTUpIiwKICAgICAgICAgICJ0eXBlIjogInN0cmluZyIsCiAgICAgICAgICAiZW51bSI6IFsKICAgICAgICAgICAgIkNBU0giLAogICAgICAgICAgICAiUEhZUyIKICAgICAgICAgIF0sCiAgICAgICAgICAiZWxhYm9yYXRpb24iOiB7CiAgICAgICAgICAgICJDQVNIIjogInRoZSBkaXNjaGFyZ2Ugb2YgYW4gb2JsaWdhdGlvbiBieSBwYXltZW50IG9yIHJlY2VpcHQgb2YgYSBuZXQgY2FzaCBhbW91bnQgaW5zdGVhZCBvZiBwYXltZW50IG9yIGRlbGl2ZXJ5IGJ5IGJvdGggcGFydGllcyIsCiAgICAgICAgICAgICJQSFlTIjogInRoZSBtZWV0aW5nIG9mIGEgc2V0dGxlbWVudCBvYmxpZ2F0aW9uIHVuZGVyIGEgZGVyaXZhdGl2ZSBjb250cmFjdCB0aHJvdWdoIHRoZSByZWNlaXB0IG9yIGRlbGl2ZXJ5IG9mIHRoZSBhY3R1YWwgdW5kZXJseWluZyBpbnN0cnVtZW50KHMpIGluc3RlYWQgb2YgdGhyb3VnaCBjYXNoIHNldHRsZW1lbnQiCiAgICAgICAgICB9LAogICAgICAgICAgIm9wdGlvbnMiOiB7CiAgICAgICAgICAgICJlbnVtX3RpdGxlcyI6IFsKICAgICAgICAgICAgICAiQ2FzaCIsCiAgICAgICAgICAgICAgIlBoeXNpY2FsIgogICAgICAgICAgICBdCiAgICAgICAgICB9CiAgICAgICAgfSwKICAgICAgICAiUHJpY2VNdWx0aXBsaWVyIjogewogICAgICAgICAgImRlZmF1bHQiOiAxLAogICAgICAgICAgInRpdGxlIjogIlByaWNlIE11bHRpcGxpZXIiLAogICAgICAgICAgImRlc2NyaXB0aW9uIjogIk51bWJlciBvZiB1bml0cyBvZiB0aGUgdW5kZXJseWluZyBpbnN0cnVtZW50IHJlcHJlc2VudGVkIGJ5IGEgc2luZ2xlIGRlcml2YXRpdmUgY29udHJhY3QuIE51bWJlciBvZiB1bml0cyBvZiB0aGUgdW5kZXJseWluZyBpbnN0cnVtZW50IHJlcHJlc2VudGVkIGJ5IGEgc2luZ2xlIGRlcml2YXRpdmUgY29udHJhY3QuIEZvciBhbiBvcHRpb24gb24gYW4gaW5kZXgsIHRoZSBhbW91bnQgcGVyIGluZGV4IHBvaW50LiBGb3Igc3ByZWFkYmV0cyB0aGUgbW92ZW1lbnQgaW4gdGhlIHByaWNlIG9mIHRoZSB1bmRlcmx5aW5nIGluc3RydW1lbnQgb24gd2hpY2ggdGhlIHNwcmVhZGJldCBpcyBiYXNlZCIsCiAgICAgICAgICAidHlwZSI6ICJudW1iZXIiLAogICAgICAgICAgIm1pbmltdW0iOiAwLAogICAgICAgICAgImV4Y2x1c2l2ZU1pbmltdW0iOiB0cnVlCiAgICAgICAgfQogICAgICB9LAogICAgICAicmVxdWlyZWQiOiBbCiAgICAgICAgIk5vdGlvbmFsQ3VycmVuY3kiLAogICAgICAgICJFeHBpcnlEYXRlIiwKICAgICAgICAiT3RoZXJOb3Rpb25hbEN1cnJlbmN5IgogICAgICBdLAogICAgICAiYWRkaXRpb25hbFByb3BlcnRpZXMiOiBmYWxzZQogICAgfQogIH0sCiAgInJlcXVpcmVkIjogWwogICAgIkhlYWRlciIsCiAgICAiQXR0cmlidXRlcyIKICBdLAogICJhZGRpdGlvbmFsUHJvcGVydGllcyI6IGZhbHNlCn0=  
  "}'
```

Response
```bash
{
  "detected_type": "json_schema",
  "markdown": "# request.foreign_exchange.swap.fx_swap.instrefdatareporting\n- description: request template for foreign_exchange swap fx_swap\n- type: object\n- constraint: additionalproperties=false\n- required: attributes, header\n\n## attributes\n- type: object\n- constraint: additionalproperties=false\n- required: true\n\n### deliverytype\n- description: the delivery type as defined by cfi code: iso 10962\n- type: string\n- constraint: default=phys\n- required: false\n- enums: phys, cash\n\n### instrumentisinfarleg\n- description: isin code of the underlying instrument - far leg\n- type: string\n- constraint: pattern=^[a-z]{2}[a-z0-9]{9}[0-9]$\n- required: true\n\n### instrumentisinnearleg\n- description: isin code of the underlying instrument - near leg\n- type: string\n- constraint: pattern=^[a-z]{2}[a-z0-9]{9}[0-9]$\n- required: true\n\n### pricemultiplier\n- description: number of units of the underlying instrument represented by a single derivative contract. number of units of the underlying instrument represented by a single derivative contract. for an option on an index, the amount per index point. for spreadbets the movement in the price of the underlying instrument on which the spreadbet is based\n- type: number\n- constraint: minimum=0; exclusiveminimum=true; default=1\n- required: false\n\n## header\n- type: object\n- constraint: additionalproperties=false\n- required: true\n\n### assetclass\n- description: as defined by cfi code: iso 10962; character #2\n- type: string\n- required: true\n- enums: foreign_exchange\n\n### instrumenttype\n- description: as defined by cfi code: iso 10962; character #1\n- type: string\n- required: true\n- enums: swap\n\n### level\n- description: label assigned to the isin to describe its level in the hierarchy\n- type: string\n- required: true\n- enums: instrefdatareporting\n\n### usecase\n- description: unique label that defines the product\n- type: string\n- required: true\n- enums: fx_swap\n"
}
```

## POST /classify for Classification of text

It provides the ability to classify financial text/data into one ore more of the 'labels' you provide in the request. It returns the list of matching labels along with their scores, where a **score > 0,7** is usually where accepted results start.  

The classification uses `facebook/bart-large-mnli` [model](https://huggingface.co/facebook/bart-large-mnli) which is NLI-based **Zero Shot** Text Classification model 

**Endpoint**: `/classify`

### Example

```bash
curl -X POST http://localhost:8181/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "root: fpml:trade elements: swap fixedRateSchedule floatingRateIndex notional paymentDate",
    "labels": ["Interest Rate Swap", "FX Forward", "Bond", "Equity Option", "Credit Default Swap"],
    "multi_label": false,
    "top_k": 3
  }'
  ```

```json
{
  "model_id": "facebook/bart-large-mnli",
  "labels": [
    "Interest Rate Swap",
    "FX Forward",
    "Credit Default Swap"
  ],
  "scores": [
    0.7382494211196899,
    0.09017873555421829,
    0.08254387974739075
  ],
  "sequence": "root: fpml:trade elements: swap fixedRateSchedule floatingRateIndex notional paymentDate"
}
```


# Dev & Build images 

## Dev

**Requirements**:

- Python **3.12+**
- [`uv`](https://docs.astral.sh/uv/)
- Docker (optional)

**Development environment**:

After installing `uv`, then from the repository root:

```bash
uv venv
uv sync
```

This creates `.venv/` and installs dependencies from `pyproject.toml`  


To run the service locally:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8181 --reload
```

## Docker build & test

### Build the Docker image

The Dockerfile warms up and caches the FlashRank models during the build.

A build script is provided

```bash
./build_image.sh 
```

Or if you want to be specific about the platform/architecture, for example: amd64

```bash
docker buildx build \
  --platform linux/arm64 \
  -t falriachi/rag-utils:arm64 \
  --load \
  --progress=plain \
  .
```

**Notes**

* Re-ranking is CPU-bound and executed in a threadpool to keep the async API responsive.
* Re-ranking LLM (listwise) image and re-ranking is relatively much heavier on memory & CPU than cross-encoder (pairwise)
* Docker builds may take longer due to model downloads during the warmup step.