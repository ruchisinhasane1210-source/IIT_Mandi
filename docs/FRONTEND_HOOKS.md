# Frontend Integration Hooks & API Reference (Track D — Stage 2)

This document specifies all newly introduced backend endpoints, schemas, and UI hooks for frontend consumption. Per the project constraints, **no existing frontend files were modified**. When the UI is extended, developers should use the hooks documented below.

---

## 1. Deep Linking Scheme

All content units now generate a unified canonical deep link in the `deep_link` attribute following this format:

| Modality | Deep Link Format | Query Parameters | Description |
| :--- | :--- | :--- | :--- |
| **PDF** | `/viewer?source=<source_id>&page=<n>` | `source`, `page` | Opens PDF viewer jump-scrolled to page `n` |
| **Slides (PPTX)** | `/viewer?source=<source_id>&slide=<n>` | `source`, `slide` | Opens slide deck jumped to slide `n` |
| **Video / Audio** | `/viewer?source=<source_id>&t=<seconds>` | `source`, `t` | Jumps media player to playback timestamp in seconds |
| **Figures** | `/viewer?source=<source_id>&page=<n>` or `&slide=<n>` | `source`, `page`/`slide` | Focuses on the figure container and highlights OCR/caption |

### Recommended Viewer UI Implementation
If building `/viewer`:
- URL: `GET /viewer`
- Parse URL query string: `source`, `page`, `slide`, `t`.
- Call `GET /source/<unit_id>` to retrieve exact citation text, surrounding neighbor IDs, and bounding coordinates.

---

## 2. Ingestion & Sources Endpoints

### `POST /ingest/upload` (also `/api/ingest/upload`)
Initiates background ingestion for a file upload or remote URL.
- **Content-Type**: `multipart/form-data` OR `application/json`
- **Request Body**:
  - Multipart: `file` (file upload), optional `title`, optional `course_id` (default: `dsa_6006`).
  - JSON: `{"url": "https://...", "title": "Lecture Video", "course_id": "dsa_6006"}`
- **Response** (`202 Accepted`):
```json
{
  "job_id": "52f7823b-6401-4971-8bc4-9d5f3032b49d",
  "source_id": "09cc6992-3ae3-4416-a276-4ac8614b5661"
}
```

### `GET /ingest/status/<job_id>` (also `/api/ingest/status/<job_id>`)
Polls progress for an asynchronous ingestion job.
- **Response** (`200 OK`):
```json
{
  "job_id": "52f7823b-6401-4971-8bc4-9d5f3032b49d",
  "source_id": "09cc6992-3ae3-4416-a276-4ac8614b5661",
  "stage": "done",
  "progress": 100,
  "error": null,
  "counts": {
    "units": 18,
    "figures": 2,
    "by_modality": {"text": 14, "figure": 2, "table": 2},
    "coverage_pct": 100.0
  },
  "started_at": "2026-10-14 17:45:00",
  "finished_at": "2026-10-14 17:45:04"
}
```

### `GET /sources` (also `/api/sources`)
Lists all ingested sources with aggregated unit counts.
- **Response** (`200 OK`):
```json
{
  "sources": [
    {
      "id": "ocw_6006_lec08",
      "title": "Lecture 8: Binary Heaps",
      "type": "slides",
      "status": "done",
      "unit_count": 3,
      "num_pages": null,
      "duration_seconds": null
    }
  ]
}
```

### `DELETE /sources/<source_id>` (also `/api/sources/<source_id>`)
Removes a source along with its content units, figures, and vector embeddings.
- **Response** (`200 OK`):
```json
{
  "ok": true,
  "deleted_source_id": "ocw_6006_lec08",
  "deleted_units": 3
}
```

---

## 3. Source Unit Excerpt & Deep Link Endpoint

### `GET /source/<unit_id>` (also `/api/source/<unit_id>`)
Fetches a single unit with location details and adjacent navigation links.
- **Response** (`200 OK`):
```json
{
  "unit_id": "u_heap_1",
  "id": "u_heap_1",
  "source_id": "ocw_6006_lec08",
  "source_title": "Lecture 8: Binary Heaps",
  "excerpt": "A binary heap is a complete binary tree stored implicitly in an array...",
  "content": "A binary heap is a complete binary tree stored implicitly in an array. Index i has children at 2i+1 and 2i+2...",
  "modality": "slide",
  "location": {
    "page": null,
    "slide": 12,
    "timestamp_start": null,
    "timestamp_end": null
  },
  "deep_link": "/viewer?source=ocw_6006_lec08&slide=12",
  "neighbors": {
    "prev": null,
    "next": "u_heap_3"
  }
}
```

---

## 4. Knowledge Graph Endpoints

### `GET /knowledge/graph` (also `/api/knowledge/graph`)
Returns canonical topics, concepts, prerequisite edges, and unit counts per node.
- **Response** (`200 OK`):
```json
{
  "topics": [
    {
      "id": "t_heaps",
      "name": "Heaps & Priority Queues",
      "level": "topic",
      "summary": "Binary heaps, heapify, build-heap in O(n)."
    }
  ],
  "concepts": [
    {
      "id": "c_binary_heap",
      "name": "Binary Heap",
      "definition": "Array-backed complete binary tree satisfying heap property",
      "topic_id": "t_heaps"
    }
  ],
  "prerequisite_edges": [
    {
      "source": "t_arrays",
      "target": "t_heaps",
      "confidence": 0.85,
      "reason": "Required foundational knowledge"
    }
  ],
  "unit_counts_per_node": {
    "t_heaps": 3,
    "c_binary_heap": 2
  }
}
```

### `GET /knowledge/topic/<topic_id>` (also `/api/knowledge/topic/<topic_id>`)
Returns detailed topic metadata, child subtopics, concepts, and linked content units with source titles.

### `GET /knowledge/concept/<concept_id>` (also `/api/knowledge/concept/<concept_id>`)
Returns concept definition, incoming prerequisites, dependent concepts, and tagged units.

---

## 5. Retrieval Debug Endpoint

### `POST /retrieve/debug` (also `/api/retrieve/debug`)
Evaluates query retrieval with optional filtering and returns full hits with support scores.
- **Request Body**:
```json
{
  "query": "Where are children stored in a binary heap?",
  "k": 3,
  "filters": {
    "topic_id": "t_heaps",
    "modality": "slide"
  }
}
```
- **Response** (`200 OK`):
```json
{
  "query": "Where are children stored in a binary heap?",
  "k": 3,
  "filters": {"topic_id": "t_heaps", "modality": "slide"},
  "hits": [
    {
      "unit_id": "u_heap_1",
      "content": "A binary heap is a complete binary tree stored implicitly in an array. Index i has children at 2i+1 and 2i+2...",
      "score": 0.625,
      "modality": "slide",
      "deep_link": "/viewer?source=ocw_6006_lec08&slide=12",
      "location": {"page": null, "slide": 12, "timestamp_start": null, "timestamp_end": null},
      "topic_tags": ["t_heaps"],
      "concept_tags": ["c_binary_heap"]
    }
  ],
  "support_score": 0.652,
  "grounding_status": "grounded"
}
```
