Here’s a clean and professional `README.md` adapted to your new project, reflecting the expanded multi-channel capability:

---

# End-to-End Semantic Search Engine for Educational YouTube Content

Semantic search engine to explore multiple educational YouTube channels using sentence embeddings and vector similarity. Supports natural language queries across a curated set of tech and programming-focused channels.

---

##  Features

*  **Semantic Search** over content from multiple educational YouTube channels.
*  **ETL Pipeline** for automated fetching, processing, and embedding of video metadata.
*  **FastAPI Backend**, containerized with Docker and deployable to **Google Cloud Run**.
*  **Gradio UI** integration possible (via Hugging Face Spaces or local).
*  **CI/CD** with GitHub Actions and Google Cloud Build for automatic updates.

---

## Supported Channels

Currently indexed YouTube channels:

* `freecodecamp`
* `3blue1brown`
* `firstprinciples`
* `jeffheaton`
* `krishnaik`
* `netninja`
* `techworldnana`
* `theconstructsim`
* `yousuckatprogramming`

Each channel has its own searchable index.

---

## 🏗System Architecture

![System Architecture](https://github.com/AzzedineNed/E2E-Semantic-Search-Engine-for-Educational-YouTube-Content/blob/master/end-to-end-semantic-search-for-freecodecamp-videos.png)

---

## Semantic Search Logic

Uses the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model from `sentence-transformers` to embed video metadata and search queries.

* Comparison is done via **Manhattan distance** (L1 norm).
* Embedding indexes are stored in `.parquet` files inside `app/data/`.

Example:

```bash
curl -X POST "http://localhost:8080/search/netninja?query=routes%20basics"
```

---

##  Project Structure

```
E2E-Semantic-Search-Engine-for-Educational-YouTube-Content
├── app
│   ├── data/                        # Parquet indexes for each channel
│   ├── __init__.py
│   ├── main.py                      # FastAPI entrypoint
│   └── search_function.py           # Search logic
├── data_pipeline
│   ├── Data_Pipeline.py             # Main pipeline script
│   ├── ETL.py                       # ETL logic (fetch, clean, embed)
│   └── requirements.txt
├── .github/
│   └── workflows/
│       └── data-pipeline.yml        # Scheduled GitHub Actions workflow
├── Dockerfile
├── README.md
├── requirements.txt
└── .gitignore
```

---

##  Data Pipeline

Automated pipeline using GitHub Actions:

* 🕓 Scheduled to run **every Saturday at 4:00 AM** or manually.
* 🧬 Fetches video metadata using **YouTube Data API v3**.
* 🧼 Cleans and transforms video titles, descriptions, and transcripts.
* 🤖 Generates sentence embeddings using `sentence-transformers`.
* 💾 Saves `.parquet` indexes for each channel in `app/data/`.

> Only the `app/` directory is included in the Docker image.

---

##  Secrets & Configuration

To enable CI/CD and pipeline execution, set the following GitHub repository secrets:

| Secret Name             | Description                                         |
| ----------------------- | --------------------------------------------------- |
| `YT_API_KEY`            | YouTube Data API v3 key                             |
| `PERSONAL_ACCESS_TOKEN` | GitHub Personal Access Token (for workflow commits) |

➡️ Set these under: GitHub → `Settings` → `Secrets and variables` → `Actions`

---

##  Deployment

* **App:** FastAPI application running on port `8080`.
* **Containerized:** With Docker.
* **Cloud Ready:** Deployable to **Google Cloud Run**.
* **CI/CD:** GitHub Actions + Google Cloud Build for automated deployments on push.

---

##  Docker

### Build the container

```bash
docker build -t semantic-search-app .
```

### Run locally

```bash
docker run -p 8080:8080 semantic-search-app
```

---

## 🔧 Local Development

### Run FastAPI app

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8080
```

### Run the ETL pipeline

```bash
cd data_pipeline
python Data_Pipeline.py
```

---

## 🔍 Test the API

Example POST request:

```bash
curl "http://localhost:8080/search/netninja?query=routes%20basics"
```
---
