# AI-Based Multimodal Recommendation System
Fine-tuning OpenAI CLIP (ViT-B/32) for clothing pattern retrieval 
using both image and text queries, served via Flask API and React frontend.

## Motivation
No publicly available CLIP-based dataset or retrieval system existed 
for clothing pattern recognition in the South Asian context. 
This project addresses that gap by constructing a domain-specific 
dataset from scratch and fine-tuning CLIP for cross-modal retrieval.

## Dataset
- 70,000+ raw image-text pairs scraped across 9 clothing categories 
  using Selenium
- Cleaned to 30,832 high-quality pairs using automated pipelines
- Dataset not included due to size — scraping and cleaning 
  scripts available in /data

## Results
| Metric | Score |
|--------|-------|
| Train Accuracy | 99.92% |
| Validation Accuracy | 80.44% |
| Baseline (pre fine-tuning) | 60% |
| Improvement | +20.44% |

## How It Works
1. Image or text query is submitted via the React frontend
2. CLIP encoders convert input to a 512-dim embedding
3. Cosine similarity search retrieves top-6 matching images
4. Results returned via Flask API

## Repository Structure
| Folder / File | Contents |
|---------------|----------|
| `data/` | Dataset scraping and preprocessing scripts |
| `development/` | Experimental training and retrieval scripts |
| `src/` | Final training script and Flask API |
| `frontend/` | React web application |
| `install.yml` | Conda environment |
| `thesis.pdf` | Full project thesis |

## Setup
```bash
conda env create -f install.yml
conda activate <env_name>
cd src && python app.py
```
Frontend: cd frontend && npm install && npm start

## Authors
Alian Khan | Supervisor: Dr. Naeem Akhtar | DCIS, PIEAS
