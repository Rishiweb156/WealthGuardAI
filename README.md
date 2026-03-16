# 💎 WealthGuard AI - Predictive Financial Health Platform

**WealthGuard AI** is an financial health analysis platform that transforms traditional transaction analysis into **prescriptive AI** using **Graph RAG** and **Time-Series Forecasting**.

## Key Features

### Advanced AI Capabilities
- **Graph RAG Engine**: Knowledge graph-based relationship discovery in spending patterns using NetworkX
- **Predictive Forecasting**: 30-day balance prediction with confidence intervals using Facebook Prophet
- **ML Anomaly Detection**: Isolation Forest-based transaction anomaly detection
- **Conversational AI**: Natural language financial advisor powered by Ollama LLM

### Core Analytics
- **PDF Parsing**: Extract transactions from HDFC bank statements
- **Smart Categorization**: Rule-based + LLM hybrid transaction categorization
- **Pattern Detection**: Identify spending patterns, recurring payments, and fees
- **Interactive Visualizations**: Real-time charts with Chart.js

### Modern UI
- **Responsive Dashboard**: Beautiful dark-mode interface with gradient accents
- **Real-time Updates**: Live data refresh without page reload
- **Interactive Graphs**: D3.js force-directed spending network visualization
- **AI Chat Interface**: Conversational queries about your finances

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   WealthGuard AI Platform                    │
├─────────────────────────────────────────────────────────────┤
│  Frontend Layer (HTML5/CSS3/JS + Streamlit)                 │
│  ├─ Modern Dashboard with Chart.js                          │
│  ├─ Interactive Graph Visualization (D3.js)                 │
│  └─ Real-time Forecast Timeline                             │
├─────────────────────────────────────────────────────────────┤
│  API Layer (FastAPI)                                         │
│  ├─ RESTful endpoints with CORS                             │
│  ├─ File upload & processing pipeline                       │
│  └─ WebSocket support for real-time updates                 │
├─────────────────────────────────────────────────────────────┤
│  Intelligence Layer                                         │
│  ├─ Graph Engine (NetworkX) - Relationship Discovery        │
│  ├─ Forecaster (Prophet) - Time-Series Prediction           │
│  ├─ Anomaly Detector (Isolation Forest) - ML Detection      │
│  └─ LLM Agent (Ollama) - Conversational Interface           │
├─────────────────────────────────────────────────────────────┤
│  Data Pipeline (Prefect)                                     │
│  └─ PDF → Parse → Timeline → Categorize → Analyze → Predict │
└─────────────────────────────────────────────────────────────┘
```

---

##  Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose (for containerized deployment)
- Ollama (for LLM features)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/wealthguard-ai.git
cd wealthguard-ai

# 2. Virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the LLM model (Ollama must be running)
ollama pull llama3.2:3b

# 4. Run backend
python -m uvicorn server:app --reload --port 8000

# 5. Access dashboard
open http://localhost:8000
```

### Docker Deployment

```bash
# Start all services
docker-compose up --build

# Access services:
# - Dashboard: http://localhost:8000
# - Streamlit: http://localhost:8501
# - MLflow: http://localhost:5000
# - Ollama: http://localhost:11434
```
<!-- Frontend: http://localhost:8501
Backend API: http://localhost:8000/docs (API documentation)
MLflow: http://localhost:5000 -->
---

## 📖 Usage

### 1. Upload Bank Statements
- Click "Upload Statements" button
- Drag & drop or browse PDF files (max 10)
- System automatically processes and categorizes transactions

### 2. Dashboard Analytics
- View account overview with key metrics
- Analyze spending trends with interactive charts
- Review AI-generated insights and patterns

### 3. Graph RAG Analysis
- Navigate to "Graph RAG" tab
- Click "Analyze Spending Graph"
- Discover hidden relationships in spending patterns
- View top spending sources and clusters

### 4. Predictive Forecasting
- Go to "Predictions" tab
- Click "Generate Forecast"
- View 30-day balance prediction with confidence intervals
- Get AI-powered recommendations

### 5. AI Assistant
- Open "AI Assistant" tab
- Ask questions like:
  - "What are my top expenses this month?"
  - "Show me recurring payments"
  - "When did I last spend on shopping?"
  - "Predict my balance for next month"

---

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance async API framework
- **NetworkX**: Graph algorithms for relationship discovery
- **Prophet**: Facebook's time-series forecasting library
- **Scikit-learn**: Machine learning (Isolation Forest)
- **Ollama**: Local LLM inference
- **Prefect**: Workflow orchestration
- **MLflow**: Experiment tracking

### Frontend
- **HTML5/CSS3**: Modern semantic markup
- **Vanilla JavaScript**: No framework dependencies
- **Chart.js**: Beautiful chart visualizations
- **D3.js**: Force-directed graph layouts
- **Streamlit**: Alternative dashboard UI

### Data Processing
- **Pandas**: Data manipulation
- **PDFPlumber**: PDF text extraction
- **Tabula-py**: Tabular data extraction
- **PyPDF2**: PDF metadata parsing

---

## 📊 API Endpoints

### Core Pipeline
```
POST /parse_pdfs              - Upload and parse PDF statements
POST /build_timeline          - Create chronological timeline
POST /categorize_transactions - Categorize using rules + LLM
POST /analyze_transactions    - Detect patterns and anomalies
POST /generate_visualizations - Create chart data
POST /generate_stories        - Generate narrative summary
```

### Advanced AI 
```
GET  /graph-insights          - Graph RAG analysis
GET  /forecast                - Predictive forecasting
GET  /ml-anomalies            - ML-based anomaly detection
POST /conversational-query    - AI chat interface
```

### Data Retrieval
```
GET  /transactions            - Get all transactions
GET  /visualizations          - Get chart data
GET  /stories                 - Get financial narrative
GET  /health                  - System health check
```

---

## Highlights 

### Engineering Best Practices
- **Microservices Architecture**: Containerized services with Docker Compose
- **API Design**: RESTful endpoints with OpenAPI documentation
- **Error Handling**: Comprehensive try-catch with user-friendly error messages
- **Performance**: Sub-500ms API response times, async processing
- **Testing**: Unit tests with pytest, integration tests for full pipeline

### Innovation
- **Graph RAG**: First to apply knowledge graphs to personal finance analysis
- **Predictive AI**: Transformed descriptive analytics into prescriptive recommendations
- **Conversational Finance**: Natural language queries with LLM-powered responses

---

## Security & Privacy

- **Local Processing**: All data processed locally, no cloud uploads
- **No API Keys Required**: Uses open-source models (Ollama)
- **Data Isolation**: Each user's data stored separately
- **Secure File Upload**: File type validation, size limits
- **CORS Protection**: Configured CORS policies

---


## Contact

**Project Maintainer**: [Dadi Rishitha]
- GitHub: [@yourusername](https://github.com/Rishiweb156)
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/dadi-rishitha-867404283/)
- Email:142201005@smail.iitpkd.ac.in

---

**Built with Python, FastAPI, and AI**