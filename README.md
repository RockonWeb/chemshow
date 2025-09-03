# 🧬 Metabolome Handbook

Educational metabolomics reference for biochemistry and chemistry courses. Search metabolites by mass, formula, or name, and annotate your LC-MS data.

## 📋 Features

**Search Metabolites**: Find by mass (m/z), chemical formula, or name

**Database Integration**: Links to HMDB, KEGG, ChEBI, PubChem, and UniProt

**CSV Annotation**: Upload LC-MS peak lists for automatic annotation

**Export Results**: Download results as CSV or Excel files

* **Educational Focus**: Designed for learning biochemistry and chemistry

## 🚀 Quick Start

### Option 1: Using Virtual Environment (.venv)

#### Prerequisites

- Python 3.11+
- pip

#### Setup

```bash
# Clone or download the project
cd Protein

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create data directory
mkdir -p data

# Import sample data
python data/import_data.py

# Start API server (in one terminal)
uvicorn api.app.main:app --host 0.0.0.0 --port 8000 --reload

# Start UI (in another terminal - don't forget to activate .venv)
streamlit run ui/main.py --server.address 0.0.0.0 --server.port 8501
```

#### Access the Application

- **Web Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

### Option 2: Using Docker

#### Prerequisites

- Docker
- Docker Compose

#### Quick Start with SQLite (Development)

```bash
# Start all services with SQLite
docker-compose -f docker-compose.dev.yml up --build

# Access:
# Web Interface: http://localhost:8501
# API: http://localhost:8000/docs
```

#### Production Setup with PostgreSQL

```bash
# Start all services with PostgreSQL
docker-compose up --build

# Access:
# Web Interface: http://localhost:8501
# API: http://localhost:8000/docs
# Database: localhost:5432
```

## 📊 Sample Data

The project includes sample metabolites data with:

- 15+ common metabolites (glucose, pyruvate, amino acids, etc.)
- Biochemical pathways (Glycolysis, TCA cycle, etc.)
- Enzyme associations
- External database IDs (HMDB, KEGG, ChEBI, PubChem)

Sample CSV file for testing annotation: `data/sample_data.csv`

## 🔧 API Endpoints

### Search

- `GET /metabolites/search` - Search metabolites
- `GET /metabolites/{id}` - Get metabolite details

### Annotation

- `POST /annotate/csv` - Annotate CSV file
- `POST /annotate/mz-list` - Annotate m/z list

### Utility

- `GET /health` - Health check
- `GET /` - API information

### Example API Usage

```python
import requests

# Search by mass
response = requests.get("http://localhost:8000/metabolites/search", 
                       params={"mass": 180.063, "tol_ppm": 10})
print(response.json())

# Search by name
response = requests.get("http://localhost:8000/metabolites/search", 
                       params={"q": "glucose"})
print(response.json())
```

## 📁 Project Structure

```
Protein/
├── api/                    # FastAPI backend
│   ├── app/               # Application code
│   ├── database/          # Database configuration
│   ├── models/            # SQLAlchemy models
│   ├── schemas/           # Pydantic schemas
│   └── services/          # Business logic
├── ui/                    # Streamlit frontend
├── data/                  # Data and import scripts
├── docker/                # Docker configurations
├── alembic/               # Database migrations
├── tests/                 # Test files
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Production docker setup
├── docker-compose.dev.yml # Development docker setup
└── README.md             # This file
```

## 🧪 Educational Use Cases

### Laboratory Exercises

1. **Mass Spectrometry Analysis**: Upload peak lists from LC-MS experiments
2. **Metabolite Identification**: Practice identifying unknowns by mass
3. **Pathway Mapping**: Connect metabolites to biochemical pathways

### Example Assignments

- Annotate 20 LC-MS peaks and identify key metabolites
- Find metabolites for mass 180.063 ±10 ppm and create a reference table
- Map identified metabolites to glycolysis pathway

## 🛠️ Development

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Adding New Data

1. Edit `data/import_data.py` to add more metabolites
2. Run the import script: `python data/import_data.py`
3. Or use the API to add data programmatically

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=api tests/
```

## 📚 Data Sources

- **HMDB**: Human Metabolome Database
- **ChEBI**: Chemical Entities of Biological Interest
- **KEGG**: Kyoto Encyclopedia of Genes and Genomes
- **PubChem**: Chemical information database
- **LIPID MAPS**: Lipidomics data and tools

## ⚖️ License and Usage

This is an educational tool designed for academic use. Please check individual database licenses:

- HMDB: Free for academic use
- ChEBI: Open data
- KEGG: Academic use allowed, commercial use requires license
- PubChem: Public domain

## 🔧 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=sqlite:///./data/metabolome.db

# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# UI
STREAMLIT_PORT=8501
```

### Customization

- Modify search tolerances in the UI
- Add new metabolite classes in `data/import_data.py`
- Extend API endpoints in `api/app/main.py`
- Customize UI layout in `ui/main.py`

## 🆘 Troubleshooting

### Common Issues

1. **Import errors**: Make sure `.venv` is activated and all dependencies are installed
2. **Database errors**: Check if data directory exists and is writable
3. **API connection errors**: Ensure API is running on port 8000
4. **Permission errors**: On Linux/Mac, you might need to set file permissions

### Getting Help

- Check the API documentation at `/docs`
- Review log files for error details
- Ensure all services are running and healthy

## 🚧 Future Enhancements

- Support for mzML file format
- Pathway visualization
- User authentication
- More sophisticated mass spectrometry features
- Integration with more databases
- Advanced statistical analysis tools

---

**🧬 Metabolome Handbook** - Making metabolomics accessible for education!
