# Development Guide

## Setting Up Development Environment

### Prerequisites

- Python 3.9 or higher
- Neo4j Community Edition
- Docker (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Sbursu/Carbon-EF.git
   cd Carbon-EF
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scriptsctivate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Project Structure

- `/data`: Contains all data processing scripts and datasets
  - `/raw`: Raw data files
  - `/processed`: Processed and harmonized datasets
  - `/scripts`: Data processing scripts

- `/neo4j`: Neo4j database configuration and import files
  - `/import`: CSV files for Neo4j import
  - `/scripts`: Database setup scripts

- `/training`: Model training code and datasets
  - `/scripts`: Training scripts
  - `/notebooks`: Jupyter notebooks
  - `/data`: Training datasets

### Development Workflow

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

3. Push your changes and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

### Testing

Run tests using pytest:
```bash
pytest
```

### Code Style

We use flake8 for linting. Check your code style:
```bash
flake8 .
```

### Documentation

- Update relevant documentation when making changes
- Follow the existing documentation style
- Include docstrings in your code

### Deployment

Deployment is handled automatically through GitHub Actions when changes are merged into the main branch.

## Getting Help

- Check the existing documentation in the `docs` directory
- Open an issue on GitHub
- Contact the maintainers
