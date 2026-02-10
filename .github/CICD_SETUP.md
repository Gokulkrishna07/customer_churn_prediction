# CI/CD Setup Instructions

## GitHub Actions Workflows

This project includes two workflows:

### 1. CI (Continuous Integration) - `ci.yaml`
- **Triggers**: On pull requests and pushes to `main`
- **Actions**:
  - Runs linting with flake8
  - Runs tests with pytest
  - Validates code quality

### 2. CD (Continuous Deployment) - `cd.yaml`
- **Triggers**: On push to `main` branch
- **Actions**:
  - Connects to VM via SSH
  - Pulls latest code
  - Installs dependencies
  - Runs training with MLflow

## Required GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions → New repository secret

Add the following secrets:

| Secret Name | Description | Example |
|------------|-------------|---------|
| `VM_HOST` | VM IP address | `103.49.125.28` |
| `VM_USERNAME` | SSH username | `ubuntu` or `root` |
| `VM_PASSWORD` | SSH password | `your-password` |
| `VM_PORT` | SSH port (default: 22) | `22` |
| `VM_PROJECT_PATH` | Project path on VM | `/home/ubuntu/customer-churn-prediction` |

## Setup Steps

### 1. Add GitHub Secrets
```
VM_HOST=103.49.125.28
VM_USERNAME=your-username
VM_PASSWORD=your-password
VM_PORT=22
VM_PROJECT_PATH=/path/to/project
```

### 2. Prepare VM
SSH into your VM and run:
```bash
# Clone repository (if not already done)
cd ~
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Push to GitHub
```bash
git add .
git commit -m "Add CI/CD workflows"
git push origin main
```

## Workflow Behavior

- **Pull Request**: CI runs tests and linting
- **Merge to main**: CI runs, then CD deploys and trains model on VM
- **Direct push to main**: CI and CD both run

## Monitoring

- View workflow runs: GitHub repository → Actions tab
- Check MLflow experiments: http://103.49.125.28:8501/mlflow
