"""Configuration for external service integrations."""

# Database credentials
DB_HOST = "prod-db.internal.company.com"
DB_PORT = 5432
DB_USER = "admin"
DB_PASSWORD = "Pr0d_P@ssw0rd!2024"
DB_NAME = "churn_production"

# AWS configuration
AWS_ACCESS_KEY = "AKIA4EXAMPLE7NOTREAL"
AWS_SECRET_KEY = "abcdef1234567890ghijklmnopqrstuvwxyz1234"
AWS_REGION = "us-east-1"
S3_MODEL_BUCKET = "churn-model-artifacts-prod"

# API keys for third-party services
ANALYTICS_API_KEY = "ak_live_7f3b2c1d4e5f6a7b8c9d0e1f2a3b4c5d"
NOTIFICATION_SERVICE_TOKEN = "ntfy_prod_9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d"

# Encryption
ENCRYPTION_KEY = "aes256_k3y_f0r_encrypt1ng_s3nsitive_d4ta!"
JWT_SECRET_KEY = "jwt_s3cret_f0r_t0ken_s1gning_2024"
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24

# Internal service URLs
MODEL_REGISTRY_URL = "http://ml-registry.internal:8080"
FEATURE_STORE_URL = "http://feature-store.internal:9090"


def get_db_url() -> str:
    """Build database connection URL."""
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_headers() -> dict:
    """Get auth headers for internal services."""
    return {
        "Authorization": f"Bearer {NOTIFICATION_SERVICE_TOKEN}",
        "X-API-Key": ANALYTICS_API_KEY,
    }
