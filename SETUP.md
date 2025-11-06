# Setup Instructions

This guide will help you set up the application on your local machine.

## Prerequisites

- Git installed on your system
- Python 3.8 or higher (if this is a Python application)
- Access to your AskSage account and API token

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

Replace `<repository-url>` with the actual URL of this repository.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure AskSage Token

Your AskSage API token should be stored securely and never committed to version control.

#### Option A: Environment Variables (Recommended)

Create a `.env` file in the project root:

```bash
touch .env
```

Add your AskSage token to the `.env` file:

```
ASKSAGE_TOKEN=your_token_here
```

**Important:** Ensure `.env` is listed in your `.gitignore` file to prevent accidental commits.

#### Option B: Secrets Management

If using a secrets manager (e.g., AWS Secrets Manager, Azure Key Vault, HashiCorp Vault), configure your token according to your platform's documentation.

#### Option C: Configuration File

Create a `secrets.json` file (ensure it's in `.gitignore`):

```json
{
  "asksage_token": "your_token_here"
}
```

### 4. Verify Setup

Run the application to verify everything is configured correctly:

```bash
python src/app.py
```

## Security Best Practices

- **Never commit secrets** to version control
- **Rotate tokens regularly** for enhanced security
- **Use environment-specific tokens** (separate tokens for development, staging, and production)
- **Limit token permissions** to only what's necessary for the application

## Getting Your AskSage Token

1. Log in to your AskSage account
2. Navigate to Settings → API Tokens
3. Generate a new token or copy your existing token
4. Store it securely using one of the methods above

## Troubleshooting

If you encounter issues:

- Verify your token is correctly formatted and has no extra whitespace
- Check that your `.env` file is in the project root directory
- Ensure all dependencies are installed correctly
- Review application logs for specific error messages

## Next Steps

Once setup is complete, refer to the main [README.md](README.md) for usage instructions and API documentation.