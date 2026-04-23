#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Clarix Retail — One-shot Azure resource provisioning script
# Run once: bash azure-setup.sh
# Prerequisites: Azure CLI installed and logged in (az login)
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── Edit these values ─────────────────────────────────────────────────────────
RESOURCE_GROUP="clarix-retail-rg"
LOCATION="eastus"
APP_NAME="clarix-retail-asr-2026"
SQL_SERVER="clarix-sql-asr-2026"
SQL_DB="clarix-retail-db"
SQL_ADMIN="clarixadmin"
SQL_PASSWORD="ClarixRetail2026!"
STORAGE_ACCOUNT="clarixretailasr"
BLOB_CONTAINER="retail-data"
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
# ─────────────────────────────────────────────────────────────────────────────

echo "=== 1. Creating Resource Group ==="
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

echo "=== 2. Creating Storage Account ==="
az storage account create \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Standard_LRS \
  --kind StorageV2

echo "=== 3. Creating Blob Container ==="
az storage container create \
  --name "$BLOB_CONTAINER" \
  --account-name "$STORAGE_ACCOUNT" \
  --public-access off

STORAGE_CONN_STR=$(az storage account show-connection-string \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --query connectionString -o tsv)
echo "Storage connection string captured."

echo "=== 4. Creating Azure SQL Server ==="
az sql server create \
  --name "$SQL_SERVER" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --admin-user "$SQL_ADMIN" \
  --admin-password "$SQL_PASSWORD"

echo "=== 5. Creating Azure SQL Database (free tier / Basic) ==="
az sql db create \
  --resource-group "$RESOURCE_GROUP" \
  --server "$SQL_SERVER" \
  --name "$SQL_DB" \
  --edition Basic \
  --capacity 5

echo "=== 6. Opening SQL Server firewall to Azure services ==="
az sql server firewall-rule create \
  --resource-group "$RESOURCE_GROUP" \
  --server "$SQL_SERVER" \
  --name "AllowAzureServices" \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0

SQL_CONN_STR="Driver={ODBC Driver 18 for SQL Server};Server=tcp:${SQL_SERVER}.database.windows.net,1433;Database=${SQL_DB};Uid=${SQL_ADMIN};Pwd=${SQL_PASSWORD};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
echo "SQL connection string built."

echo "=== 7. Creating App Service Plan (B1 — Free for Students) ==="
az appservice plan create \
  --name "${APP_NAME}-plan" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku B1 \
  --is-linux

echo "=== 8. Creating Web App (Python 3.11) ==="
az webapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --plan "${APP_NAME}-plan" \
  --runtime "PYTHON:3.11"

echo "=== 9. Setting App Service environment variables ==="
az webapp config appsettings set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --settings \
    AZURE_SQL_CONN_STR="$SQL_CONN_STR" \
    AZURE_STORAGE_CONN_STR="$STORAGE_CONN_STR" \
    AZURE_BLOB_CONTAINER="$BLOB_CONTAINER" \
    GEMINI_API_KEY="$GEMINI_API_KEY" \
    SECRET_KEY="clarix-retail-azure-$(openssl rand -hex 16)" \
    SCM_DO_BUILD_DURING_DEPLOYMENT=true

echo "=== 10. Setting startup command ==="
az webapp config set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --startup-file "bash startup.sh"

echo ""
echo "✅ All Azure resources created!"
echo ""
echo "Next step — deploy your code:"
echo "  cd /Users/renunallamotu/Desktop/CloudComputing/final_project"
echo "  zip -r deploy.zip . -x '*.db' -x '__pycache__/*' -x '.git/*' -x 'uploads/*'"
echo "  az webapp deployment source config-zip \\"
echo "    --resource-group $RESOURCE_GROUP --name $APP_NAME --src deploy.zip"
echo ""
echo "Your app will be live at: https://${APP_NAME}.azurewebsites.net"
