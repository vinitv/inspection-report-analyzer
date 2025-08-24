#!/bin/bash

echo "🔍 Testing .env file reading..."

# Check if .env file exists
if [ -f .env ]; then
    echo "✅ .env file found"
    echo "📄 .env file contents:"
    cat .env
    echo ""
else
    echo "❌ .env file not found"
    exit 1
fi

# Test the loading method
echo "🔄 Testing environment variable loading..."

# Load variables using the same method as deploy.sh
while IFS= read -r line; do
    # Skip empty lines and comments
    if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
        # Export the variable
        export "$line"
        echo "   Loaded: ${line%%=*}"
    fi
done < .env

echo ""
echo "🔍 Checking loaded variables:"

# Check each variable
for var in OPENAI_API_KEY REPAIR_API_KEY LANGSMITH_API_KEY; do
    value="${!var}"
    if [ -n "$value" ]; then
        echo "✅ $var: ${value:0:8}... (length: ${#value})"
    else
        echo "❌ $var: Not set"
    fi
done

echo ""
echo "🎯 Test complete!"
