#!/bin/bash

# Wait for Neo4j to be ready
echo "Waiting for Neo4j to be ready..."
sleep 10

# Execute the schema setup script
/Applications/Docker.app/Contents/Resources/bin/docker exec neo4j-ef-kg cypher-shell -u neo4j -p complex-password-here -f /var/lib/neo4j/import/setup_schema.cypher

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Schema setup completed successfully"
else
    echo "Error setting up schema"
    exit 1
fi 
