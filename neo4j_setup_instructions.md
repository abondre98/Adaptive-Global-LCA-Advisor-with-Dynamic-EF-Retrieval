   # Neo4j Setup Instructions

## Prerequisites

Before setting up Neo4j for the Emission Factor Knowledge Graph, you need to have Docker properly installed and running on your system.

## Step 1: Install Docker Desktop

### For macOS:

1. Download Docker Desktop for Mac from [Docker Hub](https://hub.docker.com/editions/community/docker-ce-desktop-mac)
2. Double-click the downloaded .dmg file and drag Docker to your Applications folder
3. Open Docker from your Applications folder
4. Wait for Docker to start (the whale icon in the status bar will stop animating when Docker is ready)
5. Verify installation by running:
   ```bash
   docker --version
   docker run hello-world
   ```

### For Windows:

1. Download Docker Desktop for Windows from [Docker Hub](https://hub.docker.com/editions/community/docker-ce-desktop-windows)
2. Run the installer and follow the prompts
3. Start Docker Desktop from the Start menu
4. Wait for Docker to start
5. Verify installation by running:
   ```bash
   docker --version
   docker run hello-world
   ```

### For Linux:

1. Update your package index:

   ```bash
   sudo apt-get update
   ```

2. Install prerequisites:

   ```bash
   sudo apt-get install \
      apt-transport-https \
      ca-certificates \
      curl \
      gnupg \
      lsb-release
   ```

3. Add Docker's official GPG key:

   ```bash
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
   ```

4. Set up the stable repository:

   ```bash
   echo \
      "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

5. Install Docker Engine:

   ```bash
   sudo apt-get update
   sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
   ```

6. Verify installation:

   ```bash
   sudo docker run hello-world
   ```

7. (Optional) Add your user to the docker group to run Docker without sudo:
   ```bash
   sudo usermod -aG docker $USER
   ```
   Then log out and log back in for the changes to take effect.

## Step 2: Set Up Neo4j Using Docker

Once Docker is installed and running, follow these steps to set up Neo4j:

1. Make sure all the necessary files are in place:

   - `docker-compose.yml`
   - `neo4j/import/schema.cypher`
   - `scripts/prepare_neo4j_data.py`
   - `scripts/import_neo4j_data.sh`

2. Create the required directories:

   ```bash
   mkdir -p neo4j/data neo4j/logs neo4j/import neo4j/plugins
   ```

3. Download Neo4j plugins:

   ```bash
   curl -L https://github.com/neo4j/apoc/releases/download/5.11.0/apoc-5.11.0-core.jar -o neo4j/plugins/apoc-5.11.0-core.jar
   curl -L https://github.com/neo4j/graph-data-science/releases/download/2.4.0/neo4j-graph-data-science-2.4.0.jar -o neo4j/plugins/neo4j-graph-data-science-2.4.0.jar
   ```

4. Make the scripts executable:

   ```bash
   chmod +x scripts/prepare_neo4j_data.py scripts/import_neo4j_data.sh
   ```

5. Start Neo4j using Docker Compose:

   ```bash
   docker compose up -d
   ```

6. Verify the Neo4j container is running:
   ```bash
   docker ps
   ```
   You should see the `neo4j-ef-kg` container in the list.

## Step 3: Prepare and Import Data

1. Run the data preparation script:

   ```bash
   python scripts/prepare_neo4j_data.py
   ```

   This will convert the harmonized dataset to Neo4j-friendly CSV files.

2. Run the import script:
   ```bash
   bash scripts/import_neo4j_data.sh
   ```
   This will import all the data into Neo4j and create the knowledge graph structure.

## Step 4: Access Neo4j Browser

1. Open your web browser and navigate to:

   ```
   http://localhost:7474
   ```

2. Log in with the following credentials:

   - Username: `neo4j`
   - Password: `complex-password-here` (the one specified in docker-compose.yml)

3. Run a test query to verify the setup:
   ```cypher
   MATCH (n) RETURN count(n)
   ```

## Step 5: Explore the Knowledge Graph

Try some of the sample queries from `neo4j/sample_queries.cypher` to explore the knowledge graph. For example:

```cypher
// Count all nodes by type
MATCH (n)
RETURN labels(n) AS NodeType, count(n) AS Count
ORDER BY Count DESC;

// Regional comparison for energy emission factors
MATCH (ef:EmissionFactor)-[:HAS_ENTITY_TYPE]->(et:EntityType {type_name: 'energy'})
MATCH (ef)-[:APPLIES_TO_REGION]->(r:Region)
WHERE r.region_code IN ['USA', 'FR', 'GLB']
RETURN r.name AS Region, avg(ef.ef_value) AS AvgEmissionFactor, count(ef) AS Count
ORDER BY AvgEmissionFactor DESC;
```

## Troubleshooting

### Common Issues

1. **Docker container won't start**:

   - Check Docker is running: `docker info`
   - Check for port conflicts: `lsof -i :7474` and `lsof -i :7687`
   - Check logs: `docker logs neo4j-ef-kg`

2. **Data import fails**:

   - Check CSV file format and column names
   - Verify file paths are correct
   - Check Neo4j logs: `docker logs neo4j-ef-kg`

3. **Cannot connect to Neo4j Browser**:

   - Verify the container is running: `docker ps`
   - Check the port mapping: `docker port neo4j-ef-kg`
   - Try using a different browser

4. **Out of memory errors**:
   - Adjust memory settings in docker-compose.yml
   - Increase Docker's resource allocation in Docker Desktop settings

### Getting Help

If you encounter issues not covered in this guide:

1. Check the Neo4j documentation: https://neo4j.com/docs/
2. Docker documentation: https://docs.docker.com/
3. Neo4j Community forums: https://community.neo4j.com/
