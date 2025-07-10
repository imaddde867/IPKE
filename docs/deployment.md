# Deployment and Maintenance Guide

This guide provides instructions for deploying and maintaining the Industrial Knowledge Extraction System using Docker and Docker Compose.

## Prerequisites

- **Docker Desktop:** Ensure Docker Desktop is installed and running on your system. This includes Docker Engine and Docker Compose.

## Deployment Steps

1.  **Clone the Repository:**
    ```bash
    git clone ..
    cd explainium-2.0
    ```

2.  **Build Docker Images:**
    Navigate to the root directory of the project (where `docker-compose.yml` is located) and build the Docker images. This step downloads all necessary dependencies and sets up the application environment.
    ```bash
    docker-compose build
    ```
    *Note: If you encounter issues or want to ensure a fresh build, you can use `docker-compose build --no-cache`.*

3.  **Start Services:**
    Start all the services defined in `docker-compose.yml` in detached mode. This will bring up the FastAPI application, Celery worker, Redis, PostgreSQL, Elasticsearch, and Apache Tika.
    ```bash
    docker-compose up -d
    ```

4.  **Verify Services (Optional but Recommended):**
    Check the status of your running Docker containers:
    ```bash
    docker-compose ps
    ```
    You should see `Up` status for `app`, `celery_worker`, `redis`, `db`, `tika`, and `elasticsearch` services.

5.  **Access the Application:**
    - **FastAPI Backend:** The API will be accessible at `http://localhost:8000`.
    - **Web Interface (React Frontend):**
        1.  Navigate to the frontend directory:
            ```bash
            cd src/frontend
            ```
        2.  Start the React development server:
            ```bash
            npm start
            ```
            This will typically open your browser to `http://localhost:3000`.

## Maintenance and Troubleshooting

### Stopping Services

To stop all running services:

```bash
docker-compose stop
```

### Stopping and Removing Services (Clean Up)

To stop and remove all containers, networks, and volumes associated with the project (useful for a clean restart or when making schema changes):

```bash
docker-compose down -v
```

### Viewing Logs

To view the logs of all services:

```bash
docker-compose logs -f
```

To view logs for a specific service (e.g., `app`):

```bash
docker-compose logs -f app
```

### Database Migrations

For schema changes in a production environment, you would typically use a tool like Alembic for database migrations. For this project, `Base.metadata.create_all(bind=engine)` is used, which creates tables if they don't exist. **Be aware that `docker-compose down -v` will remove all data volumes, effectively resetting the database.**

### AI Model Downloads

Initial runs of the Celery worker might involve downloading large AI models (e.g., Hugging Face models, Whisper models). This can take time and consume bandwidth. Ensure stable internet connectivity during the first build and run.

### Common Issues

- **"Cannot connect to the Docker daemon"**: Ensure Docker Desktop is running and fully initialized.
- **Port Conflicts**: If `localhost:8000`, `localhost:3000`, `localhost:5432`, `localhost:6379`, `localhost:9998`, `localhost:9200`, or `localhost:9300` are already in use, you will need to free them up or modify the `ports` mapping in `docker-compose.yml`.
- **Memory/CPU Issues**: Running all services, especially AI models, can be resource-intensive. Ensure your system has sufficient RAM and CPU resources allocated to Docker.
