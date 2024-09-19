# Loan Prediction API

This project is a FastAPI-based API for predicting loan approvals. It uses a machine learning model to make predictions based on input data.

## Prerequisites

- Docker installed on your machine
- Git (optional, for cloning the repository)

## Getting Started

### 1. Clone the Repository (if applicable)

If the project is in a Git repository, clone it:

```bash
git clone https://github.com/your-username/loan-prediction-api.git
cd loan-prediction-api
```

### 2. Build the Docker Image

From the root directory of the project (where the Dockerfile is located), run:

```bash
docker build -t loan-prediction-api .
```

This command builds a Docker image and tags it as `loan-prediction-api`.

### 3. Run the Docker Container

After the image is built, run the container:

```bash
docker run -p 8000:8000 loan-prediction-api
```

This command:
- Starts a container from the `loan-prediction-api` image
- Maps port 8000 of the container to port 8000 on your host machine

The API should now be running and accessible at `http://localhost:8000`.

## Using the API

### API Documentation

Once the container is running, you can view the interactive API documentation by opening a web browser and navigating to:

```
http://localhost:8000/docs
```

This Swagger UI provides a detailed overview of all available endpoints and allows you to try out the API directly from the browser.

### Making Predictions

#### Single Prediction

To make a single prediction, send a POST request to `/predict` with the required input data.

#### Bulk Prediction

For bulk predictions, use the `/predict_bulk` endpoint with a list of input data.

### Retrieving Scores

- To list all available score files: GET `/scores`
- To retrieve a specific score file: GET `/scores/{filename}`

## Stopping the Container

To stop the running container:

1. Find the container ID:
   ```bash
   docker ps
   ```
2. Stop the container:
   ```bash
   docker stop <container_id>
   ```

## Rebuilding the Image

If you make changes to the API code, rebuild the Docker image:

```bash
docker build -t loan-prediction-api .
```

Then run the container again as described in step 3.

## Troubleshooting

If you encounter any issues:

1. Ensure Docker is running on your machine.
2. Check if the required ports are not in use by other applications.
3. Verify that all necessary files (including the model and preprocessor) are present in the correct locations within the project directory.

For any persistent issues, please open an issue in the project repository or contact the maintainer.
