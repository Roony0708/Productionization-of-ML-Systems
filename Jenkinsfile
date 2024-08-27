pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                // Clone the Git repository
                git 'https://github.com/Roony0708/Productionization-of-ML-Systems.git'
            }
        }

        stage('Build') {
            steps {
                // Install dependencies and run unit tests (for Windows)
                bat 'pip install -r requirements.txt'
                bat 'pytest'
            }
        }

        stage('Docker Build') {
            steps {
                // Build Docker image (Docker CLI on Windows)
                bat 'docker build -t abhishekyd/flight-price-pred .'
            }
        }

        stage('Docker Push') {
            steps {
                // Push Docker image to DockerHub
                withCredentials([string(credentialsId: 'dockerhub-credentials', variable: 'DOCKERHUB_PASSWORD')]) {
                    bat 'docker login -u yourusername -p %DOCKERHUB_PASSWORD%'
                    bat 'docker push abhishekyd/flight-price-pred'
                }
            }
        }

        stage('Deploy') {
            steps {
                // Deploy to Kubernetes or any server (assuming kubectl is configured)
                bat 'kubectl apply -f deployment.yml'
            }
        }
    }
    
    post {
        always {
            // Send notifications or cleanup actions
            // Example: Clean up Docker images
        }
    }
}
