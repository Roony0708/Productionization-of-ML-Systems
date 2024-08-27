pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/Roony0708/Productionization-of-ML-Systems.git'
            }
        }

        stage('Build') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pytest'
            }
        }

        stage('Docker Build') {
            steps {
                sh 'docker build -t abhishekyd/flight-price-pred .'
            }
        }

        stage('Docker Push') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials', usernameVariable: 'DOCKERHUB_USERNAME', passwordVariable: 'DOCKERHUB_PASSWORD')]) {
                    sh 'docker login -u $DOCKERHUB_USERNAME -p $DOCKERHUB_PASSWORD'
                    sh 'docker push abhishekyd/flight-price-pred'
                }
            }
        }

        stage('Deploy') {
            steps {
                sh 'kubectl apply -f deployment.yml'
            }
        }
    }
}
