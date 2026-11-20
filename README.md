# SCSE3040 MLOps Mini Project

## Group Information

- **Group:** G186
- **Problem ID:** MP089
- **Problem:** Bakery Bread Sales (Nagpur)
- **Course:** SCSE3040 MLOps

## Group Members

| Name | Enrolment Number |
|---|---|
| Samarth | S24CSEU2311 |
| Bhumika Sharma | S24CSEU2310 |
| Anany | S24CSEU2309 |

## Project Overview

This project focuses on developing an end-to-end MLOps pipeline for predicting daily bakery bread sales in Nagpur. The pipeline includes data auditing and cleaning, data version control, model training and experiment tracking, model deployment through an API, testing, containerization, monitoring, and model retraining.

## Objective

To predict the number of bread loaves sold daily (`loaves_sold`) using the available business and environmental features, while building a reproducible and deployable MLOps pipeline.

## Dataset

The project uses the dataset provided for problem MP089:

- `train.csv`
- `later.csv`
- `holdout_features.csv`

## Project Pipeline

- Data auditing and preprocessing
- Data version control using DVC
- Model training and experiment tracking using MLflow
- Model evaluation using MAE
- Prediction service using FastAPI
- Automated testing using pytest
- Containerization using Docker
- Deployment configuration using Kubernetes
- Data drift monitoring and model retraining
  

## Repository Structure

The repository contains source code, data version-control files, model artifacts, tests, configuration files, deployment files, and prediction outputs as the project progresses.

## Problem

**MP089 – Bakery Bread Sales (Nagpur)**
y_type`, `outlet`. The schema rejects `orders_previous_day > 850` and `price_rs < 10`.
