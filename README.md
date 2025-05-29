# Robustness Testing for Autonomous Driving in Urban Environments

This repository accompanies the ASE 2025 submission:
**"Robustness Testing for Autonomous Driving Systems in Urban Environments: A Regional Scenario-Based Approach"**.

## 🚘 Overview

This project proposes a regional-level scenario testing framework designed to evaluate the adaptability of Autonomous Driving Systems (ADS) in dynamic urban environments. It combines data-driven traffic flow prediction, realistic motorcycle modeling, and a mutation-based scenario testing methodology to simulate real-world urban conditions.

## 🧠 Key Components

### 1. T-DDSTGCN: Traffic Flow Prediction
- **Model**: Turning-Dual Dynamic Spatial-Temporal Graph Convolutional Network
- **Purpose**: Predict traffic speeds and turning probabilities at intersections
- **Code**: See `src/`
- **How to Use**: train files for DDSTGCN: python train.py; running Turning-DDSTGCN: the two demo files; more details can be seen in Appendix

### 2. Scenario Reconstruction
- Uses OSM maps and predicted traffic data to build realistic urban scenes.
- Scene configuration and road network processing scripts in `maps/`

### 3. POP: Motorcycle Behavior Modeling
- Based on **Level-K Game Theory** and **Social Value Orientation (SVO)**
- Simulates complex two-wheeled participant behavior

### 4. Mutation Testing
- Diversifies testing scenarios with changes in traffic density, weather, and agent behavior.

## 🧪 Datasets

- [METR-LA] and [PEMS-BAY]
- City-level traffic and OSM map data for Los Angeles and San Francisco

## 🧰 Simulation Platforms

- [PanoSim]
- [Apollo Simulator]
- [Oasis Sim]

## 📊 Evaluation Highlights

- 180 reconstructed urban test scenarios
- 775 total collisions; 662 valid cases (88.1% match real-world accidents)
- Improved robustness and accuracy in traffic flow and turning prediction

