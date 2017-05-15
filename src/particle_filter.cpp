/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <map>

#include "particle_filter.h"
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    
    // set number of particles
    num_particles = 10;
    
    // define noise generator
    default_random_engine gen;
    normal_distribution<double> N_particle_x(x, std[0]);
    normal_distribution<double> N_particle_y(y, std[1]);
    normal_distribution<double> N_particle_theta(theta, std[2]);
    
    // loop to initialize each particle around the given position with random noise
    Particle pt;
    weights = vector<double>(num_particles);
    for (int i = 0; i < num_particles; ++i) {
        pt.id = i;
        // set x, y, theta with gaussian noise
        pt.x = N_particle_x(gen);
        pt.y = N_particle_y(gen);
        pt.theta = N_particle_theta(gen);
        // set particle weight
        pt.weight = 1.0;
        weights[i] = 1.0;
        // add to the particle list
        particles.push_back(pt);
        cout << "Particle " << i << " Init: " << pt.x << "," << pt.y << "," << pt.theta << endl;
    }
    
    // set is_initialized to true
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

    // define noise generator
    default_random_engine gen;
    normal_distribution<double> N_particle_x(0, std_pos[0]);
    normal_distribution<double> N_particle_y(0, std_pos[1]);
    normal_distribution<double> N_particle_theta(0, std_pos[2]);
    
    // loop to do motion prediction for each particle
    for (int i = 0; i < particles.size(); ++i) {
        // when yawrate is zero
        if(fabs(yaw_rate) <= 0.0001){
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }// when yawrate is not zero
        else{
            particles[i].x += velocity/yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
            particles[i].theta += yaw_rate * delta_t;
        }
        // add noisy to x, y and theta
        particles[i].x += N_particle_x(gen);
        particles[i].y += N_particle_y(gen);
        particles[i].theta += N_particle_theta(gen);
        //cout << "Particle " << i << " Predict: " << particles[i].x << "," << particles[i].y << "," << particles[i].theta << endl;
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted_landmarks, vector<LandmarkObs>& observations) {
    
    LandmarkObs landmark;
    for (auto& obs:observations) {
        // initialize the min distance
        double dist_min = numeric_limits<double>::max();
        double dist_tmp = 0;
        // loop on each predicted landmards
        for (int i = 0; i < predicted_landmarks.size(); ++i) {
            landmark = predicted_landmarks[i];
            dist_tmp = dist(obs.x, obs.y, landmark.x, landmark.y);
            if (dist_tmp < dist_min) {
                obs.id = landmark.id;
                dist_min = dist_tmp;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   vector<LandmarkObs> observations, Map map_landmarks) {
	
    // loop to update the weight for each particle
    for (int i = 0; i < particles.size(); ++i) {
        
        // transform the observations' local coordinates to map coordinates
        vector<LandmarkObs> ts_observations;
        LandmarkObs ts_obs;
        for (int j = 0; j < observations.size(); ++j){
            ts_obs = observations[j];
            double x = particles[i].x + ts_obs.x * cos(particles[i].theta) - ts_obs.y * sin(particles[i].theta);
            double y = particles[i].y + ts_obs.x * sin(particles[i].theta) + ts_obs.y * cos(particles[i].theta);
            ts_observations.push_back(LandmarkObs{ts_obs.id, x, y});
        }
        
        // get predicted map landmarks that are within the sensor range
        vector<LandmarkObs> predicted_landmarks;
        map<int, Map::single_landmark_s> mapping_landmark;
        Map::single_landmark_s m_landmark;
        for (int j = 0; j < map_landmarks.landmark_list.size(); ++j){
            // get current map landmark
            m_landmark = map_landmarks.landmark_list[j];
            // compute the distance to particle and compare to the sensor range
            if(dist(m_landmark.x_f, m_landmark.y_f, particles[i].x, particles[i].y) <= sensor_range){
                predicted_landmarks.push_back(LandmarkObs{m_landmark.id_i, m_landmark.x_f, m_landmark.y_f});
                mapping_landmark.insert(make_pair(m_landmark.id_i, m_landmark));
            }
        }
        
        // find the nearest map landmark to the each observation by data association
        dataAssociation(predicted_landmarks, ts_observations);
        
        // loop on each observation
        double final_weight = 1;
        double compo_x = 0;
        double compo_y = 0;
        Map::single_landmark_s landmark;
        for(int j = 0; j < ts_observations.size(); ++j){
            // get corresponding landmark
            landmark = mapping_landmark[ts_observations[j].id];
            // calculate the Multivariate-Gaussian probability
            compo_x = (ts_observations[j].x - landmark.x_f) / std_landmark[0];
            compo_y = (ts_observations[j].y - landmark.y_f) / std_landmark[1];
            final_weight *= 1/(2 * M_PI * std_landmark[0] * std_landmark[1]) * exp(-0.5 * (compo_x * compo_x + compo_y * compo_y));
            //cout << "Obs " << j << " [x, y]=[" << ts_observations[j].x << "," << ts_observations[j].y << "] Landmark " << ts_observations[j].id << " [x, y]=[" << landmark.x_f << "," << landmark.y_f << "] Weight:" << final_weight <<  endl;
        }
        
        // update the weight
        particles[i].weight = final_weight;
        weights[i] = final_weight;
    }
}

void ParticleFilter::resample() {

    // define the resampled particles list
    int size = particles.size();
    vector<Particle> resampled_particles;
    resampled_particles.resize(size);
    
    // define random generator
    default_random_engine gen;
    // resample by using the discrete distribution
    discrete_distribution<> d(weights.begin(), weights.end());
    for(int i=0; i<size; ++i) {
        resampled_particles[i] = particles[d(gen)];
    }
    
    // update the particles list
    particles = resampled_particles;
}

void ParticleFilter::write(string filename) {
	// You don't need to modify this file.
	ofstream dataFile;
	dataFile.open(filename, ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
