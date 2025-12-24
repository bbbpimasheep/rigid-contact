// main.cpp
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <iomanip>
#include <sstream>
#include <filesystem> // C++17 standard

#include "simulation.h"
#include "type.h"

// Configuration parameters
const std::string MESH_PATH   = "assets/bunny.obj"; // Ensure a valid obj file exists at this path
const std::string OUTPUT_DIR  = "output";
const int   NUM_OBJECTS       = 50;                // Number of objects to spawn
const int   TOTAL_FRAMES      = 300;               // Total frames to simulate
const float OBJECT_MASS       = 1.0f;

// Scene boundary settings
const Vec3 SCENE_MIN(-1.8f, -1.8f,  0.0f);          
const Vec3 SCENE_MAX( 1.8f,  1.8f,  3.0f);

// Helper function: Generate random float
float random_float(float min, float max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  return dis(gen);
}

// Helper function: Generate random quaternion (for random rotation)
Quat random_quat() {
  float u1 = random_float(0.0f, 1.0f);
  float u2 = random_float(0.0f, 1.0f);
  float u3 = random_float(0.0f, 1.0f);

  float sqrt1_u1 = std::sqrt(1.0f - u1);
  float sqrt_u1  = std::sqrt(u1);

  const float PI = 3.14159265359f;
  return Quat(
    sqrt1_u1 * std::sin(2.0f * PI * u2),
    sqrt1_u1 * std::cos(2.0f * PI * u2),
    sqrt_u1  * std::sin(2.0f * PI * u3),
    sqrt_u1  * std::cos(2.0f * PI * u3)
  ).normalized();
}

int main() {
  // Prepare output directory
  // namespace fs = std::filesystem;
  // if (!fs::exists(OUTPUT_DIR)) {
  //   if(!fs::create_directory(OUTPUT_DIR)) {
  //     std::cerr << "Error: Could not create output directory: " << OUTPUT_DIR << std::endl;
  //     return -1;
  //   }
  // }

  // Initialize simulation
  std::cout << "[SIM_INFO] Initializing Simulation..." << std::endl;
  Simulation sim;
  sim.initialize();
  sim.set_environment(SCENE_MIN, SCENE_MAX);   // Set environment boundaries (closed box space)
  sim.set_iterations(20);                      // Set solver parameters
  sim.set_tolerance(1e-3f);

  // Spawn objects randomly
  std::cout << "[SIM_INFO] Spawning " << NUM_OBJECTS << " objects from: " << MESH_PATH << std::endl;
  for (int i = 0; i < NUM_OBJECTS; ++i) {
    std::string name = "body_" + std::to_string(i);
    // Add rigid body (automatically loads mesh and calculates properties)
    RigidBody& body = sim.add_body(name, MESH_PATH, OBJECT_MASS);
    if (!body.has_mesh()) {
        std::cerr << "[SIM_INFO] Error: Failed to load mesh for body " << i << ". Check file path." << std::endl;
        return -1;
    }

    BodyState initial_state;
    // Random position within boundaries, but spaced out in height to avoid initial overlap
    float pad = 0.3f;
    float x = random_float(SCENE_MIN.x() + pad, SCENE_MAX.x() - pad);
    float y = random_float(SCENE_MIN.y() + pad, SCENE_MAX.y() - pad);
    float z = random_float(2.0f, SCENE_MAX.z() - pad); // Random distribution starting from height 3.0
    
    initial_state.position     = Vec3(x, y, z);
    initial_state.orientation  = random_quat();
    initial_state.lin_velocity = Vec3(random_float(-1,1), 
                                      random_float(-1,1), 
                                      random_float(-1,1));  // Give a tiny initial random velocity to add chaos
    body.set_state(initial_state);
  }

  // Simulation loop
  std::cout << "[SIM_INFO] Starting simulation loop for " << TOTAL_FRAMES << " frames..." << std::endl;
  for (int frame = 0; frame < TOTAL_FRAMES; ++frame) {
    // Export current frame
    std::stringstream ss;
    ss << OUTPUT_DIR << "/frame_" << std::setw(4) << std::setfill('0') << frame << ".obj";
    std::string filename = ss.str();

    sim.export_frame(filename);
    sim.step(); // Execute physics step

    // Log: Print progress
    if (frame % 10 == 0)
      std::cout << "[SIM_INFO] Frame " << frame << " / " << TOTAL_FRAMES << " simulated." << std::endl;
  }

  std::cout << "[SIM_INFO] Simulation finished. Output saved to " << OUTPUT_DIR << "/" << std::endl;
  return 0;
}