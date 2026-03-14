import traci
import matplotlib.pyplot as plt
import csv
import joblib
import pandas as pd
import sys  # Added for standard program termination

# --- LOAD AI BRAIN ---
try:
    model = joblib.load("traffic_model.pkl")
    print("✅ AI traffic model loaded successfully")
except FileNotFoundError:
    print("❌ Error: traffic_model.pkl not found. Run train_model.py first.")
    sys.exit()  # Standard Python termination

# --- SUMO GUI for Demonstrations ---
SUMO_CMD = ["sumo-gui", "-c", "config.sumocfg", "--start"]

def run():
    traci.start(SUMO_CMD)

    step = 0
    speed_history = []
    vehicle_history = []
    
    # Memory for the AI sliding window (t-1, t-2, t-3)
    prev_speed, prev2_speed, prev3_speed = 0, 0, 0
    prev_vehicles, prev2_vehicles, prev3_vehicles = 0, 0, 0

    with open("traffic_data_ai.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "vehicles", "avg_speed", "ai_prediction"])

        while step < 2000 and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # --- Safe Vehicle & Speed Calculation ---
            vehicles_list = traci.vehicle.getIDList()
            vehicle_count = len(vehicles_list) if vehicles_list else 0

            if vehicle_count > 0:
                speeds = [traci.vehicle.getSpeed(vid) for vid in vehicles_list]
                avg_speed = sum(speeds) / vehicle_count
            else:
                avg_speed = 0

            # --- Calculate Features ---
            speed_change = avg_speed - prev_speed
            vehicle_change = vehicle_count - prev_vehicles
            traffic_pressure = vehicle_count / (avg_speed + 0.1)

            # --- Prepare AI Input ---
            features = pd.DataFrame([{
                "vehicles": vehicle_count,
                "avg_speed": avg_speed,
                "speed_change": speed_change,
                "vehicle_change": vehicle_change,
                "prev_speed": prev_speed,
                "prev_vehicles": prev_vehicles,
                "prev2_speed": prev2_speed,
                "prev3_speed": prev3_speed,
                "prev2_vehicles": prev2_vehicles,
                "prev3_vehicles": prev3_vehicles,
                "traffic_pressure": traffic_pressure
            }])

            # --- Protected AI Prediction (Fault Tolerance) ---
            prediction = 0
            if step > 5: 
                try:
                    prediction = model.predict(features)[0]
                except Exception as e:
                    print(f"⚠️ Prediction error at step {step}: {e}")
                    prediction = 0

            # --- STABILIZED AI CONTROL ---
            if prediction == 1 and vehicle_count > 15 and step % 20 == 0:
                print(f"🤖 Step {step}: AI Predicted Congestion. Stabilized 35s Extension.")
                tls_ids = traci.trafficlight.getIDList()
                for tls in tls_ids:
                    traci.trafficlight.setPhaseDuration(tls, 35)

            # --- Logging & Memory Update ---
            writer.writerow([step, vehicle_count, avg_speed, prediction])
            speed_history.append(avg_speed)
            vehicle_history.append(vehicle_count)

            # Shift the memory window
            prev3_speed, prev2_speed, prev_speed = prev2_speed, prev_speed, avg_speed
            prev3_vehicles, prev2_vehicles, prev_vehicles = prev2_vehicles, prev_vehicles, vehicle_count

            # --- Periodic State Logging ---
            if step % 100 == 0:
                print(f"Step {step} | Vehicles: {vehicle_count} | AvgSpeed: {round(avg_speed, 2)}")

            step += 1

    traci.close()

    # --- Clearer Graph Labels ---
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(vehicle_history, color='royalblue', label='Vehicle Count')
    plt.title("Vehicle Count vs Time (AI Traffic Control)")
    plt.ylabel("Vehicles")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(speed_history, color='crimson', label='Avg Speed')
    plt.title("Average Vehicle Speed vs Time")
    plt.xlabel("Simulation Step")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()