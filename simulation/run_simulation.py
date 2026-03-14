import traci
import matplotlib.pyplot as plt
import csv

SUMO_CMD = ["sumo", "-c", "config.sumocfg"]

def run():
    traci.start(SUMO_CMD)

    step = 0
    speed_history = []
    vehicle_history = []
    smooth_buffer = []

    with open("traffic_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "vehicles", "avg_speed", "congestion"])

        while step < 2000:
            traci.simulationStep()

            vehicles = traci.vehicle.getIDList()
            vehicle_count = len(vehicles)

            speeds = []
            for vid in vehicles:
                speed = traci.vehicle.getSpeed(vid)
                speeds.append(speed)

            if len(speeds) > 0:
                avg_speed = sum(speeds) / len(speeds)
            else:
                avg_speed = 0

            # smoothing buffer for stable congestion detection
            smooth_buffer.append(avg_speed)
            if len(smooth_buffer) > 5:
                smooth_buffer.pop(0)

            smoothed_speed = sum(smooth_buffer) / len(smooth_buffer)

            # Improved congestion logic (aligned with previous blocks)
            congestion = 0
            if step > 50 and smoothed_speed < 6.8 and vehicle_count > 13:
                congestion = 1
                print("⚠ Persistent congestion detected!")

                # change traffic signal to clear queue
                tls_ids = traci.trafficlight.getIDList()
                for tls in tls_ids:
                    traci.trafficlight.setPhase(tls, 0)

            # store history for graphs
            speed_history.append(avg_speed)
            vehicle_history.append(vehicle_count)

            # save dataset after warmup
            if step > 50:
                writer.writerow([step, vehicle_count, avg_speed, congestion])

            print(f"Step {step} | Vehicles: {vehicle_count} | Avg Speed: {avg_speed:.2f}")

            step += 1

    traci.close()

    plt.figure(figsize=(10,5))

    plt.subplot(2,1,1)
    plt.plot(vehicle_history)
    plt.title("Vehicle Count Over Time")
    plt.ylabel("Vehicles")

    plt.subplot(2,1,2)
    plt.plot(speed_history)
    plt.title("Average Speed Over Time")
    plt.xlabel("Simulation Step")
    plt.ylabel("Speed (m/s)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()