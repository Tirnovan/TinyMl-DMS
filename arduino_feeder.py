import serial
import pandas as pd
import time
import re
from pathlib import Path

class ArduinoPredictor:
    def __init__(self, port, baudrate=115200, timeout=5):
        """
        Initialize serial connection to Arduino.
        
        Args:
            port: Serial port 
            baudrate: Communication speed (default 115200)
            timeout: Read timeout in seconds
        """
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # Wait for Arduino to reset
        
        # Read initial messages from Arduino
        print("Waiting for Arduino to initialize...")
        time.sleep(1)
        while self.ser.in_waiting:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"Arduino: {line}")
        print("Arduino ready!\n")
    
    def send_sensor_data(self, sensor_values):
        """
        Send sensor data to Arduino and receive prediction.
        
        Args:
            sensor_values: List or array of 16 sensor values
            
        Returns:
            tuple: (predicted_x, predicted_y, inference_time_us) or (None, None, None) if failed
        """
        # Format as CSV string
        csv_string = ','.join([str(val) for val in sensor_values])
        
        # Send to Arduino
        self.ser.write(f"{csv_string}\n".encode('utf-8'))
        time.sleep(0.1)  # Brief pause for Arduino to process
        
        # Read response
        predicted_x = None
        predicted_y = None
        inference_time_us = None
        
        # Read all available lines
        timeout_start = time.time()
        while time.time() - timeout_start < 3:  # 3 second timeout
            if self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"  {line}")
                
                # Parse prediction results
                if "Predicted X:" in line:
                    match = re.search(r'Predicted X:\s*([-+]?\d*\.?\d+)', line)
                    if match:
                        predicted_x = float(match.group(1))
                
                if "Predicted Y:" in line:
                    match = re.search(r'Predicted Y:\s*([-+]?\d*\.?\d+)', line)
                    if match:
                        predicted_y = float(match.group(1))
                
                if "Inference time:" in line:
                    # Try to match microseconds format: "Inference time: 15234 μs (15.23 ms)"
                    match_us = re.search(r'Inference time:\s*(\d+)\s*[μu]s', line)
                    if match_us:
                        inference_time_us = int(match_us.group(1))
                    else:
                        # Try to match milliseconds format: "Inference time: 15.23 ms"
                        match_ms = re.search(r'Inference time:\s*([-+]?\d*\.?\d+)\s*ms', line)
                        if match_ms:
                            inference_time_us = int(float(match_ms.group(1)) * 1000)
                
                # Check if we've received both predictions
                if (predicted_x is not None and 
                   predicted_y is not None and
                   inference_time_us is not None):
                    break
            else:
                time.sleep(0.1)
        
        return predicted_x, predicted_y, inference_time_us
    
    def close(self):
        """Close serial connection."""
        self.ser.close()


def process_csv_file(csv_path, arduino_port, output_path='predictions_results.csv'):
    """
    Read CSV file, send sensor data to Arduino, and save predictions.
    
    Args:
        csv_path: Path to input CSV file
        arduino_port: Serial port for Arduino
        output_path: Path to save results
    """
    # Read CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Verify required columns exist
    sensor_cols = [f'sensor_{i:02d}' for i in range(16)]
    required_cols = sensor_cols + ['true_x', 'true_y']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Found {len(df)} samples in CSV file")
    print(f"Sensor columns: {sensor_cols}")
    print()
    
    # Initialize Arduino connection
    print(f"Connecting to Arduino on {arduino_port}...")
    arduino = ArduinoPredictor(arduino_port)
    
    # Prepare results storage
    results = []
    
    try:
        # Process each row
        for idx, row in df.iterrows():
            print(f"\n{'='*60}")
            print(f"Processing sample {idx + 1}/{len(df)} (ID: {row.get('sample_id', 'N/A')})")
            print(f"{'='*60}")
            
            # Extract sensor values
            sensor_values = [row[col] for col in sensor_cols]
            true_x = row['true_x']
            true_y = row['true_y']
            
            print(f"True position: X={true_x:.6f}, Y={true_y:.6f}")
            print(f"Sending sensor data to Arduino...")
            
            # Get prediction from Arduino
            pred_x, pred_y, inference_time = arduino.send_sensor_data(sensor_values)
            
            # Store results
            result = {
                'sample_id': row.get('sample_id', idx),
                'true_x': true_x,
                'true_y': true_y,
                'predicted_x': pred_x,
                'predicted_y': pred_y,
                'inference_time_us': inference_time,
                'inference_time_ms': inference_time / 1000.0 if inference_time is not None else None,
                'success': pred_x is not None and pred_y is not None
            }
            
            # Add sensor values to results
            for i, col in enumerate(sensor_cols):
                result[col] = sensor_values[i]
            
            results.append(result)
            
            if pred_x is not None and pred_y is not None:
                error_x = abs(pred_x - true_x)
                error_y = abs(pred_y - true_y)
                print(f"\n✓ Prediction received!")
                print(f"  Error X: {error_x:.6f}")
                print(f"  Error Y: {error_y:.6f}")
            else:
                print(f"\n✗ Failed to get prediction")
            
            # Small delay between samples
            time.sleep(0.5)
        
    finally:
        # Close Arduino connection
        arduino.close()
        print(f"\n{'='*60}")
        print("Arduino connection closed")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    
    
    
    successful_times = results_df[results_df['inference_time_us'].notna()]['inference_time_us']
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"Total samples processed: {len(results)}")
    print(f"Successful predictions: {results_df['success'].sum()}")
    print(f"Failed predictions: {(~results_df['success']).sum()}")
    
    if len(successful_times) > 0:
        print(f"\n--- Inference Time Statistics ---")
        print(f"Average: {successful_times.mean():.2f} μs ({successful_times.mean()/1000.0:.2f} ms)")
        print(f"Median:  {successful_times.median():.2f} μs ({successful_times.median()/1000.0:.2f} ms)")
        print(f"Min:     {successful_times.min():.2f} μs ({successful_times.min()/1000.0:.2f} ms)")
        print(f"Max:     {successful_times.max():.2f} μs ({successful_times.max()/1000.0:.2f} ms)")
        print(f"Std Dev: {successful_times.std():.2f} μs")
    
    print(f"{'='*60}")
    
    return results_df


if __name__ == "__main__":
    # Configuration
    CSV_FILE ="path/to/your/sensor_data.csv"  
    ARDUINO_PORT = "/dev/cu.usbmodem14201"  
    OUTPUT_FILE = "predictions_results.csv"
    
    # Run the process
    try:
        results = process_csv_file(
            csv_path=CSV_FILE,
            arduino_port=ARDUINO_PORT,
            output_path=OUTPUT_FILE
        )
        
        print("\nPreview of results:")
        print(results[['sample_id', 'true_x', 'true_y', 'predicted_x', 'predicted_y', 'inference_time_ms']].head(10))
        
    except FileNotFoundError:
        print(f"Error: CSV file '{CSV_FILE}' not found!")
    except serial.SerialException as e:
        print(f"Error: Could not connect to Arduino on {ARDUINO_PORT}")
        print(f"Details: {e}")
        print("\nAvailable ports:")
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(f"  - {port.device}: {port.description}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()