/*
Program Inspired from this sketch:
https://docs.arduino.cc/tutorials/nano-33-ble-sense/get-started-with-machine-learning/


*/


#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>


#include "model.h"

const int numInputs = 16;
const int numOutputs = 2;

// Quantization parameters for NCA INT8
#define INPUT_SCALE 1.5434545278549194
#define INPUT_ZERO_POINT -43
#define OUTPUT_SCALE 1.5434545278549194
#define OUTPUT_ZERO_POINT -43

// Quantization parameters for CNN INT8
// #define INPUT_SCALE 0.9795611500740051
// #define INPUT_ZERO_POINT -77
// #define OUTPUT_SCALE 0.402652770280838
// #define OUTPUT_ZERO_POINT -2

// Global variables for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Static memory buffer for TFLM
constexpr int tensorArenaSize = 128 * 1024;
uint8_t tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// Buffer for CSV data
float csvData[numInputs];

// Helper function to quantize float to int8
int8_t quantizeFloat(float value, float scale, int zero_point) {
  int32_t quantized = (int32_t)round(value / scale) + zero_point;
  
  // Clamp to int8 range [-128, 127]
  if (quantized < -128) return -128;
  if (quantized > 127) return 127;
  
  return (int8_t)quantized;
}

// Helper function to dequantize int8 to float
float dequantizeInt8(int8_t value, float scale, int zero_point) {
  return (value - zero_point) * scale;
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("=================================");
  Serial.println("Position Predictor from CSV");
  Serial.println("=================================");
  Serial.println();

  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(
    tflModel, tflOpsResolver, tensorArena, tensorArenaSize);

  // Allocate memory for the model's tensors
  TfLiteStatus allocateStatus = tflInterpreter->AllocateTensors();
  Serial.print("AllocateTensors(): ");
  Serial.println(allocateStatus);



  // Get pointers for input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  Serial.print("Input type: ");
  Serial.println(tflInputTensor->type); //should be 9 (kTfLite Int8) or 1 for (kTfLite Float32)
  Serial.print("Output type: ");
  Serial.println(tflOutputTensor->type); //should be 9 (kTfLite Int8) or 1 (kTfLite Float32)

    // Print quantization parameters from the model
  if (tflInputTensor->type == kTfLiteInt8) {
    Serial.println("INT8 quantized model detected!");
    Serial.print("Input scale: ");
    Serial.println(tflInputTensor->params.scale);
    Serial.print("Input zero_point: ");
    Serial.println(tflInputTensor->params.zero_point);
    Serial.print("Output scale: ");
    Serial.println(tflOutputTensor->params.scale);
    Serial.print("Output zero_point: ");
    Serial.println(tflOutputTensor->params.zero_point);
  }

  Serial.println("Model loaded successfully!");
  Serial.println();
  Serial.println("Ready to receive data.");
  Serial.println("Paste 16 comma-separated values and press Enter:");
  Serial.println("Example: 1.0,2.5,3.2,4.1,5.6,6.3,7.8,8.2,9.1,10.5,11.2,12.8,13.4,14.6,15.3,16.1");
  Serial.println();

}

void loop() {
  // Wait for Serial input
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.length() > 0) {
      Serial.println("Received input:");
      Serial.println(input);
      Serial.println();
      
      if (parseCSVString(input)) {
        predictPosition();
      } else {
        Serial.println("Error: Could not parse CSV. Please check format.");
        Serial.println();
      }
      
      Serial.println("Ready for next input...");
      Serial.println();
    }
  }
}

bool parseCSVString(String csvString) {
  int index = 0;
  int startPos = 0;
  
  for (int i = 0; i < numInputs; i++) {
    int commaPos = csvString.indexOf(',', startPos);
    
    if (commaPos == -1) {
      // Last value (no comma after it)
      if (i == numInputs - 1) {
        csvData[index] = csvString.substring(startPos).toFloat();
        index++;
      } else {
        Serial.print("Error: Expected ");
        Serial.print(numInputs);
        Serial.print(" values, got ");
        Serial.println(i + 1);
        return false;
      }
      break;
    } else {
      csvData[index] = csvString.substring(startPos, commaPos).toFloat();
      index++;
      startPos = commaPos + 1;
    }
  }

  if (index != numInputs) {
    Serial.print("Error: Expected ");
    Serial.print(numInputs);
    Serial.print(" values, got ");
    Serial.println(index);
    return false;
  }
  // Display parsed values
    Serial.println("Parsed values:");
    for (int i = 0; i < numInputs; i++) {
      Serial.print("  [");
      Serial.print(i);
      Serial.print("]: ");
      Serial.println(csvData[i]);
    }
    Serial.println();

    return true;
}


void predictPosition() {

  if (tflInputTensor->type == kTfLiteInt8) {
    //Quantize input data to INT8
    for (int i = 0; i < numInputs; i++) {
      tflInputTensor->data.int8[i] = quantizeFloat(csvData[i], INPUT_SCALE, INPUT_ZERO_POINT);
    }
  } else {
    //Float model - direct copy
    for (int i = 0; i < numInputs; i++){
      tflInputTensor->data.f[i] = csvData[i];
    }
  }
 
  unsigned long startTime = micros();

  // Run inference
  TfLiteStatus invokeStatus = tflInterpreter->Invoke();

  unsigned long endTime = micros();
  unsigned long inferenceTime = endTime - startTime;

  if (invokeStatus != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // Read output (X, Y position)
  float predictedX, predictedY;

  if (tflOutputTensor->type == kTfLiteInt8) {
   predictedX = dequantizeInt8(tflOutputTensor->data.int8[0], OUTPUT_SCALE, OUTPUT_ZERO_POINT);
   predictedY = dequantizeInt8(tflOutputTensor->data.int8[1], OUTPUT_SCALE, OUTPUT_ZERO_POINT);
  } else {
    predictedX = tflOutputTensor->data.f[0];
    predictedY = tflOutputTensor->data.f[1];
  }
  // Print results
  Serial.println("=== PREDICTION RESULTS ===");
  Serial.print("Predicted X: ");
  Serial.println(predictedX, 6);
  Serial.print("Predicted Y: ");
  Serial.println(predictedY, 6);
  Serial.println("==========================");
  Serial.println();

  // Print inference time
  Serial.print("Inference time: ");
  Serial.print(inferenceTime);
  Serial.print(" μs (");
  Serial.print(inferenceTime / 1000.0, 2);
  Serial.println ( "ms)");
  Serial.println(); 
}