/**
 * CrowdDynamix ESP32 Audio Sensor Module
 * 
 * Reads audio data from INMP441 I2S microphone and sends
 * processed audio metrics to the Python backend server.
 * 
 * Hardware Connections (ESP32 -> INMP441):
 *   - 3.3V  -> VDD
 *   - GND   -> GND & L/R (L/R to GND for left channel)
 *   - GPIO25 -> WS (Word Select / LRCLK)
 *   - GPIO33 -> SD (Serial Data)
 *   - GPIO32 -> SCK (Serial Clock / BCLK)
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <driver/i2s.h>
#include <math.h>

// ============================================================================
// Configuration - MODIFY THESE VALUES
// ============================================================================

// WiFi credentials
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Backend server configuration
const char* SERVER_HOST = "192.168.1.100";  // Change to your server IP
const int SERVER_PORT = 8000;

// Sensor identification
const char* CHOKE_POINT_ID = "choke_point_1";  // Must match backend choke point ID
const char* DEVICE_ID = "esp32_audio_001";

// ============================================================================
// I2S Configuration for INMP441
// ============================================================================

#define I2S_PORT          I2S_NUM_0
#define I2S_WS            25    // Word Select (LRCLK)
#define I2S_SD            33    // Serial Data
#define I2S_SCK           32    // Serial Clock (BCLK)

#define SAMPLE_RATE       16000
#define SAMPLE_BITS       32
#define SAMPLES_PER_READ  512
#define DMA_BUF_COUNT     4
#define DMA_BUF_LEN       1024

// Audio buffer
int32_t i2sBuffer[SAMPLES_PER_READ];

// ============================================================================
// Audio Processing Configuration
// ============================================================================

// Calibration values (adjust based on your environment)
#define NOISE_FLOOR           500       // Baseline noise level
#define AMBIENT_THRESHOLD     2000      // Below this = ambient
#define LOUD_THRESHOLD        8000      // Above this = loud
#define DISTRESSED_THRESHOLD  15000     // Above this = distressed (screaming)
#define SPIKE_THRESHOLD       20000     // Sudden loud sound detection
#define SPIKE_RATIO           3.0       // Current/average ratio for spike

// Timing
#define SEND_INTERVAL_MS      500       // Send data every 500ms
#define SAMPLE_WINDOW_MS      100       // Sample window for averaging

// ============================================================================
// Global Variables
// ============================================================================

// Audio metrics (running values)
float currentSoundLevel = 0.0;
float averageSoundLevel = 0.0;
float peakSoundLevel = 0.0;
float soundEnergyLevel = 0.0;

// Spike detection
bool spikeDetected = false;
float spikeIntensity = 0.0;
float previousAverage = 0.0;

// Exponential moving average factor
const float EMA_ALPHA = 0.1;

// Timing
unsigned long lastSendTime = 0;
unsigned long lastSampleTime = 0;

// WiFi reconnection
unsigned long lastWifiCheck = 0;
const unsigned long WIFI_CHECK_INTERVAL = 10000;  // 10 seconds

// HTTP client
HTTPClient http;

// ============================================================================
// Function Prototypes
// ============================================================================

void setupWiFi();
void setupI2S();
void readAudioSamples();
void processAudioMetrics();
void sendDataToServer();
String determineAudioCharacter();
float normalizeLevel(float level);
void reconnectWiFi();

// ============================================================================
// Setup
// ============================================================================

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println();
  Serial.println("========================================");
  Serial.println("CrowdDynamix ESP32 Audio Sensor");
  Serial.println("========================================");
  Serial.println();
  
  // Initialize WiFi
  setupWiFi();
  
  // Initialize I2S for INMP441
  setupI2S();
  
  Serial.println("Setup complete. Starting audio monitoring...");
  Serial.println();
}

// ============================================================================
// Main Loop
// ============================================================================

void loop() {
  unsigned long currentTime = millis();
  
  // Check WiFi connection periodically
  if (currentTime - lastWifiCheck >= WIFI_CHECK_INTERVAL) {
    lastWifiCheck = currentTime;
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi disconnected. Reconnecting...");
      reconnectWiFi();
    }
  }
  
  // Read and process audio samples continuously
  readAudioSamples();
  processAudioMetrics();
  
  // Send data to server at configured interval
  if (currentTime - lastSendTime >= SEND_INTERVAL_MS) {
    lastSendTime = currentTime;
    
    if (WiFi.status() == WL_CONNECTED) {
      sendDataToServer();
    } else {
      Serial.println("WiFi not connected. Skipping data send.");
    }
    
    // Reset spike detection after sending
    spikeDetected = false;
    spikeIntensity = 0.0;
  }
}

// ============================================================================
// WiFi Functions
// ============================================================================

void setupWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("WiFi connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("Signal Strength (RSSI): ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
  } else {
    Serial.println();
    Serial.println("Failed to connect to WiFi. Will retry in loop.");
  }
}

void reconnectWiFi() {
  WiFi.disconnect();
  delay(1000);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 10) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println(" Reconnected!");
  } else {
    Serial.println(" Reconnection failed.");
  }
}

// ============================================================================
// I2S Functions
// ============================================================================

void setupI2S() {
  Serial.println("Configuring I2S for INMP441...");
  
  // I2S configuration
  const i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = DMA_BUF_COUNT,
    .dma_buf_len = DMA_BUF_LEN,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  
  // I2S pin configuration
  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };
  
  // Install and configure I2S driver
  esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("Failed to install I2S driver: %d\n", err);
    return;
  }
  
  err = i2s_set_pin(I2S_PORT, &pin_config);
  if (err != ESP_OK) {
    Serial.printf("Failed to set I2S pins: %d\n", err);
    return;
  }
  
  // Clear DMA buffers
  i2s_zero_dma_buffer(I2S_PORT);
  
  Serial.println("I2S configured successfully.");
}

// ============================================================================
// Audio Processing Functions
// ============================================================================

void readAudioSamples() {
  size_t bytesRead = 0;
  
  // Read samples from I2S
  esp_err_t result = i2s_read(
    I2S_PORT,
    &i2sBuffer,
    sizeof(i2sBuffer),
    &bytesRead,
    portMAX_DELAY
  );
  
  if (result != ESP_OK || bytesRead == 0) {
    return;
  }
  
  int samplesRead = bytesRead / sizeof(int32_t);
  
  // Calculate RMS (Root Mean Square) for sound level
  double sumSquares = 0.0;
  int32_t maxSample = 0;
  
  for (int i = 0; i < samplesRead; i++) {
    // INMP441 outputs 24-bit data in 32-bit frame, left-aligned
    // Shift right to get actual value and remove DC offset
    int32_t sample = i2sBuffer[i] >> 8;  // Convert to 24-bit
    
    // Take absolute value for amplitude
    int32_t absSample = abs(sample);
    
    sumSquares += (double)sample * (double)sample;
    
    if (absSample > maxSample) {
      maxSample = absSample;
    }
  }
  
  // RMS calculation
  double rms = sqrt(sumSquares / samplesRead);
  currentSoundLevel = (float)rms;
  
  // Track peak
  if (maxSample > peakSoundLevel) {
    peakSoundLevel = (float)maxSample;
  }
  
  // Decay peak slowly
  peakSoundLevel *= 0.995;
}

void processAudioMetrics() {
  // Store previous average for spike detection
  previousAverage = averageSoundLevel;
  
  // Update exponential moving average
  averageSoundLevel = (EMA_ALPHA * currentSoundLevel) + ((1.0 - EMA_ALPHA) * averageSoundLevel);
  
  // Normalize sound level to 0-1 range
  soundEnergyLevel = normalizeLevel(currentSoundLevel);
  
  // Spike detection: sudden increase in sound level
  if (previousAverage > 0 && currentSoundLevel > SPIKE_THRESHOLD) {
    float ratio = currentSoundLevel / previousAverage;
    if (ratio > SPIKE_RATIO) {
      spikeDetected = true;
      spikeIntensity = min(1.0f, (ratio - SPIKE_RATIO) / SPIKE_RATIO);
      
      Serial.println("*** AUDIO SPIKE DETECTED ***");
      Serial.printf("  Ratio: %.2f, Intensity: %.2f\n", ratio, spikeIntensity);
    }
  }
}

float normalizeLevel(float level) {
  // Normalize to 0-1 range based on thresholds
  float normalized = (level - NOISE_FLOOR) / (DISTRESSED_THRESHOLD - NOISE_FLOOR);
  return constrain(normalized, 0.0f, 1.0f);
}

String determineAudioCharacter() {
  if (currentSoundLevel >= DISTRESSED_THRESHOLD || spikeDetected) {
    return "distressed";
  } else if (currentSoundLevel >= LOUD_THRESHOLD) {
    return "loud";
  } else {
    return "ambient";
  }
}

// ============================================================================
// Server Communication
// ============================================================================

void sendDataToServer() {
  // Build server URL
  String url = "http://";
  url += SERVER_HOST;
  url += ":";
  url += SERVER_PORT;
  url += "/sensor/audio";  // Endpoint for audio sensor data
  
  // Create JSON payload matching AudioSensorData model
  StaticJsonDocument<512> doc;
  
  doc["choke_point_id"] = CHOKE_POINT_ID;
  doc["device_id"] = DEVICE_ID;
  doc["timestamp"] = millis() / 1000.0;  // Seconds since boot
  
  // Audio metrics
  doc["sound_energy_level"] = soundEnergyLevel;
  doc["spike_detected"] = spikeDetected;
  doc["spike_intensity"] = spikeIntensity;
  doc["audio_character"] = determineAudioCharacter();
  
  // Additional raw data for debugging/calibration
  JsonObject raw = doc.createNestedObject("raw_data");
  raw["current_level"] = currentSoundLevel;
  raw["average_level"] = averageSoundLevel;
  raw["peak_level"] = peakSoundLevel;
  raw["wifi_rssi"] = WiFi.RSSI();
  
  String jsonPayload;
  serializeJson(doc, jsonPayload);
  
  // Debug output
  Serial.println("--- Sending Audio Data ---");
  Serial.printf("Energy Level: %.3f\n", soundEnergyLevel);
  Serial.printf("Audio Character: %s\n", determineAudioCharacter().c_str());
  Serial.printf("Spike: %s (intensity: %.2f)\n", 
                spikeDetected ? "YES" : "no", spikeIntensity);
  
  // Send HTTP POST request
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  
  int httpResponseCode = http.POST(jsonPayload);
  
  if (httpResponseCode > 0) {
    Serial.printf("Server Response: %d\n", httpResponseCode);
    
    if (httpResponseCode == 200 || httpResponseCode == 201) {
      String response = http.getString();
      Serial.println("Response: " + response);
    }
  } else {
    Serial.printf("HTTP Error: %s\n", http.errorToString(httpResponseCode).c_str());
  }
  
  http.end();
  Serial.println();
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Print current audio levels (for debugging/calibration)
 */
void printDebugInfo() {
  Serial.println("=== Audio Debug Info ===");
  Serial.printf("Current Level: %.0f\n", currentSoundLevel);
  Serial.printf("Average Level: %.0f\n", averageSoundLevel);
  Serial.printf("Peak Level: %.0f\n", peakSoundLevel);
  Serial.printf("Normalized Energy: %.3f\n", soundEnergyLevel);
  Serial.printf("Character: %s\n", determineAudioCharacter().c_str());
  Serial.printf("Free Heap: %d bytes\n", ESP.getFreeHeap());
  Serial.println();
}
