/**
 * @file receiver.ino
 * @brief LoRa receiver for BSEC2 environmental sensor packets
 *
 * Receives packets and sends data to an API server via WiFi
 * Optimized for high-speed packet reception with non-blocking API calls
 */

#include "display.h"
#include "lora_config.h"
#include "packet.h"
#include <Arduino.h>
#include <HTTPClient.h>
#include <LoRaWan_APP.h>
#include <WiFi.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/task.h>

// --- CONFIGURATION ---
const char *WIFI_SSID = "9.0 GHz";
const char *WIFI_PASS = "notsahilraj";
const char *API_ENDPOINT = "http://10.94.2.241:5000/api/sensor";
const uint16_t API_QUEUE_SIZE = 50;   // Buffer up to 50 packets
const uint16_t API_TIMEOUT_MS = 2000; // 2 second timeout
// ---------------------

#define BAUD 115200

static OledDisplay oledDisplay;
static RadioEvents_t radioEvents;

static LoRaPacket rxPacket;
static AnalogPacket rxAnalog; // Cache for analog data
static uint8_t rxBuffer[PACKET_SIZE + 10];
static uint16_t rxSize = 0;
static bool packetReceived = false;

static int16_t lastRssi = 0;
static int8_t lastSnr = 0;
static uint32_t packetsReceived = 0;
static uint32_t packetsError = 0;
static int lastHttpStatus = 0;
static uint32_t apiQueuedCount = 0;
static uint32_t apiSentCount = 0;
static uint32_t apiFailedCount = 0;

// FreeRTOS Queue for API requests
static QueueHandle_t apiQueue = NULL;

// Structure for queued API data
struct ApiPayload {
  char json[1024];
};

static void onTxDone() {}
static void onTxTimeout() {}

static void onRxDone(uint8_t *payload, uint16_t size, int16_t rssi,
                     int8_t snr) {
  // Prevent buffer overwrite if previous packet not processed
  if (packetReceived) {
    Radio.Rx(0); // Re-enable RX immediately
    return;      // Discard current packet
  }

  lastRssi = rssi;
  lastSnr = snr;
  rxSize = size;

  if (size <= sizeof(rxBuffer)) {
    memcpy(rxBuffer, payload, size);
    packetReceived = true; // Signal that a new packet is ready for processing
  }
  Radio.Rx(0); // Re-enable RX
}

static void onRxTimeout() { Radio.Rx(0); }

static void onRxError() {
  packetsError++;
  Serial.println("[RX] CRC Error");
  Radio.Rx(0);
}

// Connect to WiFi
static void connectToWifi() {
  Serial.printf("Connecting to %s...", WIFI_SSID);
  oledDisplay.clear();
  oledDisplay.write(0, 0, " Connecting WiFi ");
  oledDisplay.write(0, 2, WIFI_SSID);
  oledDisplay.commit();

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi Connected!");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
    oledDisplay.write(0, 3, "Connected!");
  } else {
    Serial.println("\nWiFi Failed!");
    oledDisplay.write(0, 3, "Failed!");
  }
  oledDisplay.commit();
  delay(1000);
}

static const char *getIaqLabel(uint16_t iaq) {
  if (iaq <= 50)
    return "good";
  if (iaq <= 100)
    return "ok";
  if (iaq <= 150)
    return "fair";
  if (iaq <= 200)
    return "poor";
  if (iaq <= 300)
    return "bad";
  return "hazard";
}

// Construct JSON payload
static void createJsonPayload(char *json, size_t maxLen) {
  snprintf(json, maxLen,
           "{\"device_id\":%u,"
           "\"sequence\":%u,"
           "\"uptime\":%u,"
           "\"temperature\":%.2f,"
           "\"humidity\":%.2f,"
           "\"pressure\":%.3f,"
           "\"iaq\":%u,"
           "\"iaq_accuracy\":%u,"
           "\"iaq_label\":\"%s\","
           "\"static_iaq\":%u,"
           "\"co2_ppm\":%u,"
           "\"voc_ppm\":%.2f,"
           "\"gas_percent\":%u,"
           "\"stabilized\":%s,"
           "\"run_in_complete\":%s,"
           "\"mq135_raw\":%u,"
           "\"anemometer_raw\":%u,"
           "\"rssi\":%d,"
           "\"snr\":%d}",
           rxPacket.deviceId, rxPacket.sequence, rxPacket.uptime,
           rxPacket.temperature / 100.0, rxPacket.humidity / 100.0,
           rxPacket.pressure / 1000.0, rxPacket.iaq, rxPacket.iaqAccuracy,
           getIaqLabel(rxPacket.iaq), rxPacket.staticIaq,
           rxPacket.co2Equivalent, rxPacket.breathVoc / 100.0,
           rxPacket.gasPercentage, rxPacket.stabStatus ? "true" : "false",
           rxPacket.runInStatus ? "true" : "false", rxAnalog.mq135,
           rxAnalog.anemometer, lastRssi, lastSnr);
}

// Queue payload for async sending
static void queueApiPayload() {
  ApiPayload payload;
  createJsonPayload(payload.json, sizeof(payload.json));

  if (xQueueSend(apiQueue, &payload, 0) == pdTRUE) {
    apiQueuedCount++;
  } else {
    Serial.println("[API] Queue full, dropping packet");
  }
}

// FreeRTOS task for sending API requests (runs asynchronously)
static void apiSenderTask(void *parameter) {
  ApiPayload payload;
  HTTPClient http;

  while (true) {
    // Wait for data in the queue (blocks until available)
    if (xQueueReceive(apiQueue, &payload, portMAX_DELAY) == pdTRUE) {
      if (WiFi.status() == WL_CONNECTED) {
        http.begin(API_ENDPOINT);
        http.addHeader("Content-Type", "application/json");
        http.setTimeout(API_TIMEOUT_MS);

        int httpResponseCode = http.POST(payload.json);
        lastHttpStatus = httpResponseCode;

        if (httpResponseCode > 0) {
          apiSentCount++;
          // Only print success for debugging
          // Serial.printf("[API] Success: %d\n", httpResponseCode);
        } else {
          apiFailedCount++;
          Serial.printf("[API] Error: %s\n",
                        http.errorToString(httpResponseCode).c_str());
        }
        http.end();
      } else {
        apiFailedCount++;
        Serial.println("[API] WiFi disconnected");
        // Try to reconnect
        WiFi.disconnect();
        WiFi.reconnect();
        vTaskDelay(1000 / portTICK_PERIOD_MS);
      }
    }
  }
}

static char buffer[64];

static void drawReceiverDashboard() {
  oledDisplay.clear();

  // Line 0: Packet Stats
  sprintf(buffer, "RX:%lu Q:%lu", packetsReceived, apiQueuedCount);
  oledDisplay.write(0, 0, buffer);

  // Line 1: Temp & Humidity
  sprintf(buffer, "T:%d.%02dC H:%u.%02u%%", rxPacket.temperature / 100,
          abs(rxPacket.temperature % 100), rxPacket.humidity / 100,
          rxPacket.humidity % 100);
  oledDisplay.write(0, 1, buffer);

  // Line 2: IAQ & CO2
  sprintf(buffer, "IAQ:%u CO2:%u", rxPacket.iaq, rxPacket.co2Equivalent);
  oledDisplay.write(0, 2, buffer);

  // Line 3: API Status / Pressure
  sprintf(buffer, "API:%d Sent:%lu", lastHttpStatus, apiSentCount);
  oledDisplay.write(0, 3, buffer);

  // Line 4: Analog Sensors
  sprintf(buffer, "MQ:%u AN:%u", rxAnalog.mq135, rxAnalog.anemometer);
  oledDisplay.write(0, 4, buffer);

  oledDisplay.commit();
}

static void drawWaitingScreen() {
  static uint8_t dots = 0;
  oledDisplay.clear();
  oledDisplay.write(0, 0, " LORA -> API ");

  if (WiFi.status() == WL_CONNECTED) {
    sprintf(buffer, "WiFi: OK");
    oledDisplay.write(0, 1, buffer);
  } else {
    oledDisplay.write(0, 1, "WiFi: Disconnected");
  }

  sprintf(buffer, "Q:%lu Sent:%lu", uxQueueMessagesWaiting(apiQueue),
          apiSentCount);
  oledDisplay.write(0, 2, buffer);

  sprintf(buffer, "Waiting%.*s", (dots % 4) + 1, "....");
  oledDisplay.write(0, 3, buffer);

  oledDisplay.commit();
  dots++;
}

void setup() {
  Serial.begin(BAUD);
  Mcu.begin(HELTEC_BOARD, SLOW_CLK_TPYE);

  oledDisplay.init();

  // Setup WiFi
  connectToWifi();

  // Create API queue
  apiQueue = xQueueCreate(API_QUEUE_SIZE, sizeof(ApiPayload));
  if (apiQueue == NULL) {
    Serial.println("[FATAL] Failed to create API queue");
    while (1) {
      delay(1000);
    }
  }

  // Create API sender task (runs on separate core)
  xTaskCreatePinnedToCore(apiSenderTask, // Task function
                          "API Sender",  // Task name
                          8192,          // Stack size (bytes)
                          NULL,          // Parameters
                          1,             // Priority
                          NULL,          // Task handle
                          0 // Core ID (0 = separate from main loop)
  );

  // Setup Radio
  radioEvents.TxDone = onTxDone;
  radioEvents.TxTimeout = onTxTimeout;
  radioEvents.RxDone = onRxDone;
  radioEvents.RxTimeout = onRxTimeout;
  radioEvents.RxError = onRxError;

  Radio.Init(&radioEvents);
  Radio.SetChannel(RF_FREQUENCY);
  Radio.SetTxConfig(MODEM_LORA, TX_OUTPUT_POWER, 0, LORA_BANDWIDTH,
                    LORA_SPREADING_FACTOR, LORA_CODINGRATE,
                    LORA_PREAMBLE_LENGTH, LORA_FIX_LENGTH_PAYLOAD_ON, true, 0,
                    0, LORA_IQ_INVERSION_ON, 3000);

  Radio.SetRxConfig(MODEM_LORA, LORA_BANDWIDTH, LORA_SPREADING_FACTOR,
                    LORA_CODINGRATE, 0, LORA_PREAMBLE_LENGTH,
                    LORA_FIX_LENGTH_PAYLOAD_ON, 0, true, 0, 0,
                    LORA_IQ_INVERSION_ON, true, RX_TIMEOUT_VALUE);

  Radio.Rx(0);

  // Init packets
  initPacket(&rxPacket, 0);
  initAnalogPacket(&rxAnalog, 0);

  Serial.println("Setup Complete. Listening...");
  Serial.println("[OPTIMIZED] Using async API queue");
}

void loop() {
  Radio.IrqProcess();

  if (packetReceived) {
    // Check Packet Type (Byte 1)
    uint8_t packetType = rxBuffer[1];

    // Throttle API sending to 200ms intervals
    static uint32_t lastApiSend = 0;
    uint32_t now = millis();
    bool canSendApi = (now - lastApiSend >= 200);

    if (packetType == PACKET_TYPE_ENV) {
      if (decodePacket(rxBuffer, rxSize, rxPacket)) {
        packetsReceived++;
        Serial.println("[OK] ENV Packet Received");

        // Send to API if throttle allows
        if (canSendApi) {
          queueApiPayload();
          lastApiSend = now;
        }
        drawReceiverDashboard();
      } else {
        packetsError++;
        Serial.printf("[ERR] ENV Decode Failed. Size: %u (Expected: %u)\n",
                      rxSize, PACKET_SIZE);
        Serial.print("Hex: ");
        for (int i = 0; i < rxSize && i < 40; i++)
          Serial.printf("%02X ", rxBuffer[i]);
        Serial.println();

        if (rxSize >= PACKET_SIZE) {
          uint16_t calcCRC = bsecCRC16(rxBuffer, PACKET_SIZE - 2);
          uint16_t recvCRC = ((uint16_t)rxBuffer[PACKET_SIZE - 2] << 8) |
                             rxBuffer[PACKET_SIZE - 1];
          Serial.printf("CRC Calc: 0x%04X, Recv: 0x%04X\n", calcCRC, recvCRC);
        }
      }
    } else if (packetType == PACKET_TYPE_ANALOG) {
      if (decodeAnalogPacket(rxBuffer, rxSize, rxAnalog)) {
        // Update analog data and send to API live
        Serial.printf("[RX] ANALOG: MQ:%u AN:%u\n", rxAnalog.mq135,
                      rxAnalog.anemometer);

        // Send to API if throttle allows
        if (canSendApi) {
          queueApiPayload();
          lastApiSend = now;
        }
        drawReceiverDashboard();
      } else {
        // Silent fail for analog decode errors to avoid log spam
      }
    } else {
      Serial.printf("[ERR] Unknown Packet Type: 0x%02X\n", packetType);
    }

    // Release buffer lock after processing
    packetReceived = false;
  } else {
    static uint32_t lastUpdate = 0;
    if (millis() - lastUpdate > 1000) {
      lastUpdate = millis();
      if (packetsReceived == 0) {
        drawWaitingScreen();
      }
    }
  }
}
