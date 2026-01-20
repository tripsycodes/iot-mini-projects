/**
 * @file receiver.ino
 * @brief LoRa receiver for BSEC2 environmental sensor packets
 *
 * Receives packets and sends data to an API server via WiFi
 */

#include "display.h"
#include "lora_config.h"
#include "packet.h"
#include <Arduino.h>
#include <HTTPClient.h>
#include <LoRaWan_APP.h>
#include <WiFi.h>

// --- CONFIGURATION ---
const char *WIFI_SSID = "TP-Link_F765";
const char *WIFI_PASS = "nottplink";
const char *API_ENDPOINT = "http://192.168.50.101:5000/api/sensor";
// ---------------------

#define BAUD 115200

static OledDisplay oledDisplay;
static RadioEvents_t radioEvents;

static LoRaPacket rxPacket;
static uint8_t rxBuffer[PACKET_SIZE + 10];
static uint16_t rxSize = 0;
static bool packetReceived = false;

static int16_t lastRssi = 0;
static int8_t lastSnr = 0;
static uint32_t packetsReceived = 0;
static uint32_t packetsError = 0;
static int lastHttpStatus = 0;

static void onTxDone() {}
static void onTxTimeout() {}

static void onRxDone(uint8_t *payload, uint16_t size, int16_t rssi,
                     int8_t snr) {
  lastRssi = rssi;
  lastSnr = snr;
  rxSize = size;

  if (size <= sizeof(rxBuffer)) {
    memcpy(rxBuffer, payload, size);
    packetReceived = true;
  }
  Radio.Rx(0);
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
static String createJsonPayload() {
  char json[1024];
  sprintf(json,
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
          "\"rssi\":%d,"
          "\"snr\":%d}",
          rxPacket.deviceId, rxPacket.sequence, rxPacket.uptime,
          rxPacket.temperature / 100.0, rxPacket.humidity / 100.0,
          rxPacket.pressure / 1000.0, rxPacket.iaq, rxPacket.iaqAccuracy,
          getIaqLabel(rxPacket.iaq), rxPacket.staticIaq, rxPacket.co2Equivalent,
          rxPacket.breathVoc / 100.0, rxPacket.gasPercentage,
          rxPacket.stabStatus ? "true" : "false",
          rxPacket.runInStatus ? "true" : "false", lastRssi, lastSnr);
  return String(json);
}

// Send payload to API
static void sendToApi(String payload) {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(API_ENDPOINT);
    http.addHeader("Content-Type", "application/json");

    Serial.println("POSTing to API...");
    int httpResponseCode = http.POST(payload);
    lastHttpStatus = httpResponseCode;

    if (httpResponseCode > 0) {
      Serial.printf("API Response: %d\n", httpResponseCode);
      String response = http.getString();
      Serial.println(response);
    } else {
      Serial.printf("API Error: %s\n",
                    http.errorToString(httpResponseCode).c_str());
    }
    http.end();
  } else {
    Serial.println("WiFi disconnected, reconnecting...");
    WiFi.disconnect();
    WiFi.reconnect();
    lastHttpStatus = -1;
  }
}

static char buffer[64];

static void drawReceiverDashboard() {
  oledDisplay.clear();

  // Line 0: Packet Stats
  sprintf(buffer, "RX#%lu API:%d", packetsReceived, lastHttpStatus);
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
  sprintf(buffer, "P:%u.%03uMPa", rxPacket.pressure / 1000,
          rxPacket.pressure % 1000);
  oledDisplay.write(0, 3, buffer);

  // Line 4: Signal
  sprintf(buffer, "RSSI:%d SNR:%d", lastRssi, lastSnr);
  oledDisplay.write(0, 4, buffer);

  oledDisplay.commit();
}

static void drawWaitingScreen() {
  static uint8_t dots = 0;
  oledDisplay.clear();
  oledDisplay.write(0, 0, " LORA -> API ");

  if (WiFi.status() == WL_CONNECTED) {
    sprintf(buffer, "WiFi: OK (%s)", WiFi.localIP().toString().c_str());
    oledDisplay.write(0, 1, buffer);
  } else {
    oledDisplay.write(0, 1, "WiFi: Disconnected");
  }

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

  // Setup Radio
  radioEvents.TxDone = onTxDone;
  radioEvents.TxTimeout = onTxTimeout;
  radioEvents.RxDone = onRxDone;
  radioEvents.RxTimeout = onRxTimeout;
  radioEvents.RxError = onRxError;

  Radio.Init(&radioEvents);
  Radio.SetChannel(RF_FREQUENCY);
  Radio.SetRxConfig(MODEM_LORA, LORA_BANDWIDTH, LORA_SPREADING_FACTOR,
                    LORA_CODINGRATE, 0, LORA_PREAMBLE_LENGTH,
                    LORA_FIX_LENGTH_PAYLOAD_ON, 0, true, 0, 0,
                    LORA_IQ_INVERSION_ON, true, RX_TIMEOUT_VALUE);

  Radio.Rx(0);
  Serial.println("Setup Complete. Listening...");
}

void loop() {
  Radio.IrqProcess();

  if (packetReceived) {
    packetReceived = false;

    if (decodePacket(rxBuffer, rxSize, rxPacket)) {
      packetsReceived++;
      Serial.println("[OK] Packet Received");

      String payload = createJsonPayload();
      Serial.println(payload);

      sendToApi(payload);
      drawReceiverDashboard();
    } else {
      packetsError++;
      Serial.println("[ERR] JSON Decode Failed");
    }
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
