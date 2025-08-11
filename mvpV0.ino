#include <esp_camera.h>
#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <Ticker.h>

// --- Configuration ---
const char *ssid = "Digi_Nova";
const char *password = "diginova";

#define CAMERA_MODEL_AI_THINKER

#include "camera_pins.h"

// --- System Variables ---
AsyncWebServer server(80);
AsyncWebSocket ws("/ws");

volatile bool capture_and_send = false;
volatile int frames_to_send = 0;  // For calibration bursts

// --- CHANGE HERE: Slower capture rate for calibration ---
// Delay in milliseconds. 250ms = 4 FPS.
const int calibration_delay_ms = 500;

#define LED_PIN 33  // or whatever GPIO you're using for blinking
Ticker blinkTicker;
bool ledState = false;
unsigned long lastWiFiCheck = 0;

void blink() {
  digitalWrite(LED_PIN, LOW);
  delay(200);
  digitalWrite(LED_PIN, HIGH);
}

void blinkThreeTimes() {
  for (int i = 1; i < 7; i++) {
    digitalWrite(LED_PIN, i % 2 == 0);
    delay(200);
  }
  digitalWrite(LED_PIN, HIGH);
}

void onWsEvent(AsyncWebSocket *server, AsyncWebSocketClient *client, AwsEventType type, void *arg, uint8_t *data, size_t len);

bool camera_initialized = false;
camera_config_t config;  // make global so you can reuse it

bool initCamera() {
  if (camera_initialized)
    return true;
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }
  camera_initialized = true;
  return true;
}

void deinitCamera() {
  if (!camera_initialized)
    return;
  esp_camera_deinit();
  camera_initialized = false;
}

void sendFrameTask(void *pvParameters) {
  camera_fb_t *fb = NULL;

  while (true) {
    if (capture_and_send) {
      if (!initCamera()) {
        vTaskDelay(1000 / portTICK_PERIOD_MS);
        continue;
      }

      fb = esp_camera_fb_get();
      if (!fb) {
        Serial.println("Camera capture failed");
        vTaskDelay(1000 / portTICK_PERIOD_MS);
        continue;
      }

      if (ws.count() > 0) {
        ws.binaryAll(fb->buf, fb->len);
      }

      esp_camera_fb_return(fb);

      if (frames_to_send > 0) {
        frames_to_send--;
        if (frames_to_send == 0) {
          capture_and_send = false;
          deinitCamera();
        }
      }

      vTaskDelay(calibration_delay_ms / portTICK_PERIOD_MS);
    } else {
      vTaskDelay(100 / portTICK_PERIOD_MS);
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  delay(500);

  if (!psramFound()) {
    Serial.println("PSRAM not found - camera will fail");
  }

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;  //vga for testing
  config.jpeg_quality = 10;
  config.fb_count = 2;

  if (initCamera()) {
    Serial.println("Camera first initialized");
  }

  // WiFi Connection with blinking
  blinkTicker.attach(1, blink);
  WiFi.begin(ssid, password);
  Serial.println("\nConnecting to wifi..");
  while (WiFi.status() != WL_CONNECTED) {
    delay(100);
  }
  blinkTicker.detach();
  blinkThreeTimes();

  Serial.println("\nWiFi connected");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // WebSocket Server
  ws.onEvent(onWsEvent);
  server.addHandler(&ws);
  server.begin();

  // Start the task for sending frames
  // Reduce stack size so it doesn't eat PSRAM
  xTaskCreatePinnedToCore(sendFrameTask, "SendFrameTask", 2048, NULL, 1, NULL, 1);
}

void onWsEvent(AsyncWebSocket *server, AsyncWebSocketClient *client, AwsEventType type, void *arg, uint8_t *data, size_t len) {
  if (type == WS_EVT_CONNECT) {
    Serial.printf("WebSocket client #%u connected\n", client->id());
  } else if (type == WS_EVT_DISCONNECT) {
    Serial.printf("WebSocket client #%u disconnected\n", client->id());
    capture_and_send = false;
  } else if (type == WS_EVT_DATA) {
    AwsFrameInfo *info = (AwsFrameInfo *)arg;
    if (info->final && info->index == 0 && info->len == len && info->opcode == WS_TEXT) {
      data[len] = 0;
      String msg = (char *)data;
      Serial.printf("Received command: %s\n", msg.c_str());
      if (msg == "start_capture") {
        frames_to_send = -1;  // Continuous capture
        capture_and_send = true;
      } else if (msg == "stop_capture") {
        capture_and_send = false;
      } else if (msg == "wake_camera") {
        if (initCamera()) {
          Serial.println("Camera woken up manually.");
        }
      } else if (msg == "sleep_camera") {
        deinitCamera();
        Serial.println("Camera put to sleep manually.");
      } else if (msg.startsWith("start_calibration:")) {
        String countStr = msg.substring(msg.indexOf(':') + 1);
        int frameCount = countStr.toInt();
        if (frameCount > 0) {
          initCamera();
          delay(500);  // 0.5s wait before first capture
          frames_to_send = frameCount;
          capture_and_send = true;
          Serial.printf("Starting calibration, will send %d frames.\n", frameCount);
        }
      } else if (msg == "capture_single_frame") {
        initCamera();
        delay(500);
        frames_to_send = 1;
        capture_and_send = true;
      }
    }
  }
}

void loop() {
  ws.cleanupClients();

  // Check every 5s if WiFi disconnected
  if (millis() - lastWiFiCheck > 5000) {
    lastWiFiCheck = millis();
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi disconnected. Reconnecting...");
      blinkTicker.attach(1, blink);
      WiFi.disconnect();
      WiFi.begin(ssid, password);
      while (WiFi.status() != WL_CONNECTED) {
        delay(100);
      }
      blinkTicker.detach();
      blinkThreeTimes();
      Serial.println("Reconnected to WiFi.");
    }
  }
}