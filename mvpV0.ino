#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <esp_camera.h>
#include <WiFiUdp.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

// --- Configuration ---
const char* ssid = "Digi_Nova";
const char* password = "diginova";
const char* udpAddress = "192.168.137.1"; // CHANGE THIS to your PC's IP
const int udpPort = 10103; // CHANGE THIS for each ESP: 10101, 10102, 10103

#define CAMERA_MODEL_AI_THINKER

#include "camera_pins.h"

// --- System Variables ---
AsyncWebServer server(80);
AsyncWebSocket ws("/ws");
WiFiUDP udp;

volatile bool capture_and_send = false;
volatile int frames_to_send = 0; // For calibration bursts

// --- CHANGE HERE: Slower capture rate for calibration ---
// Delay in milliseconds. 250ms = 4 FPS.
const int calibration_delay_ms = 250; 

void onWsEvent(AsyncWebSocket *server, AsyncWebSocketClient *client, AwsEventType type, void *arg, uint8_t *data, size_t len);

void sendFrameTask(void *pvParameters) {
    camera_fb_t *fb = NULL;
    const char* start_marker = "IMAGE_START";
    const char* end_marker = "IMAGE_END";

    while (true) {
        if (capture_and_send) {
            fb = esp_camera_fb_get();
            if (!fb) {
                Serial.println("Camera capture failed");
                vTaskDelay(1000 / portTICK_PERIOD_MS);
                continue;
            }

            udp.beginPacket(udpAddress, udpPort);
            udp.write((const uint8_t *)start_marker, strlen(start_marker));
            udp.write(fb->buf, fb->len);
            udp.write((const uint8_t *)end_marker, strlen(end_marker));
            udp.endPacket();

            esp_camera_fb_return(fb);

            if (frames_to_send > 0) {
                frames_to_send--;
                if (frames_to_send == 0) {
                    capture_and_send = false;
                }
                // --- CHANGE HERE: Apply the calibration delay ---
                vTaskDelay(calibration_delay_ms / portTICK_PERIOD_MS); 
            } else {
                // For normal live streaming, be as fast as possible
                vTaskDelay(10 / portTICK_PERIOD_MS);
            }
        } else {
            vTaskDelay(100 / portTICK_PERIOD_MS);
        }
    }
}


void setup() {
    Serial.begin(115200);

    // Camera Init
    camera_config_t config;
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
    
    // --- CHANGE HERE: Higher resolution and quality ---
    config.frame_size = FRAMESIZE_SVGA; // 800x600 resolution
    config.jpeg_quality = 10; // 0-63 lower means higher quality
    config.fb_count = 2; // Use 2 frame buffers for higher resolutions

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x", err);
        return;
    }

    // WiFi Connection
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());

    // WebSocket Server
    ws.onEvent(onWsEvent);
    server.addHandler(&ws);
    server.begin();

    // Start the task for sending frames
    xTaskCreatePinnedToCore(sendFrameTask, "SendFrameTask", 4096, NULL, 1, NULL, 1);
}

void onWsEvent(AsyncWebSocket *server, AsyncWebSocketClient *client, AwsEventType type, void *arg, uint8_t *data, size_t len) {
    if (type == WS_EVT_CONNECT) {
        Serial.printf("WebSocket client #%u connected\n", client->id());
    } else if (type == WS_EVT_DISCONNECT) {
        Serial.printf("WebSocket client #%u disconnected\n", client->id());
        capture_and_send = false;
    } else if (type == WS_EVT_DATA) {
        AwsFrameInfo *info = (AwsFrameInfo*)arg;
        if (info->final && info->index == 0 && info->len == len && info->opcode == WS_TEXT) {
            data[len] = 0;
            String msg = (char*)data;
            Serial.printf("Received command: %s\n", msg.c_str());

            if (msg == "start_capture") {
                frames_to_send = -1; // Continuous capture
                capture_and_send = true;
            } else if (msg == "stop_capture") {
                capture_and_send = false;
            } 
            // --- CHANGE HERE: Dynamic frame count for calibration ---
            else if (msg.startsWith("start_calibration:")) {
                String countStr = msg.substring(msg.indexOf(':') + 1);
                int frameCount = countStr.toInt();
                if (frameCount > 0) {
                    frames_to_send = frameCount;
                    capture_and_send = true;
                    Serial.printf("Starting calibration, will send %d frames.\n", frameCount);
                }
            }
            else if (msg == "capture_single_frame") {
                frames_to_send = 1;
                capture_and_send = true;
            }
        }
    }
}

void loop() {
    ws.cleanupClients();
}