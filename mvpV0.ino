#include <esp_camera.h>
#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

// --- Configuration ---
const char *ssid = "Digi_Nova";
const char *password = "diginova";

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// --- System Variables ---
AsyncWebServer *server = nullptr;
AsyncWebSocket *ws = nullptr;

volatile bool capture_requested = false;
volatile bool streaming_active = false;

#define LED_PIN 33

// Camera configuration optimized for high quality without PSRAM
camera_config_t config;
bool camera_initialized = false;

void setupOptimalCamera() {
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
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

  Serial.printf("PSRAM found: %s\n", psramFound() ? "YES" : "NO");
  Serial.printf("Free DRAM: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Largest free DRAM block: %d bytes\n", heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

  if (psramFound()) {
    Serial.println("PSRAM detected - using enhanced settings");
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.frame_size = FRAMESIZE_UXGA;  // 1600x1200
    config.jpeg_quality = 4;             // Higher quality with PSRAM
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
  } else {
    Serial.println("No PSRAM - using conservative DRAM settings");
    config.fb_location = CAMERA_FB_IN_DRAM;

    // Check available memory and adjust accordingly
    size_t free_dram = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    size_t largest_block = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);

    Serial.printf("Available DRAM: %d, Largest block: %d\n", free_dram, largest_block);

    // Progressive fallback based on available memory
    if (largest_block > 200000) {
      // Try SXGA first
      config.frame_size = FRAMESIZE_SXGA;  // 1280x1024
      config.jpeg_quality = 8;
    } else if (largest_block > 150000) {
      // Fall back to XGA
      config.frame_size = FRAMESIZE_XGA;  // 1024x768
      config.jpeg_quality = 8;
    } else if (largest_block > 100000) {
      // Fall back to SVGA
      config.frame_size = FRAMESIZE_SVGA;  // 800x600
      config.jpeg_quality = 10;
    } else {
      // Last resort - VGA
      config.frame_size = FRAMESIZE_VGA;  // 640x480
      config.jpeg_quality = 12;
    }

    config.fb_count = 1;  // Single frame buffer only
  }

  Serial.printf("Selected frame size: %d, quality: %d\n", config.frame_size, config.jpeg_quality);
}

bool initCamera() {
  if (camera_initialized) return true;
  Serial.println(config.fb_location);
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return false;
  }

  // Fine-tune camera settings for best quality
  sensor_t *s = esp_camera_sensor_get();
  if (s != NULL) {
    // Optimize image quality settings
    s->set_brightness(s, 0);                  // -2 to 2
    s->set_contrast(s, 0);                    // -2 to 2
    s->set_saturation(s, 0);                  // -2 to 2
    s->set_special_effect(s, 0);              // 0 to 6 (0-No Effect, 1-Negative, 2-Grayscale, 3-Red Tint, 4-Green Tint, 5-Blue Tint, 6-Sepia)
    s->set_whitebal(s, 1);                    // 0 = disable , 1 = enable
    s->set_awb_gain(s, 1);                    // 0 = disable , 1 = enable
    s->set_wb_mode(s, 0);                     // 0 to 4 - if awb_gain enabled (0 - Auto, 1 - Sunny, 2 - Cloudy, 3 - Office, 4 - Home)
    s->set_exposure_ctrl(s, 1);               // 0 = disable , 1 = enable
    s->set_aec2(s, 0);                        // 0 = disable , 1 = enable
    s->set_ae_level(s, 0);                    // -2 to 2
    s->set_aec_value(s, 300);                 // 0 to 1200
    s->set_gain_ctrl(s, 1);                   // 0 = disable , 1 = enable
    s->set_agc_gain(s, 0);                    // 0 to 30
    s->set_gainceiling(s, (gainceiling_t)0);  // 0 to 6
    s->set_bpc(s, 0);                         // 0 = disable , 1 = enable
    s->set_wpc(s, 1);                         // 0 = disable , 1 = enable
    s->set_raw_gma(s, 1);                     // 0 = disable , 1 = enable
    s->set_lenc(s, 1);                        // 0 = disable , 1 = enable
    s->set_hmirror(s, 0);                     // 0 = disable , 1 = enable
    s->set_vflip(s, 0);                       // 0 = disable , 1 = enable
    s->set_dcw(s, 1);                         // 0 = disable , 1 = enable
    s->set_colorbar(s, 0);                    // 0 = disable , 1 = enable

    Serial.println("Camera sensor configured for optimal quality");
  }

  camera_initialized = true;
  Serial.printf("Camera initialized successfully at %dx%d, quality=%d\n",
                config.frame_size, config.frame_size, config.jpeg_quality);
  return true;
}

void deinitCamera() {
  if (!camera_initialized) return;
  esp_camera_deinit();
  camera_initialized = false;
  Serial.println("Camera deinitialized");
}

// Method 1: WebSocket with chunked sending for large images
void sendImageViaWebSocket(camera_fb_t *fb) {
  if (!ws || ws->count() == 0) return;

  const size_t chunk_size = 1024;  // Send in 1KB chunks to avoid memory issues
  size_t remaining = fb->len;
  uint8_t *data = fb->buf;

  // Send image size first
  String header = "IMG:" + String(fb->len);
  ws->textAll(header);
  delay(10);  // Small delay to ensure header is received first

  // Send image data in chunks
  while (remaining > 0) {
    size_t to_send = (remaining > chunk_size) ? chunk_size : remaining;
    ws->binaryAll(data, to_send);
    data += to_send;
    remaining -= to_send;
    delay(1);  // Small delay between chunks
  }

  Serial.printf("Sent %d bytes via WebSocket in chunks\n", fb->len);
}

// Method 2: HTTP endpoint for direct image download (often better for high quality)
void setupImageEndpoint() {
  server->on("/capture", HTTP_GET, [](AsyncWebServerRequest *request) {
    if (!initCamera()) {
      request->send(500, "text/plain", "Camera initialization failed");
      return;
    }

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      request->send(500, "text/plain", "Camera capture failed");
      return;
    }

    // Send image with proper headers
    AsyncWebServerResponse *response = request->beginResponse_P(
      200,
      "image/jpeg",
      fb->buf,
      fb->len);

    response->addHeader("Content-Disposition", "inline; filename=capture.jpg");
    response->addHeader("Access-Control-Allow-Origin", "*");

    // Add timestamp header
    char ts[32];
    snprintf(ts, 32, "%lld.%06ld", fb->timestamp.tv_sec, fb->timestamp.tv_usec);
    response->addHeader("X-Timestamp", ts);

    request->send(response);
    esp_camera_fb_return(fb);

    Serial.printf("Sent %d byte image via HTTP\n", fb->len);
  });
}

// Method 3: MJPEG Stream endpoint (for continuous viewing)
void setupStreamEndpoint() {
  server->on("/stream", HTTP_GET, [](AsyncWebServerRequest *request) {
    if (!initCamera()) {
      request->send(500, "text/plain", "Camera initialization failed");
      return;
    }

    AsyncWebServerResponse *response = request->beginChunkedResponse(
      "multipart/x-mixed-replace; boundary=frame",
      [](uint8_t *buffer, size_t maxLen, size_t index) -> size_t {
        static camera_fb_t *fb = nullptr;
        static bool boundary_sent = false;
        static size_t fb_index = 0;

        if (!boundary_sent) {
          const char *boundary = "\r\n--frame\r\nContent-Type: image/jpeg\r\n\r\n";
          size_t boundary_len = strlen(boundary);
          if (maxLen < boundary_len) return 0;
          memcpy(buffer, boundary, boundary_len);
          boundary_sent = true;
          return boundary_len;
        }

        if (!fb) {
          fb = esp_camera_fb_get();
          if (!fb) return 0;
          fb_index = 0;
        }

        size_t remaining = fb->len - fb_index;
        size_t to_send = (remaining > maxLen) ? maxLen : remaining;

        memcpy(buffer, fb->buf + fb_index, to_send);
        fb_index += to_send;

        if (fb_index >= fb->len) {
          esp_camera_fb_return(fb);
          fb = nullptr;
          boundary_sent = false;
          delay(100);  // Control frame rate - adjust as needed
        }

        return to_send;
      });

    response->addHeader("Access-Control-Allow-Origin", "*");
    request->send(response);
  });
}

void captureTask(void *pvParameters) {
  while (true) {
    if (capture_requested) {
      capture_requested = false;

      if (!initCamera()) {
        Serial.println("Camera init failed in capture task");
        vTaskDelay(1000 / portTICK_PERIOD_MS);
        continue;
      }

      digitalWrite(LED_PIN, LOW);  // LED on during capture

      camera_fb_t *fb = esp_camera_fb_get();
      if (!fb) {
        Serial.println("Camera capture failed");
        digitalWrite(LED_PIN, HIGH);
        vTaskDelay(1000 / portTICK_PERIOD_MS);
        continue;
      }

      Serial.printf("Captured: %dx%d, %d bytes, quality=%d\n",
                    fb->width, fb->height, fb->len, config.jpeg_quality);

      // Send via WebSocket if clients connected
      if (ws && ws->count() > 0) {
        sendImageViaWebSocket(fb);
      }

      esp_camera_fb_return(fb);
      digitalWrite(LED_PIN, HIGH);  // LED off
    }

    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
}

void onWsEvent(AsyncWebSocket *server, AsyncWebSocketClient *client, AwsEventType type, void *arg, uint8_t *data, size_t len) {
  if (type == WS_EVT_CONNECT) {
    Serial.printf("WebSocket client #%u connected\n", client->id());
  } else if (type == WS_EVT_DISCONNECT) {
    Serial.printf("WebSocket client #%u disconnected\n", client->id());
  } else if (type == WS_EVT_DATA) {
    AwsFrameInfo *info = (AwsFrameInfo *)arg;
    if (info->final && info->index == 0 && info->len == len && info->opcode == WS_TEXT) {
      data[len] = 0;
      String msg = (char *)data;
      Serial.printf("Received command: %s\n", msg.c_str());

      if (msg == "capture") {
        capture_requested = true;
      } else if (msg == "wake_camera") {
        initCamera();
      } else if (msg == "sleep_camera") {
        deinitCamera();
      } else if (msg.startsWith("quality:")) {
        int quality = msg.substring(8).toInt();
        if (quality >= 0 && quality <= 63) {
          sensor_t *s = esp_camera_sensor_get();
          if (s) {
            s->set_quality(s, quality);
            Serial.printf("Quality set to %d\n", quality);
          }
        }
      } else if (msg.startsWith("framesize:")) {
        String sizeStr = msg.substring(10);
        framesize_t framesize;

        if (sizeStr == "UXGA") framesize = FRAMESIZE_UXGA;
        else if (sizeStr == "SXGA") framesize = FRAMESIZE_SXGA;
        else if (sizeStr == "XGA") framesize = FRAMESIZE_XGA;
        else if (sizeStr == "SVGA") framesize = FRAMESIZE_SVGA;
        else if (sizeStr == "VGA") framesize = FRAMESIZE_VGA;
        else return;

        sensor_t *s = esp_camera_sensor_get();
        if (s) {
          s->set_framesize(s, framesize);
          Serial.printf("Frame size set to %s\n", sizeStr.c_str());
        }
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.println("Starting ESP32-CAM High Quality Capture");

  // Setup camera configuration
  setupOptimalCamera();

  // Initialize camera early
  if (!initCamera()) {
    Serial.println("CRITICAL: Camera initialization failed!");
  }

  // WiFi connection
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    digitalWrite(LED_PIN, LOW);
    delay(500);
    digitalWrite(LED_PIN, HIGH);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi connected! IP address: ");
  Serial.println(WiFi.localIP());

  // Create web server
  server = new AsyncWebServer(80);
  ws = new AsyncWebSocket("/ws");

  ws->onEvent(onWsEvent);
  server->addHandler(ws);

  // Setup different capture methods
  setupImageEndpoint();   // HTTP GET /capture
  setupStreamEndpoint();  // HTTP GET /stream

  // Simple web interface
  server->on("/", HTTP_GET, [](AsyncWebServerRequest *request) {
    String html = R"(
<!DOCTYPE html>
<html>
<head>
    <title>ESP32-CAM High Quality Capture</title>
    <style>
        body { font-family: Arial; margin: 20px; }
        button { padding: 10px 20px; margin: 10px; font-size: 16px; }
        img { max-width: 100%; border: 1px solid #ccc; }
        .controls { margin: 20px 0; }
    </style>
</head>
<body>
    <h1>ESP32-CAM High Quality Capture</h1>
    
    <div class=\"controls\">
        <button onclick=\"captureImage()\">Capture High Quality Image</button>
        <button onclick=\"downloadImage()\">Download Latest Image</button>
        <button onclick=\"startStream()\">Start MJPEG Stream</button>
        <button onclick=\"stopStream()\">Stop Stream</button>
    </div>
    
    <div class=\"controls\">
        Quality: <select id=\"quality\" onchange=\"setQuality()\">
            <option value=\"0\">Highest (0)</option>
            <option value=\"4\">High (4)</option>
            <option value=\"6\" selected>Good (6)</option>
            <option value=\"10\">Medium (10)</option>
            <option value=\"15\">Low (15)</option>
        </select>
        
        Size: <select id=\"framesize\" onchange=\"setFrameSize()\">
            <option value=\"UXGA\" selected>UXGA (1600x1200)</option>
            <option value=\"SXGA\">SXGA (1280x1024)</option>
            <option value=\"XGA\">XGA (1024x768)</option>
            <option value=\"SVGA\">SVGA (800x600)</option>
            <option value=\"VGA\">VGA (640x480)</option>
        </select>
    </div>
    
    <div id=\"imageContainer\">
        <img id=\"capturedImage\" style=\"display:none;\">
        <img id=\"streamImage\" style=\"display:none;\">
    </div>
    
    <div id=\"status\"></div>

    <script>
        const ws = new WebSocket('ws://' + window.location.host + '/ws');
        let imageData = [];
        let expectedSize = 0;
        
        ws.onmessage = function(event) {
            if (typeof event.data === 'string' && event.data.startsWith('IMG:')) {
                expectedSize = parseInt(event.data.split(':')[1]);
                imageData = [];
                document.getElementById('status').textContent = 'Receiving image: ' + expectedSize + ' bytes...';
            } else if (event.data instanceof Blob) {
                event.data.arrayBuffer().then(buffer => {
                    imageData.push(new Uint8Array(buffer));
                    
                    let totalReceived = imageData.reduce((sum, chunk) => sum + chunk.length, 0);
                    if (totalReceived >= expectedSize) {
                        displayReceivedImage();
                    }
                });
            }
        };
        
        function displayReceivedImage() {
            let totalSize = imageData.reduce((sum, chunk) => sum + chunk.length, 0);
            let fullImage = new Uint8Array(totalSize);
            let offset = 0;
            
            for (let chunk of imageData) {
                fullImage.set(chunk, offset);
                offset += chunk.length;
            }
            
            let blob = new Blob([fullImage], {type: 'image/jpeg'});
            let url = URL.createObjectURL(blob);
            
            let img = document.getElementById('capturedImage');
            img.src = url;
            img.style.display = 'block';
            
            document.getElementById('status').textContent = 'Image received: ' + totalSize + ' bytes';
        }
        
        function captureImage() {
            ws.send('capture');
            document.getElementById('status').textContent = 'Capturing...';
        }
        
        function downloadImage() {
            window.open('/capture', '_blank');
        }
        
        function startStream() {
            let img = document.getElementById('streamImage');
            img.src = '/stream';
            img.style.display = 'block';
            document.getElementById('capturedImage').style.display = 'none';
        }
        
        function stopStream() {
            let img = document.getElementById('streamImage');
            img.src = '';
            img.style.display = 'none';
        }
        
        function setQuality() {
            let quality = document.getElementById('quality').value;
            ws.send('quality:' + quality);
        }
        
        function setFrameSize() {
            let framesize = document.getElementById('framesize').value;
            ws.send('framesize:' + framesize);
        }
    </script>
</body>
</html>
)";
    request->send(200, "text/html", html);
  });

  server->begin();
  Serial.println("Web server started");

  // Start capture task
  xTaskCreatePinnedToCore(captureTask, "CaptureTask", 4096, NULL, 1, NULL, 1);

  Serial.println("Setup complete!");
  Serial.println("Available endpoints:");
  Serial.println("  http://" + WiFi.localIP().toString() + "/ - Web interface");
  Serial.println("  http://" + WiFi.localIP().toString() + "/capture - Direct image download");
  Serial.println("  http://" + WiFi.localIP().toString() + "/stream - MJPEG stream");
  Serial.println("  ws://" + WiFi.localIP().toString() + "/ws - WebSocket control");
}

void loop() {
  if (ws) ws->cleanupClients();
  delay(1000);
}