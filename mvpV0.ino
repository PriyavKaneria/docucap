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

// Thresholds
const size_t WS_MAX_SAFE_SIZE = 200000; // if fb->len <= this, send via websocket in one frame

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

  if (psramFound()) {
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.frame_size = FRAMESIZE_SVGA;  // reasonable default
    config.jpeg_quality = 8;             // 0..63, lower is better quality
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
  } else {
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }
}

bool initCamera() {
  if (camera_initialized) return true;
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return false;
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);
    s->set_special_effect(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_wb_mode(s, 0);
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s, 0);
    s->set_ae_level(s, 0);
    s->set_aec_value(s, 300);
    s->set_gain_ctrl(s, 1);
    s->set_agc_gain(s, 0);
    s->set_gainceiling(s, (gainceiling_t)0);
    s->set_bpc(s, 0);
    s->set_wpc(s, 1);
    s->set_raw_gma(s, 1);
    s->set_lenc(s, 1);
    s->set_hmirror(s, 0);
    s->set_vflip(s, 0);
    s->set_dcw(s, 1);
    s->set_colorbar(s, 0);
  }

  camera_initialized = true;
  Serial.printf("Camera initialized successfully (framesize=%d, quality=%d)\n", config.frame_size, config.jpeg_quality);
  return true;
}

void deinitCamera() {
  if (!camera_initialized) return;
  esp_camera_deinit();
  camera_initialized = false;
  Serial.println("Camera deinitialized");
}

// Send image over websocket in one binary frame (if small enough)
void sendImageViaWebSocketSingle(camera_fb_t *fb) {
  if (!ws || ws->count() == 0) return;

  // Single binary frame
  ws->binaryAll(fb->buf, fb->len);
  Serial.printf("Sent %d bytes via WebSocket (single frame)\n", fb->len);
}

// Fallback: if too large, instruct client to fetch via HTTP GET /capture?ts=...
void sendImageViaHTTPNotify(camera_fb_t *fb) {
  if (!ws || ws->count() == 0) return;
  // Create a short notification with a timestamped URL
  String url = "/capture?ts=" + String(millis());
  ws->textAll("HTTP:" + url);
  Serial.printf("Notified clients to fetch via HTTP: %s (size=%d)\n", url.c_str(), fb->len);
}

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

    AsyncWebServerResponse *response = request->beginResponse_P(
      200,
      "image/jpeg",
      fb->buf,
      fb->len);

    response->addHeader("Content-Disposition", "inline; filename=capture.jpg");
    response->addHeader("Access-Control-Allow-Origin", "*");

    // Add timestamp header
    char ts[32];
    snprintf(ts, 32, "%llu", (unsigned long long)millis());
    response->addHeader("X-Timestamp", ts);

    request->send(response);
    esp_camera_fb_return(fb);

    Serial.printf("Sent %d byte image via HTTP\n", fb->len);
  });
}

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
        static size_t fb_index = 0;
        static bool boundary_sent = false;

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
          // small delay to control frame rate
          delay(100);
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

      // Choose transport based on size
      if (fb->len <= WS_MAX_SAFE_SIZE) {
        sendImageViaWebSocketSingle(fb);
      } else {
        // large image -> instruct client to fetch via HTTP
        sendImageViaHTTPNotify(fb);
      }

      esp_camera_fb_return(fb);
      digitalWrite(LED_PIN, HIGH);  // LED off
    }

    vTaskDelay(50 / portTICK_PERIOD_MS);
  }
}

void handleWsCommand(const String &msg) {
  if (msg == "init") {
    initCamera();
  } else if (msg == "capture") {
    capture_requested = true;
  } else if (msg == "deinit") {
    deinitCamera();
  } else if (msg == "sleep") {
    // For safety: deinit camera and disconnect WiFi for low power.
    // Deep sleep would require wake source planning; so we do software sleep.
    deinitCamera();
    WiFi.disconnect(true);
    WiFi.mode(WIFI_OFF);
    Serial.println("Entered low-power software sleep: camera deinitialized, WiFi off");
  } else if (msg.startsWith("quality:")) {
    int quality = msg.substring(8).toInt();
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
      s->set_quality(s, quality);
      Serial.printf("Quality set to %d\n", quality);
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

void onWsEvent(AsyncWebSocket *serverWs, AsyncWebSocketClient *client, AwsEventType type, void *arg, uint8_t *data, size_t len) {
  if (type == WS_EVT_CONNECT) {
    Serial.printf("WebSocket client #%u connected\n", client->id());
  } else if (type == WS_EVT_DISCONNECT) {
    Serial.printf("WebSocket client #%u disconnected\n", client->id());
  } else if (type == WS_EVT_DATA) {
    AwsFrameInfo *info = (AwsFrameInfo *)arg;
    if (info->final && info->index == 0 && info->len == len && info->opcode == WS_TEXT) {
      // Text message command
      String msg = String((char *)data);
      Serial.printf("Received command: %s\n", msg.c_str());
      handleWsCommand(msg);
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH); // LED off (active low)

  Serial.println("Starting ESP32-CAM Optimized Server");

  setupOptimalCamera();

  // WiFi connection
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    digitalWrite(LED_PIN, LOW);
    delay(200);
    digitalWrite(LED_PIN, HIGH);
    delay(200);
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

  setupImageEndpoint();
  setupStreamEndpoint();

  // Serve improved client HTML/JS
  server->on("/", HTTP_GET, [](AsyncWebServerRequest *request) {
    String html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>ESP32-CAM Optimized Interface</title>
  <style>body{font-family:Arial;margin:10px;}button{margin:4px;padding:8px}</style>
</head>
<body>
  <h2>ESP32-CAM Optimized</h2>
  <div>
    <button onclick="wsSend('init')">Init Camera</button>
    <button onclick="wsSend('capture')">Capture</button>
    <button onclick="wsSend('deinit')">Deinit Camera</button>
    <button onclick="wsSend('sleep')">Low-Power Sleep (soft)</button>
  </div>

  <div style="margin-top:8px">
    Quality: <select id="quality" onchange="setQuality()">
      <option value="4">4 (high)</option>
      <option value="6" selected>6 (good)</option>
      <option value="10">10 (medium)</option>
      <option value="15">15 (low)</option>
    </select>

    Size: <select id="framesize" onchange="setFrameSize()">
      <option value="UXGA">UXGA</option>
      <option value="SXGA">SXGA</option>
      <option value="XGA">XGA</option>
      <option value="SVGA" selected>SVGA</option>
      <option value="VGA">VGA</option>
    </select>
  </div>

  <div id="status" style="margin-top:8px">Status: idle</div>
  <div style="margin-top:8px"><img id="img" style="max-width:100%;border:1px solid #ccc;display:block"></div>
  <div id="timing"></div>

<script>
  const ws = new WebSocket('ws://' + window.location.host + '/ws');
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => { document.getElementById('status').innerText = 'WS connected'; };
  ws.onclose = () => { document.getElementById('status').innerText = 'WS closed'; };
  ws.onerror = (e) => { document.getElementById('status').innerText = 'WS error'; };

  let lastCaptureStart = 0;

  ws.onmessage = function(event) {
    if (typeof event.data === 'string') {
      const txt = event.data;
      if (txt.startsWith('HTTP:')) {
        // server asks us to fetch via HTTP
        const url = txt.substring(5);
        fetchImage(url);
      } else {
        document.getElementById('status').innerText = 'MSG: ' + txt;
      }
      return;
    }

    // Binary ArrayBuffer received -> complete image (we send single binary frame)
    const receiveTime = performance.now();
    const delta = (receiveTime - lastCaptureStart).toFixed(1);

    const arrayBuffer = event.data;
    const blob = new Blob([arrayBuffer], {type: 'image/jpeg'});
    const url = URL.createObjectURL(blob);
    const img = document.getElementById('img');
    img.onload = () => { URL.revokeObjectURL(url); };
    img.src = url;

    document.getElementById('status').innerText = 'Image received: ' + blob.size + ' bytes';
    document.getElementById('timing').innerText = 'Request→Complete: ' + delta + ' ms';
  };

  function fetchImage(path) {
    lastCaptureStart = performance.now();
    document.getElementById('status').innerText = 'Fetching via HTTP: ' + path;
    fetch(path).then(resp => resp.blob()).then(blob => {
      const receiveTime = performance.now();
      const delta = (receiveTime - lastCaptureStart).toFixed(1);
      const url = URL.createObjectURL(blob);
      const img = document.getElementById('img');
      img.onload = () => { URL.revokeObjectURL(url); };
      img.src = url;
      document.getElementById('status').innerText = 'Image fetched HTTP: ' + blob.size + ' bytes';
      document.getElementById('timing').innerText = 'Request→Complete: ' + delta + ' ms';
    }).catch(err => {
      document.getElementById('status').innerText = 'HTTP fetch failed';
    });
  }

  function wsSend(msg) {
    if (ws.readyState === WebSocket.OPEN) {
      if (msg === 'capture') {
        lastCaptureStart = performance.now();
      }
      ws.send(msg);
    } else {
      alert('WebSocket not open');
    }
  }

  function setQuality() {
    const q = document.getElementById('quality').value;
    wsSend('quality:' + q);
  }

  function setFrameSize() {
    const s = document.getElementById('framesize').value;
    wsSend('framesize:' + s);
  }
</script>
</body>
</html>
)rawliteral";
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
