#include <esp_camera.h>
#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "esp_sleep.h"

// --- Configuration ---
const char *ssid = "Digi_Nova";
const char *password = "diginova";

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// --- Pins ---
#define LED_PIN 33          // active low indicator (same as before)
#define WAKE_BUTTON_PIN 14  // external RTC GPIO for wake (use a pull-up button to GND)

// --- System ---
AsyncWebServer *server = nullptr;
AsyncWebSocket *ws = nullptr;

volatile bool capture_requested = false;
volatile bool test_mode_active = false;
volatile uint32_t test_interval_ms = 1000;  // ms between captures in test

// camera config
camera_config_t config;
bool camera_initialized = false;

// thresholds
const size_t WS_MAX_SAFE_SIZE = 120000;  // send via WS if <= this size

TaskHandle_t testTaskHandle = NULL;

// --- Camera setup & helpers ---
void setupOptimalCameraDefaults() {
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
    config.frame_size = FRAMESIZE_SVGA;  // default
    config.jpeg_quality = 8;
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
    Serial.printf("Camera init failed 0x%x\n", err);
    return false;
  }
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);
    s->set_special_effect(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_aec_value(s, 300);
    s->set_gain_ctrl(s, 1);
    s->set_wpc(s, 1);
    s->set_raw_gma(s, 1);
    s->set_lenc(s, 1);
    s->set_dcw(s, 1);
    s->set_colorbar(s, 0);
  }

  camera_initialized = true;
  Serial.printf("Camera initialized (framesize=%d quality=%d)\n", config.frame_size, config.jpeg_quality);
  return true;
}

void deinitCamera() {
  if (!camera_initialized) return;
  esp_camera_deinit();
  camera_initialized = false;
  Serial.println("Camera deinitialized");
}

// --- Transport ---
void sendImageViaWebSocketSingle(camera_fb_t *fb) {
  if (!ws || ws->count() == 0) return;
  // send single binary frame
  ws->binaryAll(fb->buf, fb->len);
  Serial.printf("WS single send %d bytes\n", fb->len);
}

void sendImageViaHTTPNotify(camera_fb_t *fb) {
  if (!ws || ws->count() == 0) return;
  String url = "/capture?ts=" + String(millis());
  ws->textAll("HTTP:" + url);
  Serial.printf("Notify HTTP %s size=%d\n", url.c_str(), fb->len);
}

// --- Endpoints ---
void setupImageEndpoint(AsyncWebServer *server) {
  server->on("/capture", HTTP_GET, [](AsyncWebServerRequest *request) {
    if (!initCamera()) {
      request->send(500, "text/plain", "Camera init failed");
      return;
    }
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      request->send(500, "text/plain", "Capture failed");
      return;
    }

    AsyncWebServerResponse *response = request->beginResponse("image/jpeg", fb->len, [fb](uint8_t *buffer, size_t maxLen, size_t index) -> size_t {
      // simple streaming wrapper — return whole buffer at once
      if (index == 0) {
        size_t toCopy = fb->len;
        memcpy(buffer, fb->buf, toCopy);
        return toCopy;
      }
      return 0;
    });

    response->addHeader("Content-Disposition", "inline; filename=capture.jpg");
    response->addHeader("Access-Control-Allow-Origin", "*");
    request->send(response);
    esp_camera_fb_return(fb);
    Serial.printf("HTTP /capture sent %d bytes\n", fb->len);
  });
}

void setupStreamEndpoint(AsyncWebServer *server) {
  server->on("/stream", HTTP_GET, [](AsyncWebServerRequest *request) {
    if (!initCamera()) {
      request->send(500, "text/plain", "Camera init failed");
      return;
    }
    AsyncWebServerResponse *response = request->beginChunkedResponse("multipart/x-mixed-replace; boundary=frame", [](uint8_t *buffer, size_t maxLen, size_t index) -> size_t {
      static camera_fb_t *fb = nullptr;
      static size_t fb_index = 0;
      static bool boundary_sent = false;

      if (!boundary_sent && buffer != NULL) {
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
      if (buffer != NULL) {
        memcpy(buffer, fb->buf + fb_index, to_send);
      }
      fb_index += to_send;

      if (fb_index >= fb->len) {
        esp_camera_fb_return(fb);
        fb = nullptr;
        boundary_sent = false;
        delay(100);  // control fps
      }
      return to_send;
    });
    response->addHeader("Access-Control-Allow-Origin", "*");
    request->send(response);
  });
}

// --- Capture task ---
void captureOnce() {
  if (!initCamera()) return;
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("fb null");
    return;
  }

  Serial.printf("Captured %dx%d %d bytes\n", fb->width, fb->height, fb->len);
  if (fb->len <= WS_MAX_SAFE_SIZE) sendImageViaWebSocketSingle(fb);
  else sendImageViaHTTPNotify(fb);

  esp_camera_fb_return(fb);
}

void captureTask(void *pvParameters) {
  while (true) {
    if (capture_requested) {
      capture_requested = false;
      digitalWrite(LED_PIN, LOW);
      captureOnce();
      digitalWrite(LED_PIN, HIGH);
    }
    vTaskDelay(20 / portTICK_PERIOD_MS);
  }
}

// --- Test mode task (captures repeatedly at requested FPS for duration) ---
void testModeTask(void *pv) {
  uint32_t duration_ms = 5000;  // 5s default
  uint32_t interval = test_interval_ms;
  uint32_t start = millis();
  while (millis() - start < duration_ms && test_mode_active) {
    digitalWrite(LED_PIN, LOW);
    captureOnce();
    digitalWrite(LED_PIN, HIGH);
    vTaskDelay(interval / portTICK_PERIOD_MS);
  }
  test_mode_active = false;
  vTaskDelete(NULL);
}

// --- Websocket handlers ---
void handleWsCommand(const String &msg) {
  if (msg == "init") initCamera();
  else if (msg == "capture") {
    capture_requested = true;
  } else if (msg == "deinit") deinitCamera();
  else if (msg == "sleep_for:") {
    // not used
  } else if (msg.startsWith("sleep_for:")) {
    // format sleep_for:seconds
    int secs = msg.substring(10).toInt();
    if (secs <= 0) secs = 10;
    deinitCamera();
    WiFi.disconnect(true);
    WiFi.mode(WIFI_OFF);
    Serial.printf("Sleeping for %d seconds\n", secs);
    // enable ext0 wake on wake button (active LOW)
    esp_sleep_enable_ext0_wakeup((gpio_num_t)WAKE_BUTTON_PIN, 0);
    esp_sleep_enable_timer_wakeup((uint64_t)secs * 1000000ULL);
    delay(50);
    Serial.println("Going to deep sleep now");
    esp_deep_sleep_start();
  } else if (msg.startsWith("quality:")) {
    int q = msg.substring(8).toInt();
    sensor_t *s = esp_camera_sensor_get();
    if (s) s->set_quality(s, q);
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
    if (s) s->set_framesize(s, framesize);
  } else if (msg.startsWith("start_test:")) {
    int fps = msg.substring(11).toInt();
    if (fps < 1) fps = 1;
    if (fps > 30) fps = 30;
    test_interval_ms = 1000 / fps;
    if (!test_mode_active) {
      test_mode_active = true;
      xTaskCreatePinnedToCore(testModeTask, "TestMode", 4096, NULL, 1, &testTaskHandle, 1);
    }
  } else if (msg == "stop_test") {
    test_mode_active = false;
    if (testTaskHandle) {
      vTaskDelete(testTaskHandle);
      testTaskHandle = NULL;
    }
  }
}

void onWsEvent(AsyncWebSocket *serverWs, AsyncWebSocketClient *client, AwsEventType type, void *arg, uint8_t *data, size_t len) {
  if (type == WS_EVT_CONNECT) Serial.printf("WS Client #%u connected\n", client->id());
  else if (type == WS_EVT_DISCONNECT) Serial.printf("WS Client #%u disconnected\n", client->id());
  else if (type == WS_EVT_DATA) {
    AwsFrameInfo *info = (AwsFrameInfo *)arg;
    if (info->opcode == WS_TEXT) {
      String msg = String((char *)data);
      handleWsCommand(msg);
    }
  }
}

// --- Setup & loop ---
void setup() {
  Serial.begin(115200);
  Serial.println();
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);
  pinMode(WAKE_BUTTON_PIN, INPUT_PULLUP);  // button to GND

  setupOptimalCameraDefaults();

  // Check wake reason
  esp_sleep_wakeup_cause_t cause = esp_sleep_get_wakeup_cause();
  if (cause == ESP_SLEEP_WAKEUP_TIMER) Serial.println("Woke from deep sleep (timer)");
  else if (cause == ESP_SLEEP_WAKEUP_EXT0) Serial.println("Woke from deep sleep (ext0 button)");
  else Serial.println("Normal boot or other wake reason");

  // Connect WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 10000) {
    Serial.print('.');
    delay(300);
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) Serial.printf("WiFi %s IP=%s\n", ssid, WiFi.localIP().toString().c_str());
  else Serial.println("WiFi not connected (continuing)\n");

  // Web server + WS
  server = new AsyncWebServer(80);
  ws = new AsyncWebSocket("/ws");
  ws->onEvent(onWsEvent);
  server->addHandler(ws);

  setupImageEndpoint(server);
  setupStreamEndpoint(server);

  // Serve client UI
  server->on("/", HTTP_GET, [](AsyncWebServerRequest *request) {
    const char *html = R"rawliteral(
<!doctype html>
<html>
<head><meta charset="utf-8"><title>ESP32-CAM DeepSleep Test</title>
<style>body{font-family:Arial;margin:10px}button{margin:4px;padding:8px}</style>
</head>
<body>
  <h3>ESP32-CAM DeepSleep & Test</h3>
  <div>
    <button onclick="wsSend('init')">Init Camera</button>
    <button onclick="wsSend('capture'); startCaptureTimer();">Capture</button>
    <button onclick="wsSend('deinit')">Deinit Camera</button>
    <button onclick="wsSend('sleep_for:10')">Sleep 10s (deep)</button>
  </div>
  <div style="margin-top:8px">
    Quality: <select id="quality" onchange="setQuality()"><option value="4">4</option><option value="6" selected>6</option><option value="10">10</option></select>
    Size: <select id="framesize" onchange="setFrame()"><option value="SVGA" selected>SVGA</option><option value="VGA">VGA</option></select>
  </div>

  <div style="margin-top:8px">
    <label>Test FPS: <input id="fps" type="number" value="1" min="1" max="30" style="width:60px"></label>
    <button onclick="startTest()">Start 5s Test</button>
    <button onclick="stopTest()">Stop Test</button>
  </div>

  <div id="status">Status: idle</div>
  <div id="timing">Avg: -</div>
  <div><img id="img" style="max-width:100%;border:1px solid #ccc;display:block"></div>

<script>
  const ws = new WebSocket('ws://' + window.location.host + '/ws');
  ws.binaryType = 'arraybuffer';
  let lastStart = 0;
  let samples = [];

  ws.onopen = () => document.getElementById('status').innerText = 'WS connected';
  ws.onclose = () => document.getElementById('status').innerText = 'WS closed';
  ws.onerror = () => document.getElementById('status').innerText = 'WS error';

  ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
      const txt = event.data;
      if (txt.startsWith('HTTP:')) { fetchImage(txt.substring(5)); return; }
      document.getElementById('status').innerText = txt;
      return;
    }

    const now = performance.now();
    const delta = now - lastStart;
    samples.push(delta);
    const blob = new Blob([event.data], {type:'image/jpeg'});
    const url = URL.createObjectURL(blob);
    const img = document.getElementById('img');
    img.onload = () => URL.revokeObjectURL(url);
    img.src = url;
    document.getElementById('status').innerText = 'Image received ' + blob.size + ' bytes';
    updateAvg();
  };

  function fetchImage(path) {
    lastStart = performance.now();
    document.getElementById('status').innerText = 'HTTP fetch ' + path;
    fetch(path).then(r=>r.blob()).then(blob=>{
      const now = performance.now(); samples.push(now - lastStart);
      const url = URL.createObjectURL(blob); const img = document.getElementById('img'); img.onload = ()=>URL.revokeObjectURL(url); img.src = url;
      document.getElementById('status').innerText = 'HTTP image ' + blob.size + ' bytes'; updateAvg();
    }).catch(()=>{document.getElementById('status').innerText='HTTP fail';});
  }

  function wsSend(msg) { if (ws.readyState===WebSocket.OPEN) { if (msg==='capture') lastStart=performance.now(); ws.send(msg);} else alert('WS not open'); }
  function setQuality(){ wsSend('quality:'+document.getElementById('quality').value); }
  function setFrame(){ wsSend('framesize:'+document.getElementById('framesize').value); }

  // Test controls
  let testTimer = null;
  function startTest(){ samples=[]; const fps = parseInt(document.getElementById('fps').value)||1; const interval = 1000/Math.min(30, Math.max(1,fps)); wsSend('start_test:'+fps); document.getElementById('status').innerText='Test started';
    // client-side capture rate measure: also send periodic capture commands (server runs its own loop too)
    lastStart = performance.now();
    testTimer = setInterval(()=>{ lastStart = performance.now(); wsSend('capture'); }, interval);
    // stop after 5s
    setTimeout(()=>{ stopTest(); }, 5000);
  }
  function stopTest(){ wsSend('stop_test'); if (testTimer) clearInterval(testTimer); testTimer=null; document.getElementById('status').innerText='Test stopped'; updateAvg(); }

  function updateAvg(){ if (samples.length===0) return; const sum = samples.reduce((a,b)=>a+b,0); const avg = (sum/samples.length).toFixed(1); document.getElementById('timing').innerText = 'Avg over ' + samples.length + ' = ' + avg + ' ms'; }

  function startCaptureTimer(){ lastStart = performance.now(); wsSend('capture'); }
</script>
</body>
</html>
)rawliteral";
    request->send(200, "text/html", html);
  });

  server->begin();
  Serial.println("Server started");

  // start capture task
  xTaskCreatePinnedToCore(captureTask, "CaptureTask", 4096, NULL, 1, NULL, 1);
}

void loop() {
  if (ws) ws->cleanupClients();
  delay(1000);
}
