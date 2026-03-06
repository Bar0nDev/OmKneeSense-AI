#include <HX711.h>

// ---------------- Pins ----------------
static const int PIN_DOUT = 4;
static const int PIN_SCK  = 5;

// ---------------- Calibration ----------------
volatile float CAL_FACTOR = 30.0f;  // temporary start value
static const int TARE_SAMPLES = 12;

// ---------------- Filtering ----------------
static const float EMA_ALPHA = 0.20f;

HX711 scale;

// median-of-3
static inline long median3(long a, long b, long c) {
  if (a > b) { long t=a; a=b; b=t; }
  if (b > c) { long t=b; b=c; c=t; }
  if (a > b) { long t=a; a=b; b=t; }
  return b;
}

// State
long r0=0, r1=0, r2=0;
bool primed = false;

float ema_g = 0.0f;
bool ema_init = false;

void setup() {
  Serial.begin(115200);
  delay(200);

  scale.begin(PIN_DOUT, PIN_SCK);
  scale.set_scale(CAL_FACTOR);

  delay(300);
  scale.tare(TARE_SAMPLES);
}

void loop() {

  if (!scale.is_ready()) return;

  long raw = scale.read();

  // ---- Median filter ----
  if (!primed) {
    r0 = raw; r1 = raw; r2 = raw;
    primed = true;
  } else {
    r0 = r1; r1 = r2; r2 = raw;
  }

  long raw_med = median3(r0, r1, r2);

  // ---- Convert to grams ----
  long net = raw_med - scale.get_offset();
  float g = (float)net / CAL_FACTOR;

  // ---- EMA ----
  if (!ema_init) {
    ema_g = g;
    ema_init = true;
  } else {
    ema_g = (1.0f - EMA_ALPHA) * ema_g + EMA_ALPHA * g;
  }

  // ---- Stream ONLY numeric values (Python friendly) ----
  Serial.print(raw_med);
  Serial.print(",");
  Serial.println(ema_g, 3);
}
