# Projektplan

## 1. Mål

Bygga ett system som kan räkna repetitioner (reps) robust i realtid, först i simulering och sedan på riktig robot och en robot som kan bänka.

- Roboten räknar reps för minst 1 övning (t.ex. squat eller push-up).
- Roboten ska kunna bänka antingen genom att lära med VR headset eller skriva manuell kod och simulera.

## 2. Problem vi ska lösa

1. Hur roboten "ser" rörelsen (kamera/visuell signal).
2. Hur rörelse tolkas till en rep (state machine + trösklar).
3. Hur lösningen blir stabil
4. Roboten ska kunna se "barbellen

## 3. Spår

Vet inte vad vi kör på än

### Spår A: CV (Computer Vision)

### Spår B: ML (Machine Learning)

## 4. 8-veckors plan

### Vecka 1: Miljö och baseline

### Vecka 2: Data och annotation

### Vecka 3: Prototyp A (klassisk CV)

### Vecka 4: Prototyp B (pose-estimering)

### Vecka 5: Metodval och robusthet

### Vecka 6: Integration i robotpipeline

### Vecka 7: Sim-to-real och säkerhet

### Vecka 8: Slutdemo och rapport

## 5. Mätvärden (måste följas varje vecka)

- Count accuracy (% korrekta reps).
- False positives (extra reps).
- False negatives (missade reps).
- Klara av att greppa barbellen på bänkpress
