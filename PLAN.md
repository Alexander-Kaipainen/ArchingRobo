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

## 5. 8-veckors plan

### Vecka 1: Miljö och baseline

- Sätt upp repo, dependencies och körbar demo i MuJoCo/PyBullet.
- Definiera en första övning (squat eller push-up).
- Logga video + tidsstämplar + "ground truth" rep-count för några sekvenser.

### Vecka 2: Data och annotation

- Samla 20-40 korta sekvenser med variation (vinkel, hastighet, avstånd).
- Märk start/slut på reps i en enkel annoteringsfil (CSV/JSON).
- Skapa eval-script som räknar precision/recall och total count error.

### Vecka 3: Prototyp A (klassisk CV)

- Bygg första pipeline för rörelseanalys.
- Implementera enkel state machine: up -> down -> up = 1 rep.
- Mät stabilitet och latency.

### Vecka 4: Prototyp B (pose-estimering)

- Testa keypoints + vinkeltrösklar för samma övning.
- Kör exakt samma testdata och jämför mot Prototyp A.
- Dokumentera för-/nackdelar.

### Vecka 5: Metodval och robusthet

- Välj huvudspår baserat på mätresultat.
- Lägg till filtrering (smoothing, hysteresis, debounce).
- Testa edge cases: halva reps, pauser, snabba reps.

### Vecka 6: Integration i robotpipeline

- Integrera rep-räknare i er befintliga kodstruktur.
- Lägg till tydliga API-anrop, t.ex. `count_reps(frame) -> state, total`.
- Säkerställ att det funkar i realtid i sim-loop.

### Vecka 7: Sim-to-real och säkerhet

- Test på riktig kamera/robot.
- Kalibrera kameravinkel, ROI och trösklar.
- Lägg fallback-regler om vision tappar signal.

### Vecka 8: Slutdemo och rapport

- Kör sluttest med fasta scenarier.
- Sammanställ resultat: noggrannhet, latency, feltyper.
- Förbered demo + teknisk presentation.

## 6. Roller i teamet (förslag)

- Person 1: Sim/robotintegration.
- Person 2: CV/pose-modell.
- Person 3: Data/annotation + test/evaluering.
- Person 4: Systemarkitektur + demo/rapport.

## 7. Mätvärden (måste följas varje vecka)

- Count accuracy (% korrekta reps).
- False positives (extra reps).
- False negatives (missade reps).
- Latency per frame och total FPS.

## 8. Risker och motåtgärder

- Risk: för lite data -> Motåtgärd: snabb datainsamling varje vecka.
- Risk: metod blir för tung -> Motåtgärd: profilering och enklare modell.
- Risk: sim funkar men inte verklighet -> Motåtgärd: tidig test på riktig kamera.

## 9. Nästa konkreta steg (denna vecka)

1. Välj första övning att räkna reps på.
2. Spela in 10 korta sekvenser och märk reps manuellt.
3. Bygg första eval-script för "pred vs truth".
4. Prototypa en enkel baseline (antingen klassisk CV eller pose) på materialet.
