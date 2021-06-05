# Maschinelle Übersetzung

## Abgabetermine

| Thema | Deadline |Abgabe commitNr| geschafft |
| - | ----- | - | - |
|Einführung und Metriken|26.04|2e6ca3e0|:heavy_check_mark:|
|Vokabular und Subword Units|10.05|03e07168908f743b9b72da835122fcd4ef25f26d|:heavy_check_mark:|
|Feed Forward Netzwerke|31.05|17e7dd51f2253cb3a2d152ed8534fad1263850f3|:heavy_check_mark:|
|Suche|14.06|-|-|
|Rekurrente Neuronale Netzwerke|-|-|-|
|Blatt 6|-|-|-|

## Testattermin

Uhrzeit: 14:00-15:00

## Vorlesungtermine 

|Datum  |Thema    	|
|-------|------   	|
|13.04  |Metriken 	|  
|27.04  |Vokabular und Subword Units|
|04.05  |Grundlagen Neuronale Netze und Tensorflow|
|11.05  |Feedforward Modell|
|01.06  |Suche  |	
|15.06  |		| 
|06.07  |		|
|20.07  |		|


## Aufgabenblatt 1 - Metriken 

In dieser Übungsblatt geht es darum die Metriken WER, PER und BLEU zu verstehen
Ein Programm zu schreiben welches diese für eine gegebene Menge von Referenz-Hypothese Paaren berechnet. 
Gemeint ist damit die Evaluation von Hypothesen durch den Vergleich mit einer oder mehreren Referenzenübersetzungen.

## Aufgabenblatt 2 - Vokabular und Subword Units

Auf dem zweiten Blatt geht es um die Implementierung des byte pair encoding algorithmus. Wir sollen ein Programm schreiben, das Folgendes kann:
    - anhand einer gegebenen Anzahl von Zusammenzug-Operationen sowie Trainingsdaten eine BPE Zerlegung zu lernen.
    - erlernte BPE Operationen auf Text anzuwenden.
    - die BPE Zerlegung rückgängig zu machen.
Darüber hinaus erstellen wir eine Klasse Dictionary, die verwendet wird, um das Vokabular aus einem gegebenen Text/einer gegebenen Datei zu behalten. Der letzte Teil dieses Blattes beschäftigt sich mit der Weiterverarbeitung von Daten - dazu soll eine Funktion zur Erzeugung von "Batches" entwickelt werden.

- [x] BPE
- [x] Klasse Dictionary
- [x] Funktion Batch

## Aufgabenblatt 3 - Feed Forward Netzwerke

In diesem Übungsblatt besteht die Aufgabe darin, erste Features des Feed-Forward-Übersetzungsmodells einzubauen. Der Code soll um neue Funktionalitäten erweitert werden, die für den Aufbau eines Feed-Forward-Netzwerks grundlegend sind. Hierfür steht die folgende Liste zur Verfügung. Für die Implementierung wird die Python-Interface von Tensorflow verwendet. Weiterhin sollte die Implementierung die Anforderungen (für das NN) auf Seite 36 in den Folien berücksichtigen. 

- [x] Speicherung von Batches verwerfen. 
- [x] Zusammenzugsoperationen in csv speichern
- [x] Implementierung von SGD. (Stochastik Gradient Descent)
- [x] Modelle nach Userauswahl speichern.
- [x] Metriken wie Accurracy und Perplexity in regelmäßigen Abständen ausgeben
- [x] Auf einem gegebenen Entwicklungsdaten alle n Updates das Modell auswerten
- [ ] Netzwerkstruktur deutlich machen (details im Aufgabenblatt + Tensorboard)
- [x] BPE mit 7k Zusammenzugoperationen testen, um den niedrigsten Perplexity Wert den des Modells Entwicklungsdaten herauszufinden. 

## Aufgabenblatt 4 

In diesem Übungsblatt besteht die Aufgabe darin, das Programm in die Lage zu versetzen, sinnvolle Modellteile (z. B. ein Subwortzerlegungsmodell oder ein Vokabular) zu speichern und zu laden. Zentrale Hyperparameter sollen nicht im Programmcode vorgegeben werden, sondern vom Benutzer eingestellt werden können. Als Trainingsdaten werden die Dateien multi30k.de.gz und multi30k.en.gz bereitgestellt. Für die Entwicklung werden multi30k.dev.de und multi30k.dev.de verwendet. 

- [ ] Score des Feedforward Modells für jedes Satzpaar ausgeben
- [ ] Übersetzung aus Grundlage der Vorlesung erstellen, indem ein trainiertes Modell geladen wird und die beide Methoden greedy search sowie beam search durchführen. 
- [ ] Mithilfe einer automatisierte Methode (Python oder Bash) den BLEU Wert von jedem Checkpoint bestimmen. 
- [ ] BLEU Score bei verschiedene Beamgrößen {1, 5, 10, 50} testen.
- [ ] Drei Verschiedene durchläufe mit unterschiedlichen Random Seed auswerten. BLEU und Perplexity Werte auf den Entwicklungsdaten über die Checkpoints plotten.
- [ ] Vortragsthema aussuchen! 
