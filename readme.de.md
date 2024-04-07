# Thema und Hintergrund
Der Hauptzweck dieses Projekts besteht darin, eine Anwendung zu implementieren, die Github-Tools ausgeben kann, die den Benutzereingaben entsprechen und bei der Bewältigung der aktuellen Aufgabe helfen.

Wir alle wissen, dass das mittlerweile ChatGPT eine starkes Wissen in verschiedenen Bereichen aufweist. Vor allem in Kombination mit der Bing-Suche ist es in der Lage, bei Bedarf direkt entsprechende Websites zu liefern. Aber ChatGPT hat einige Einschränkungen:
1. ChatGPT ist auf Daten vor September 2021 trainiert, daher gibt es keine Möglichkeit, Informationen nach diesem Datum bereitzustellen.
2. ChatGPT hat keine Möglichkeit, komplexe logische Zusammenhänge zu analysieren.
3. ChatGPT kann keine zitierten Quellen auflisten und seine Zuverlässigkeit basiert auf der Zuverlässigkeit der Quelleninformationen, die von Natur aus falsch, inkonsistent oder falsch oder widersprüchlich sein können, nachdem sie von ChatGPT kombiniert wurden.

# Idee
Um die Probleme von ChatGPT zu lösen, wie z.B. die mangelnde Fähigkeit, komplexe Eingaben zu analysieren, komplexe Antworten, die Unfähigkeit, Echtzeit-Tools bereitzustellen, und mögliche Fehler in den bereitgestellten Links. Unsere Kernideen ist:
1. Die Anforderungen zu zerlegen und jeweils nur einfache Fragen an ChatGPT zu stellen
2. Die Ausgabe einzuschränken, damit sie kurz und themenbezogen ist
3. Verwendung der Github-API, um die neuesten Github-Daten zu erhalten, um die Popularität und Effektivität des Tools zu gewährleisten

# Ausführung
```
python sdsc_cataglog.py
```

# Beispiel für die Ausgabe zum Thema Sicherheit beim Radfahren
<img src="sdsc-cataglog/images/output_example.png" alt="output example" width="800" height="600">
