# Find the sample with a translation scan

Run 51 in March scanned the sample stage (`SARES31-GPS:TRX`) through the beam while counting fluorescence. The sample
edges are where the normalised fluorescence crosses half its maximum.

```python
--8<-- "docs/scripts/translation_scan.py:reduce"
```

```python
--8<-- "docs/scripts/translation_scan.py:edges"
```

```text
--8<-- "docs/assets/generated/translation_edges.txt"
```

![Translation scan](../assets/generated/translation_scan.svg)

The synthetic sample spans −0.9 to −0.3 mm. The same code works for any 1-D scan coordinate.
