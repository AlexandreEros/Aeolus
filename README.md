Run
```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass                                
venv/Scripts/Activate.ps1 
pip install -r requirements.txt
pip install -e ".[dev]"
```
Spherical harmonics backends:
- `LatLonSphericalHarmonics`: structured latitude-longitude baseline
- `PointSetSphericalHarmonics`: generic matrix engine for arbitrary sample points
- `GeodesicSphericalHarmonics`: weight-aware geodesic-grid wrapper around the point-set engine

Example:
```
psx-bve --duration-days 100 --day-hours 24 --dt-snapshots 1728000
psx-bve --viscosity 1e5 --duration-days 10 --day-hours 24 --dt-snapshots 48
```
