# End-to-end demo: NS-3 IoT traffic -> flow records -> IF-DLDD drift detection on every dataset.
#   powershell -ExecutionPolicy Bypass -File run_demo.ps1 [-Run 1] [-SkipSim]
param([int]$Run = 1, [switch]$SkipSim)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force results | Out-Null
$proj = (wsl -d Ubuntu-22.04 -- wslpath -a ($PSScriptRoot -replace '\\', '/')).Trim()

if (-not $SkipSim) {
    Write-Host "[1/3] NS-3: simulating 30 WiFi IoT nodes, 5 traffic concepts (benign -> attacks)..."
    wsl -d Ubuntu-22.04 -- bash -lc "cd ~/ns-3 && ln -sf '$proj/ns3/iot-drift-sim.cc' scratch/iot-drift-sim.cc && ./ns3 build scratch/iot-drift-sim > /tmp/ns3build.log 2>&1 || { tail -20 /tmp/ns3build.log; exit 1; }; ./ns3 run --no-build 'iot-drift-sim --run=$Run --out=/tmp/packets.csv' && cp /tmp/packets.csv '$proj/results/packets.csv'"
    if ($LASTEXITCODE) { throw "NS-3 simulation failed" }
}
Write-Host "[2/3] Processing gateway packet log into flow records..."
python process_traffic.py results/packets.csv results/flows.csv
if ($LASTEXITCODE) { throw "traffic processing failed" }
Write-Host "[3/3] IF-DLDD drift detection (NS-3 IoT, CIC-IDS2017, SEA, Stagger, Sine1, Sine2)..."
python ifdldd.py
