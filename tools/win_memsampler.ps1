# memsampler.ps1 <csv> <phasefile> <stopfile> : every 2 s, engine / api process + system + GPU memory
param([string]$Csv, [string]$PhaseFile, [string]$StopFile)
"t,phase,eng_ws,eng_priv,eng_peakws,api_ws,api_priv,avail,standby,modified,commit,gpu_ded,gpu_shared" | Out-File -Encoding ascii $Csv
$t0 = Get-Date
while (-not (Test-Path $StopFile)) {
  $ph = if (Test-Path $PhaseFile) { (Get-Content $PhaseFile -Raw).Trim() } else { "?" }
  $e = Get-Process qwenox-engine-win -ErrorAction SilentlyContinue | Select-Object -First 1
  $a = Get-Process qwenox-win -ErrorAction SilentlyContinue | Select-Object -First 1
  $m = (Get-Counter '\Memory\Available Bytes','\Memory\Standby Cache Normal Priority Bytes','\Memory\Standby Cache Reserve Bytes','\Memory\Standby Cache Core Bytes','\Memory\Modified Page List Bytes','\Memory\Committed Bytes' -ErrorAction SilentlyContinue).CounterSamples
  $av = $m[0].CookedValue; $sb = $m[1].CookedValue + $m[2].CookedValue + $m[3].CookedValue; $mod = $m[4].CookedValue; $cm = $m[5].CookedValue
  $gd = 0; $gs = 0
  if ($e) {
    $g = (Get-Counter "\GPU Process Memory(pid_$($e.Id)*)\Dedicated Usage","\GPU Process Memory(pid_$($e.Id)*)\Shared Usage" -ErrorAction SilentlyContinue).CounterSamples
    foreach ($s in $g) { if ($s.Path -like '*dedicated*') { $gd += $s.CookedValue } else { $gs += $s.CookedValue } }
  }
  $G = 1GB
  "{0:N0},{1},{2:N2},{3:N2},{4:N2},{5:N2},{6:N2},{7:N2},{8:N2},{9:N2},{10:N2},{11:N2},{12:N2}" -f ((Get-Date) - $t0).TotalSeconds, $ph,
    ($(if ($e) { $e.WorkingSet64 } else { 0 }) / $G), ($(if ($e) { $e.PrivateMemorySize64 } else { 0 }) / $G),
    ($(if ($e) { $e.PeakWorkingSet64 } else { 0 }) / $G), ($(if ($a) { $a.WorkingSet64 } else { 0 }) / $G),
    ($(if ($a) { $a.PrivateMemorySize64 } else { 0 }) / $G), ($av / $G), ($sb / $G), ($mod / $G), ($cm / $G), ($gd / $G), ($gs / $G) |
    Out-File -Append -Encoding ascii $Csv
  Start-Sleep 2
}
